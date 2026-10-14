"""
train_alignment.py — AfriSafeLM Hausa Safety Alignment
=======================================================

Pipeline per regime (G = Gold-only, GS = Gold+Silver):

    Qwen2.5-7B-Instruct (BASE_MODEL)
          │
    ┌─────▼────────────────────────────┐
    │  QLoRA SFT                       │
    │  prompt_ha → chosen_ha           │
    │  ALL rows: benign + malicious    │
    └─────┬────────────────────────────┘
          │  outputs/SFT-{G|GS}/  (LoRA adapter only)
          │
    ┌─────▼────────────────────────────────────────────────────┐
    │  In-memory merge  (no disk I/O)                          │
    │  policy  = 4-bit base  (will receive fresh DPO LoRA)     │
    │  ref     = 4-bit base + SFT adapter merged in memory     │
    │  Both models built by load_sft_as_policy_and_ref()       │
    └─────┬────────────────────────────────────────────────────┘
          │
    ┌─────▼────────────────────────────┐
    │  QLoRA DPO  (fresh adapter)      │
    │  ref_model = frozen merged SFT   │
    │  chosen_ha  vs  rejected_ha      │
    └─────┬────────────────────────────┘
          │
    outputs/DPO-{G|GS}/  ← final checkpoint

Usage:
    python train_alignment.py

Edit the CONFIG section below before running.
"""

# ============================================================
# 0. CONFIG — edit here
# ============================================================

GOLD_XLSX_PATH   = "data/gold.xlsx"
SILVER_XLSX_PATH = "data/silver.xlsx"
OUTPUT_DIR       = "outputs"

HF_REPO_ID  = "AfroNLP/AfrisafeLM"   #  HF repo
HF_TOKEN    = ""   # HF token 

BASE_MODEL  = "Qwen/Qwen2.5-7B-Instruct"

# LoRA
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Training
SFT_LR          = 1e-4
DPO_LR          = 1e-5      # deliberately lower — DPO refines, not imitates
SFT_EPOCHS      = 2
DPO_EPOCHS      = 1
BATCH_SIZE      = 2
GRAD_ACCUM      = 8
MAX_SEQ_LEN     = 2048
MAX_PROMPT_LEN  = 512       # max prompt tokens in DPO pairs (rest = completion budget)
DPO_BETA        = 0.1
WARMUP_STEPS    = 100

# Smoke-test: set True to run on SMOKE_N rows before committing to full training
SMOKE_TEST      = False
SMOKE_N         = 500

# ============================================================
# 1. ENVIRONMENT SETUP — must run before any HF/torch imports
# ============================================================

import os
import gc

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_HF_CACHE    = os.path.join(_PROJECT_DIR, "cache", "huggingface")
os.environ.setdefault("HF_HOME",           _HF_CACHE)
os.environ.setdefault("HF_HUB_CACHE",      os.path.join(_HF_CACHE, "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_HF_CACHE, "datasets"))

import json
import random
import re
from collections import defaultdict

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from huggingface_hub import HfApi
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(42)


# ============================================================
# 2. FILE HELPERS
# ============================================================

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ============================================================
# 3. DATA LOADING
# ============================================================

def parse_harm_categories(val):
    """
    Parse numpy-style string arrays stored in macro_harm_category.
    e.g.  "['violence_physical_harm' 'hate_speech_discrimination']"
       →  ['violence_physical_harm', 'hate_speech_discrimination']

    Returns [] for absent / NaN / empty-list values.
    Callers treat [] as "benign".
    """
    if pd.isna(val) or str(val).strip() in ("", "[]", "nan"):
        return []
    return re.findall(r"'([^']+)'", str(val))


def load_and_normalize_excel(path):
    """
    Load and clean a PKU-SafeRLHF-style Excel file with Hausa columns.

    Required columns : prompt_ha, chosen_ha
    Optional columns : rejected_ha, macro_harm_category

    PROMPTS ARE MIXED — benign AND malicious.
    - macro_harm_category absent or empty → prompt tagged as "benign"
    - Rows without rejected_ha (or where chosen == rejected) are valid for
      SFT only.  They are flagged has_rejected=False and excluded from DPO.

    WHAT CHANGED vs. original:
    - rejected_ha and macro_harm_category are NO LONGER in the required-column
      check or the dropna call.  The old code silently dropped every benign row
      (NaN harm category), losing a large fraction of training data.
    """
    df = pd.read_excel(path, engine="openpyxl")

    # Only the two columns we absolutely need to train on
    required = {"prompt_ha", "chosen_ha"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df = df.dropna(subset=["prompt_ha", "chosen_ha"])
    df = df.reset_index(drop=True)

    # Fill optional columns so the loop below doesn't KeyError
    if "rejected_ha"         not in df.columns:
        df["rejected_ha"]         = None
    if "macro_harm_category" not in df.columns:
        df["macro_harm_category"] = None

    records   = []
    n_dropped = 0

    for _, row in df.iterrows():
        prompt  = str(row["prompt_ha"]).strip()
        chosen  = str(row["chosen_ha"]).strip()

        rej_raw  = row["rejected_ha"]
        rejected = str(rej_raw).strip() if pd.notna(rej_raw) else ""

        # Silently skip truly empty rows (shouldn't happen after dropna, but be safe)
        if not prompt or not chosen:
            n_dropped += 1
            continue

        # A row is valid for DPO only when rejected is non-empty AND differs from chosen.
        # chosen == rejected gives zero preference signal → degenerate DPO loss.
        has_rejected = bool(rejected) and (rejected != chosen)

        cats = parse_harm_categories(row["macro_harm_category"])
        if not cats:
            cats = ["benign"]   # benign prompt: no harm tag in source data

        records.append({
            "prompt_ha":       prompt,
            "chosen_ha":       chosen,
            "rejected_ha":     rejected,
            "has_rejected":    has_rejected,
            "harm_categories": cats,
            "primary_category": cats[0],
        })

    if n_dropped:
        print(f"  ⚠  Dropped {n_dropped} empty rows in {path}")

    n_sft = len(records)
    n_dpo = sum(1 for r in records if r["has_rejected"])
    print(f"  {os.path.basename(path)}: {n_sft} rows total  "
          f"({n_dpo} usable for DPO,  {n_sft - n_dpo} SFT-only / benign)")
    return records


# ============================================================
# 4. STRATIFIED SPLIT
# ============================================================

def stratified_split(data, test_frac=0.2, min_stratum_size=5):
    """
    Stratified train/test split on the FULL frozenset of harm categories.

    Benign prompts land in the "benign" stratum and are split like any other.
    Strata smaller than min_stratum_size are merged into their rarest
    single-category bucket to avoid zero-size test sets.
    """
    # Count rows per individual category (for merge-target selection)
    cat_totals = defaultdict(int)
    for ex in data:
        for c in ex["harm_categories"]:
            cat_totals[c] += 1

    def rarest_cat(cats):
        return min(cats, key=lambda c: cat_totals.get(c, 0))

    # Bucket every row by its full category frozenset
    combo_buckets = defaultdict(list)
    for ex in data:
        key = frozenset(ex["harm_categories"]) if ex["harm_categories"] \
              else frozenset({"benign"})
        combo_buckets[key].append(ex)

    # Merge tiny strata into their rarest single-category bucket
    single_buckets = defaultdict(list)
    final_buckets  = {}
    for key, items in combo_buckets.items():
        if len(items) < min_stratum_size:
            fallback = rarest_cat(key)
            single_buckets[fallback].extend(items)
        else:
            final_buckets[key] = items
    for cat, items in single_buckets.items():
        fkey = frozenset({cat})
        final_buckets.setdefault(fkey, [])
        final_buckets[fkey].extend(items)

    train, test = [], []
    print(f"  {'Stratum':<55} {'total':>6}  {'train':>6}  {'test':>5}")
    print("  " + "-" * 77)
    for key in sorted(final_buckets.keys(), key=lambda k: sorted(k)):
        items = final_buckets[key]
        random.shuffle(items)
        cut   = max(1, int(len(items) * test_frac))
        test  += items[:cut]
        train += items[cut:]
        label  = "+".join(sorted(key))
        print(f"  {label:<55} {len(items):>6}  {len(items)-cut:>6}  {cut:>5}")

    print(f"\n  Total → train={len(train)}  test={len(test)}")
    return train, test


# ============================================================
# 5. DATA PREPARATION  (Step 1)
# ============================================================

def prepare_data():
    print("\n" + "="*60)
    print("STEP 1 — Data preparation")
    print("="*60)

    gold_raw   = load_and_normalize_excel(GOLD_XLSX_PATH)
    silver_raw = load_and_normalize_excel(SILVER_XLSX_PATH)

    print(f"\nGold loaded:   {len(gold_raw)} rows")
    print(f"Silver loaded: {len(silver_raw)} rows")

    print("\nGold split:")
    gold_train,   gold_test   = stratified_split(gold_raw)
    print("\nSilver split:")
    silver_train, silver_test = stratified_split(silver_raw)

    gs_train = gold_train + silver_train

    data_dir = os.path.join(OUTPUT_DIR, "processed_data")
    os.makedirs(data_dir, exist_ok=True)
    save_jsonl(gold_train,   f"{data_dir}/train_gold.jsonl")
    save_jsonl(gold_test,    f"{data_dir}/test_gold.jsonl")
    save_jsonl(silver_train, f"{data_dir}/train_silver.jsonl")
    save_jsonl(silver_test,  f"{data_dir}/test_silver.jsonl")
    save_jsonl(gs_train,     f"{data_dir}/train_gold_silver.jsonl")

    print(f"\nGold   train={len(gold_train)}  test={len(gold_test)}")
    print(f"Silver train={len(silver_train)}  test={len(silver_test)}")
    print(f"G+S    train={len(gs_train)}")
    print(f"Splits saved → {data_dir}")

    return gold_train, gold_test, silver_train, silver_test, gs_train


# ============================================================
# 6. TOKENIZER  (Step 2)
# ============================================================

def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tok.pad_token        = tok.eos_token   # Qwen2.5 has no dedicated pad token
    tok.pad_token_id     = tok.eos_token_id  # explicit id avoids silent mismatch
    tok.padding_side     = "right"         # default; overridden to "left" inside run_dpo
    tok.model_max_length = MAX_SEQ_LEN
    return tok


# ============================================================
# 7. DATASET FORMATTING  (Step 3)
# ============================================================

def build_sft_dataset(data, tokenizer):
    """
    SFT format: every row (benign + malicious), prompt → chosen.

    Full conversation rendered via apply_chat_template so the model sees
    the correct special tokens for user/assistant turns.
    """
    def fmt(ex):
        msgs = [
            {"role": "user",      "content": ex["prompt_ha"]},
            {"role": "assistant", "content": ex["chosen_ha"]},
        ]
        return {"text": tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )}

    return Dataset.from_list([fmt(ex) for ex in data])


def build_dpo_dataset(data, tokenizer):
    """
    DPO format: only rows with a valid (chosen, rejected) pair.

    WHY we extract completions via apply_chat_template:
      Qwen2.5's in-turn terminator is <|im_end|>, NOT <|endoftext|>.
      The old code appended tokenizer.eos_token (<|endoftext|>) to chosen/rejected.
      That trained the model to emit <|endoftext|> mid-conversation, breaking
      inference.  By rendering the full [user, assistant] conversation and slicing
      off the prompt prefix, we get the completion with the correct <|im_end|>.

    DPOTrainer expects:
        prompt   — formatted up to the assistant turn opening
        chosen   — assistant completion including end-of-turn token
        rejected — same for the dispreferred response
    """
    dpo_rows = [ex for ex in data if ex["has_rejected"]]
    if not dpo_rows:
        raise ValueError(
            "No valid DPO rows found — all rejected_ha values are missing or "
            "identical to chosen_ha.  Check your data."
        )

    def fmt(ex):
        # Prompt: up to and including the assistant-turn opener
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["prompt_ha"]}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Full conversation → extract completion by stripping the prompt prefix
        def render_full(response):
            full = tokenizer.apply_chat_template(
                [{"role": "user",      "content": ex["prompt_ha"]},
                 {"role": "assistant", "content": response}],
                tokenize=False,
                add_generation_prompt=False,
            )
            # Token-based slicing — safer than string slicing which can silently
            # misalign when chat-template whitespace varies by tokenizer version.
            prompt_ids   = tokenizer(prompt_text,  add_special_tokens=False).input_ids
            full_ids     = tokenizer(full,          add_special_tokens=False).input_ids
            response_ids = full_ids[len(prompt_ids):]
            return tokenizer.decode(response_ids, skip_special_tokens=False)

        return {
            "prompt":   prompt_text,
            "chosen":   render_full(ex["chosen_ha"]),
            "rejected": render_full(ex["rejected_ha"]),
        }

    return Dataset.from_list([fmt(ex) for ex in dpo_rows])


def build_datasets(gold_train, gs_train, tokenizer):
    """Construct all four HF Dataset objects (sft_G, sft_GS, dpo_G, dpo_GS)."""
    if SMOKE_TEST:
        print(f"\n⚠  SMOKE TEST — capping each split to {SMOKE_N} rows")
        gold_train = gold_train[:SMOKE_N]
        gs_train   = gs_train[:SMOKE_N]

    sft_G  = build_sft_dataset(gold_train, tokenizer)
    sft_GS = build_sft_dataset(gs_train,   tokenizer)
    dpo_G  = build_dpo_dataset(gold_train, tokenizer)
    dpo_GS = build_dpo_dataset(gs_train,   tokenizer)

    print(f"\n  sft_G  : {len(sft_G)}  rows  (all prompts)")
    print(f"  sft_GS : {len(sft_GS)}  rows  (all prompts)")
    print(f"  dpo_G  : {len(dpo_G)}  rows  (valid pairs only)")
    print(f"  dpo_GS : {len(dpo_GS)}  rows  (valid pairs only)")

    return {"sft_G": sft_G, "sft_GS": sft_GS, "dpo_G": dpo_G, "dpo_GS": dpo_GS}


# ============================================================
# 8. MODEL UTILITIES  (Step 4)
# ============================================================

def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config():
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_base_model_for_sft():
    """
    Load quantized base model ready for QLoRA SFT.
    enable_input_require_grads() is required so gradient checkpointing
    can propagate gradients through the frozen quantized layers.
    """
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=get_bnb_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache     = False
    model.config.pad_token_id  = model.config.eos_token_id  # sync with tokenizer
    model.enable_input_require_grads()
    return model


def load_sft_as_policy_and_ref(sft_ckpt_path):
    """
    Skip the slow disk-save merge entirely.  Instead:

      1. Load the base model in 4-bit (policy model — will receive fresh DPO LoRA).
      2. Load the base model in 4-bit again (reference model).
      3. Apply the SFT LoRA to the reference model and call merge_and_unload()
         in memory so reference weights = merged SFT.

    Both models stay in memory simultaneously.  For a 7B model this is roughly
    2 × ~5 GB (4-bit) = ~10 GB VRAM, less than the 14 GB needed to save/reload
    a bf16 merged checkpoint.

    DPOTrainer receives ref_model explicitly, so it never tries to derive the
    reference by disabling the policy LoRA — the KL baseline is always the correct
    SFT-aligned model.

    WHY not pass a PeftModel directly as ref_model:
      If ref_model is a PeftModel with the adapter enabled, DPOTrainer will call
      forward() through LoRA weights on BOTH passes.  merge_and_unload() bakes the
      SFT delta into the base weights and removes the adapter, giving a plain
      AutoModelForCausalLM that DPOTrainer treats as a frozen reference with no
      adapter overhead.
    """
    print(f"\nLoading policy + reference models from SFT checkpoint (no disk merge)…")
    print(f"  SFT adapter : {sft_ckpt_path}")

    bnb = get_bnb_config()

    # ── Policy model (4-bit, will get fresh DPO LoRA from DPOTrainer) ──────────
    policy = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    policy.config.use_cache    = False
    policy.config.pad_token_id = policy.config.eos_token_id
    policy.enable_input_require_grads()
    policy = prepare_model_for_kbit_training(policy, use_gradient_checkpointing=False)
    policy.gradient_checkpointing_disable()

    # ── Reference model (4-bit base + SFT adapter merged in memory) ────────────
    ref_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    ref_base.config.use_cache    = False
    ref_base.config.pad_token_id = ref_base.config.eos_token_id

    peft_ref  = PeftModel.from_pretrained(ref_base, sft_ckpt_path)
    reference = peft_ref.merge_and_unload()   # in-memory merge — no disk I/O
    # Freeze reference; DPOTrainer expects it to require no grad.
    for p in reference.parameters():
        p.requires_grad_(False)

    print("✓ Policy model  : 4-bit base, ready for DPO LoRA")
    print("✓ Reference model: 4-bit base + SFT adapter merged in memory (frozen)")
    return policy, reference


def verify_trainable_params(model, label):
    """Print trainable vs. total parameter counts for a sanity check."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct       = 100.0 * trainable / total if total > 0 else 0.0
    print(f"  [{label}] trainable: {trainable:,} / {total:,}  ({pct:.3f}%)")


def free_memory():
    """Flush Python GC and empty the CUDA allocator cache."""
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# 9. SFT  (Step 5)
# ============================================================

def run_sft(dataset, regime_tag, tokenizer):
    out = os.path.join(OUTPUT_DIR, f"SFT-{regime_tag}")
    print(f"\n{'='*60}\nSFT-{regime_tag}  ({len(dataset)} examples)\n{'='*60}")

    model = load_base_model_for_sft()

    cfg = SFTConfig(
        output_dir=out,
        num_train_epochs=SFT_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=SFT_LR,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        bf16=True,
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="text",
        # max_seq_length removed: older TRL versions crash on this arg.
        # Truncation is handled by tok.model_max_length set in load_tokenizer().
        packing=False,                        # keep chat-template boundaries intact
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        peft_config=get_lora_config(),
        processing_class=tokenizer,
    )

    verify_trainable_params(trainer.model, f"SFT-{regime_tag}")
    trainer.train()
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    print(f"✓ SFT-{regime_tag} saved → {out}")

    del trainer, model
    free_memory()

    push_checkpoint(
        f"SFT-{regime_tag}", out,
        branch="main" if regime_tag == "G" else f"sft-{regime_tag.lower()}"
    )
    return out


# ============================================================
# 10. MERGE  (Step 6) — now done in-memory inside run_dpo; no disk I/O
# ============================================================
# run_merge() has been removed.  The heavy bf16 disk-save + reload cycle
# (≈ 14 GB write + 14 GB read) is replaced by an in-memory merge inside
# load_sft_as_policy_and_ref().  The two 4-bit models together use less VRAM
# than the old bf16 load and avoid all disk I/O.


# ============================================================
# 11. DPO  (Step 7)
# ============================================================

def run_dpo(dataset, sft_ckpt_path, regime_tag, tokenizer):
    """
    Run DPO without a prior disk-merge step.

    load_sft_as_policy_and_ref() builds both models in memory:
      • policy  — 4-bit base, receives a fresh LoRA from DPOTrainer
      • reference — 4-bit base with SFT adapter merged in memory (frozen)

    Passing ref_model explicitly ensures DPOTrainer never tries to derive the
    reference by disabling the policy LoRA (which would give the wrong baseline).
    """
    out = os.path.join(OUTPUT_DIR, f"DPO-{regime_tag}")
    print(f"\n{'='*60}\nDPO-{regime_tag}  ({len(dataset)} examples)\n{'='*60}")

    # DPO's generation-style forward pass requires left-padding
    tokenizer.padding_side = "left"

    policy, reference = load_sft_as_policy_and_ref(sft_ckpt_path)

    cfg = DPOConfig(
    output_dir=out,
    num_train_epochs=DPO_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=DPO_LR,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,
    bf16=True,
    optim="adamw_torch",
    weight_decay=0.01,
    max_grad_norm=1.0,
    beta=DPO_BETA,
    save_strategy="epoch",
    logging_steps=10,
    report_to="none",
    gradient_checkpointing=False
	)

    trainer = DPOTrainer(
        model=policy,
        ref_model=reference,                 # explicit frozen SFT reference — correct KL baseline
        args=cfg,
        train_dataset=dataset,
        peft_config=get_lora_config(),       # fresh LoRA on top of policy
        processing_class=tokenizer,
    )

    verify_trainable_params(trainer.model, f"DPO-{regime_tag}")
    trainer.train()
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    print(f"✓ DPO-{regime_tag} saved → {out}")

    tokenizer.padding_side = "right"   # restore for any downstream use
    del trainer, policy, reference
    free_memory()

    push_checkpoint(f"DPO-{regime_tag}", out, branch=f"dpo-{regime_tag.lower()}")
    return out


# ============================================================
# 12. HF HUB PUSH
# ============================================================

def push_checkpoint(name, local_path, branch):
    """
    Push a checkpoint folder to a specific branch of HF_REPO_ID.
    Silently skips (with a warning) when no token is configured.
    """
    token = HF_TOKEN or os.environ.get("HF_TOKEN", "")
    if not token:
        print(f"  ⚠  No HF_TOKEN found — skipping push for {name}")
        return

    api = HfApi(token=token)
    api.create_repo(repo_id=HF_REPO_ID, repo_type="model", private=True, exist_ok=True)
    if branch != "main":
        try:
            api.create_branch(repo_id=HF_REPO_ID, repo_type="model", branch=branch)
        except Exception:
            pass  # branch already exists — fine

    print(f"  ↑ Pushing {name} → {HF_REPO_ID}@{branch} ...")
    api.upload_folder(
        repo_id=HF_REPO_ID,
        repo_type="model",
        folder_path=local_path,
        path_in_repo="",
        revision=branch,
        commit_message=f"Add {name} checkpoint",
        ignore_patterns=["checkpoint-*/"],
    )
    print(f"  ✓ {name} → https://huggingface.co/{HF_REPO_ID}/tree/{branch}")


# ============================================================
# 13. MAIN
# ============================================================

def main():
    print(f"CUDA available : {torch.cuda.is_available()}")
    print(f"Device count   : {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU            : {torch.cuda.get_device_name(0)}")
        print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Step 1: Load and split data ──────────────────────────
    gold_train, gold_test, silver_train, silver_test, gs_train = prepare_data()

    # ── Step 2: Tokenizer ────────────────────────────────────
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer()

    # ── Step 3: Build HF datasets ────────────────────────────
    print("\nBuilding datasets...")
    datasets = build_datasets(gold_train, gs_train, tokenizer)

    checkpoints = {}

    # ═══════════════════════════════════════════════════════════
    # REGIME G  (Gold-only)
    # ═══════════════════════════════════════════════════════════
    # SFT
    checkpoints["SFT-G"] = run_sft(datasets["sft_G"], "G", tokenizer)
    # DPO — merge happens in-memory inside run_dpo (no disk save)
    checkpoints["DPO-G"] = run_dpo(
        datasets["dpo_G"], checkpoints["SFT-G"], "G", tokenizer
    )

    # ═══════════════════════════════════════════════════════════
    # REGIME GS  (Gold + Silver)
    # ═══════════════════════════════════════════════════════════
    checkpoints["SFT-GS"] = run_sft(datasets["sft_GS"], "GS", tokenizer)
    checkpoints["DPO-GS"] = run_dpo(
        datasets["dpo_GS"], checkpoints["SFT-GS"], "GS", tokenizer
    )

    # ── Save checkpoint registry for evaluate.py ─────────────
    ckpt_path = os.path.join(OUTPUT_DIR, "checkpoints.json")
    with open(ckpt_path, "w") as f:
        json.dump(checkpoints, f, indent=2)
    print(f"\n✓ Checkpoint registry saved → {ckpt_path}")
    print("\nAll done.  Run evaluate.py next.")


if __name__ == "__main__":
    main()
