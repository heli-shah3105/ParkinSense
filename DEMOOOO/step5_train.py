"""
=============================================================================
Step 5: Training Pipeline - Per-Speaker & Generalized Adapters
=============================================================================
1. Trains ONE generalized LoRA adapter on all train-group speakers (Phase 1)
2. Trains a per-speaker LoRA adapter for each train-group speaker, warm-started
   from the generalized adapter (Phase 2)
3. Evaluates the generalized adapter on held-out (unseen) speakers
4. Evaluates each speaker adapter on that speaker's own test split

Metrics: Word Error Rate (WER) and Character Error Rate (CER)

Usage:
  python step5_train.py --mode general --smoke_test   # quick sanity check (minutes)
  python step5_train.py --mode general                # real Phase 1
  python step5_train.py --mode speaker                # real Phase 2 (all speakers)
  python step5_train.py --mode speaker --speaker_id F3_M01
  python step5_train.py --mode speaker --skip_existing   # resume after a crash

Key fixes vs. the previous version
----------------------------------
* Transcripts are UPPERCASED and sanitized to the characters that exist in the
  wav2vec2-base-960h vocabulary (A-Z and apostrophe). Lowercase text maps to
  <unk> and silently destroys training (WER stuck at 1.0).
* A label sanity check aborts the run if too many labels tokenize to <unk>.
* ctc_zero_infinity=True is forced on the model config (no inf/NaN CTC loss).
* bf16 is used when the GPU supports it, otherwise fp16.
* Over-long / impossible (too many characters for the duration) clips are
  dropped from training only, to avoid OOM and degenerate CTC targets.
* Saved speaker adapters are flattened so adapter_config.json always sits
  directly inside output/adapters/speaker_<ID>/ (PEFT otherwise nests
  warm-started adapters one folder deeper, which breaks loading in step 6).
* GPU memory is released between speakers.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Fix Windows/Anaconda OpenMP conflict

import sys
import gc
import re
import json
import shutil
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# Avoid UnicodeEncodeError on Windows consoles (cp1252)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import torch
except ImportError:
    conda_py = r"C:\Users\Heli P Shah\ANACONDA\envs\dysarthric\python.exe"
    if os.path.exists(conda_py) and sys.executable.lower() != conda_py.lower():
        import subprocess
        print(f"[*] Re-launching with conda 'dysarthric' environment: {conda_py}")
        res = subprocess.run([conda_py] + sys.argv)
        sys.exit(res.returncode)
    else:
        print("[!] PyTorch is not installed in current environment.")
        print("[!] Please run: conda activate dysarthric")
        sys.exit(1)

import numpy as np
import pandas as pd
import soundfile as sf
import jiwer

from transformers import (
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset

from config import (
    SAMPLE_RATE, MANIFEST_DIR, ADAPTER_DIR, LOGS_DIR, CHECKPOINTS_DIR,
    SPEAKER_ADAPTER_EPOCHS, SPEAKER_ADAPTER_BATCH_SIZE, SPEAKER_ADAPTER_LR,
    SPEAKER_ADAPTER_WEIGHT_DECAY, SPEAKER_ADAPTER_WARMUP_STEPS,
    SPEAKER_ADAPTER_GRADIENT_ACCUMULATION, SPEAKER_ADAPTER_MAX_GRAD_NORM,
    GENERAL_ADAPTER_EPOCHS, GENERAL_ADAPTER_BATCH_SIZE, GENERAL_ADAPTER_LR,
    GENERAL_ADAPTER_WEIGHT_DECAY, GENERAL_ADAPTER_WARMUP_STEPS,
    GENERAL_ADAPTER_GRADIENT_ACCUMULATION, GENERAL_ADAPTER_MAX_GRAD_NORM,
    EVAL_BATCH_SIZE, LOG_EVERY_N_STEPS,
    EARLY_STOPPING_PATIENCE, INIT_SPEAKER_FROM_GENERAL, SPEAKER_SEVERITY,
    RANDOM_SEED,
)
from step4_model_setup import (
    create_speaker_adapter_model,
    create_generalized_adapter_model,
    save_adapter,
    load_processor,
)

# ─────────────────────────────────────────────────────────────────────────────
# Local tuning knobs (kept here so config.py stays unchanged)
# ─────────────────────────────────────────────────────────────────────────────
USE_GRADIENT_CHECKPOINTING = True    # Enables gradient checkpointing to guarantee zero OOM
MAX_TRAIN_DURATION_SEC = 15.0        # Drop training clips longer than this (OOM guard)
MAX_CHARS_PER_SEC = 40               # Drop clips whose transcript is impossibly long for the audio
MAX_UNK_FRACTION = 0.05              # Abort if more than 5% of label tokens are <unk>

# Smoke-test sizes (python step5_train.py --smoke_test)
SMOKE_TRAIN_SAMPLES = 200
SMOKE_VAL_SAMPLES = 50
SMOKE_HELDOUT_SAMPLES = 100
SMOKE_ACCUMULATION = 4

# ─── Logging ────────────────────────────────────────────────────────────────
os.makedirs(LOGS_DIR, exist_ok=True)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if not root_logger.handlers:
    root_logger.addHandler(logging.StreamHandler())
_log_path = os.path.join(LOGS_DIR, "05_training.log")
if not any(isinstance(h, logging.FileHandler) and
           getattr(h, "baseFilename", "").endswith("05_training.log")
           for h in root_logger.handlers):
    file_handler = logging.FileHandler(_log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root_logger.addHandler(file_handler)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def precision_flags() -> Dict[str, bool]:
    """bf16 if the GPU supports it (more stable for CTC), else fp16, else fp32."""
    if not torch.cuda.is_available():
        return {"fp16": False, "bf16": False}
    try:
        if torch.cuda.is_bf16_supported():
            return {"fp16": False, "bf16": True}
    except Exception:
        pass
    return {"fp16": True, "bf16": False}


def harden_ctc(model):
    """Force ctc_zero_infinity so impossible alignments give 0 loss instead of inf/NaN."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base.config.ctc_zero_infinity = True
    return model


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def adapter_is_complete(path: str) -> bool:
    return (os.path.exists(os.path.join(path, "adapter_config.json")) and
            (os.path.exists(os.path.join(path, "adapter_model.safetensors")) or
             os.path.exists(os.path.join(path, "adapter_model.bin"))))


def flatten_adapter_dir(path: str):
    """
    PEFT saves any adapter whose name != 'default' into <path>/<adapter_name>/.
    Warm-started speaker adapters are named 'speaker_<ID>', so move their files
    up one level so that <path>/adapter_config.json exists (what step 6 expects).
    """
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return
    for sub in os.listdir(path):
        sub_path = os.path.join(path, sub)
        if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "adapter_config.json")):
            for fn in os.listdir(sub_path):
                dst = os.path.join(path, fn)
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(os.path.join(sub_path, fn), dst)
            os.rmdir(sub_path)
            logger.info(f"  Flattened nested adapter folder '{sub}' into {path}")
            return


def save_adapter_flat(model, adapter_name: str) -> str:
    """Save adapter (+ trainable CTC head) to ADAPTER_DIR/adapter_name with a flat layout."""
    path = os.path.join(ADAPTER_DIR, adapter_name)
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    save_adapter(model, ADAPTER_DIR, adapter_name)
    flatten_adapter_dir(path)
    if not adapter_is_complete(path):
        raise RuntimeError(f"Adapter save looks incomplete at {path}: {os.listdir(path)}")
    logger.info(f"  Verified adapter files at: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def calc_wer(predictions, references):
    return float(jiwer.wer(reference=list(references), hypothesis=list(predictions)))


def calc_cer(predictions, references):
    return float(jiwer.cer(reference=list(references), hypothesis=list(predictions)))


def compute_metrics(pred, processor):
    """Compute WER and CER from model predictions."""
    pred_logits = pred.predictions
    if isinstance(pred_logits, tuple):
        pred_logits = pred_logits[0]
    pred_ids = np.argmax(pred_logits, axis=-1)

    label_ids = np.array(pred.label_ids, copy=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    valid_pairs = [(p.strip(), l.strip()) for p, l in zip(pred_str, label_str) if l.strip()]
    if not valid_pairs:
        return {"wer": 1.0, "cer": 1.0}

    preds, refs = zip(*valid_pairs)
    return {"wer": calc_wer(preds, refs), "cer": calc_cer(preds, refs)}


# ─────────────────────────────────────────────────────────────────────────────
# Data Collator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataCollatorCTCWithPadding:
    """Dynamically pads audio inputs and labels (-100 for CTC padding)."""
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        max_len = max(len(f["labels"]) for f in features)
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)
        for i, f in enumerate(features):
            labels[i, : len(f["labels"])] = torch.tensor(f["labels"], dtype=torch.long)

        batch["labels"] = labels
        return batch


# ─────────────────────────────────────────────────────────────────────────────
# Transcript sanitation + dataset preparation
# ─────────────────────────────────────────────────────────────────────────────

def allowed_characters(processor) -> set:
    """Single-character tokens in the tokenizer vocab (A-Z and apostrophe), excluding '|'."""
    return {t for t in processor.tokenizer.get_vocab() if len(t) == 1 and t != "|"}


def sanitize_transcript(text, allowed: set) -> str:
    """UPPERCASE and replace any character the vocab can't represent with a space."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    t = str(text).upper()
    t = "".join(ch if (ch in allowed or ch == " ") else " " for ch in t)
    return re.sub(r"\s+", " ", t).strip()


def prepare_frame(df: pd.DataFrame, processor, train_mode: bool, max_n: Optional[int] = None) -> pd.DataFrame:
    """
    Clean a manifest slice:
      - drop missing audio files
      - sanitize/uppercase transcripts, drop empties
      - (train_mode) drop over-long clips and clips with impossibly long transcripts
      - (optional) subsample for smoke tests
    """
    df = df[df["audio_path"].apply(os.path.exists)].copy()
    allowed = allowed_characters(processor)
    df["transcript"] = df["transcript"].apply(lambda t: sanitize_transcript(t, allowed))
    df = df[df["transcript"].str.len() > 0]

    if train_mode and "duration_sec" in df.columns:
        before = len(df)
        df = df[df["duration_sec"] <= MAX_TRAIN_DURATION_SEC]
        df = df[df["transcript"].str.len() <= df["duration_sec"] * MAX_CHARS_PER_SEC]
        dropped = before - len(df)
        if dropped:
            logger.info(f"  Dropped {dropped} clips (too long / impossible transcript length)")

    if max_n is not None and len(df) > max_n:
        df = df.sample(n=max_n, random_state=RANDOM_SEED)

    return df.reset_index(drop=True)


def log_label_sanity(df: pd.DataFrame, processor, n: int = 200):
    """Abort early if labels are mostly <unk> (the failure that broke the first run)."""
    if len(df) == 0:
        return
    tok = processor.tokenizer
    sample = df["transcript"].sample(n=min(n, len(df)), random_state=RANDOM_SEED).tolist()
    total, unk = 0, 0
    for t in sample:
        ids = tok(t).input_ids
        total += len(ids)
        unk += sum(1 for i in ids if i == tok.unk_token_id)
    frac = unk / max(total, 1)
    logger.info(f"  Label sanity: {unk}/{total} tokens are <unk> ({frac*100:.2f}%)")
    logger.info(f"  Example transcript -> {sample[0]!r}")
    if frac > MAX_UNK_FRACTION:
        raise RuntimeError(
            f"{frac*100:.1f}% of label tokens are <unk>. Transcript text does not match the "
            f"tokenizer vocabulary - aborting instead of wasting hours of training."
        )


def prepare_dataset(batch, processor, augment: bool = False):
    """Process a single sample: load audio and tokenize transcript."""
    audio, sr = sf.read(batch["audio_path"])
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Acoustic data augmentation for training robustness on consumer microphones:
    if augment:
        # 1. Random gain variation (-4 dB to +2 dB)
        gain = 10.0 ** (np.random.uniform(-4.0, 2.0) / 20.0)
        audio = audio * gain
        # 2. Light ambient noise injection (40% probability, SNR 22-35 dB)
        if np.random.random() < 0.40:
            noise_std = np.std(audio) * np.random.uniform(0.005, 0.02)
            audio = audio + np.random.normal(0.0, max(noise_std, 1e-5), size=audio.shape)
        # 3. Soft clipping guard
        audio = np.clip(audio, -1.0, 1.0)

    input_values = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="np",
    ).input_values[0]

    # wav2vec2-base-960h vocabulary is UPPERCASE only
    transcript = str(batch["transcript"]).strip().upper()
    batch["input_values"] = input_values
    batch["labels"] = processor.tokenizer(transcript).input_ids
    return batch


def to_dataset(df: pd.DataFrame, processor, augment: bool = False) -> Dataset:
    cols = ["audio_path", "transcript", "speaker_id"]
    return Dataset.from_pandas(df[cols].reset_index(drop=True)).map(
        lambda x: prepare_dataset(x, processor, augment=augment),
        remove_columns=cols,
    )


def load_speaker_frames(speaker_id: str) -> Dict[str, pd.DataFrame]:
    """Return {'train': df, 'val': df, 'test': df} for one speaker."""
    df = pd.read_csv(os.path.join(MANIFEST_DIR, "split_manifest.csv"))
    df = df[df["speaker_id"] == speaker_id]
    return {s: df[df["split"] == s] for s in ["train", "val", "test"] if (df["split"] == s).any()}


# ─────────────────────────────────────────────────────────────────────────────
# Trainer construction
# ─────────────────────────────────────────────────────────────────────────────

def build_trainer(model, processor, train_ds, eval_ds, output_dir, hp) -> Trainer:
    prec = precision_flags()
    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=hp["epochs"],
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=hp["accum"],
        learning_rate=hp["lr"],
        weight_decay=hp["wd"],
        warmup_steps=hp["warmup"],
        max_grad_norm=hp["max_grad_norm"],
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="wer" if eval_ds is not None else None,
        greater_is_better=False if eval_ds is not None else None,
        logging_steps=LOG_EVERY_N_STEPS,
        dataloader_num_workers=0,        # Windows compatibility
        remove_unused_columns=False,
        report_to="none",
        **prec,
    )
    if USE_GRADIENT_CHECKPOINTING:
        kwargs["gradient_checkpointing"] = True
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    args = TrainingArguments(**kwargs)
    logger.info(f"  Precision: {prec} | grad checkpointing: {USE_GRADIENT_CHECKPOINTING} | "
                f"batch={hp['batch_size']} x accum={hp['accum']} | epochs={hp['epochs']}")

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        compute_metrics=lambda pred: compute_metrics(pred, processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
        if eval_ds is not None else [],
    )


def general_hp(smoke: bool) -> Dict:
    hp = dict(
        epochs=GENERAL_ADAPTER_EPOCHS, batch_size=GENERAL_ADAPTER_BATCH_SIZE,
        accum=GENERAL_ADAPTER_GRADIENT_ACCUMULATION, lr=GENERAL_ADAPTER_LR,
        wd=GENERAL_ADAPTER_WEIGHT_DECAY, warmup=GENERAL_ADAPTER_WARMUP_STEPS,
        max_grad_norm=GENERAL_ADAPTER_MAX_GRAD_NORM,
    )
    if smoke:
        hp.update(epochs=1, accum=SMOKE_ACCUMULATION, warmup=5)
    return hp


def speaker_hp(smoke: bool) -> Dict:
    hp = dict(
        epochs=SPEAKER_ADAPTER_EPOCHS, batch_size=SPEAKER_ADAPTER_BATCH_SIZE,
        accum=SPEAKER_ADAPTER_GRADIENT_ACCUMULATION, lr=SPEAKER_ADAPTER_LR,
        wd=SPEAKER_ADAPTER_WEIGHT_DECAY, warmup=SPEAKER_ADAPTER_WARMUP_STEPS,
        max_grad_norm=SPEAKER_ADAPTER_MAX_GRAD_NORM,
    )
    if smoke:
        hp.update(epochs=1, accum=SMOKE_ACCUMULATION, warmup=5)
    return hp


# ─────────────────────────────────────────────────────────────────────────────
# Training Functions
# ─────────────────────────────────────────────────────────────────────────────

def train_generalized_adapter(processor, smoke: bool = False):
    """Train the generalized LoRA adapter on ALL train-group speakers' data."""
    logger.info(f"\n{'='*70}")
    logger.info("TRAINING GENERALIZED ADAPTER" + (" [SMOKE TEST]" if smoke else ""))
    logger.info(f"{'='*70}")

    df = pd.read_csv(os.path.join(MANIFEST_DIR, "split_manifest.csv"))
    group = df[df["speaker_group"] == "train"]

    train_df = prepare_frame(group[group["split"] == "train"], processor, True,
                             SMOKE_TRAIN_SAMPLES if smoke else None)
    val_df = prepare_frame(group[group["split"] == "val"], processor, True,
                           SMOKE_VAL_SAMPLES if smoke else None)

    logger.info(f"  Generalized train samples: {len(train_df)} from {train_df['speaker_id'].nunique()} speakers")
    logger.info(f"  Generalized val samples:   {len(val_df)} from {val_df['speaker_id'].nunique()} speakers")
    log_label_sanity(train_df, processor)

    model = trainer = None
    try:
        model, _ = create_generalized_adapter_model()
        harden_ctc(model)

        train_ds = to_dataset(train_df, processor, augment=True)
        eval_ds = to_dataset(val_df, processor, augment=False) if len(val_df) > 0 else None

        name = "_smoke_generalized" if smoke else "generalized_adapter"
        trainer = build_trainer(model, processor, train_ds, eval_ds,
                                os.path.join(CHECKPOINTS_DIR, name), general_hp(smoke))

        results = {}
        if eval_ds is not None:
            base_metrics = trainer.evaluate(eval_ds)
            results["baseline_val_wer"] = base_metrics.get("eval_wer")
            results["baseline_val_cer"] = base_metrics.get("eval_cer")
            logger.info(f"  BASELINE (before training) val WER={results['baseline_val_wer']:.4f} "
                        f"CER={results['baseline_val_cer']:.4f}")

        train_result = trainer.train()
        results["train_loss"] = float(train_result.training_loss)
        logger.info(f"  Training loss: {train_result.training_loss:.4f}")

        if eval_ds is not None:
            final_metrics = trainer.evaluate(eval_ds)
            results["final_val_wer"] = final_metrics.get("eval_wer")
            results["final_val_cer"] = final_metrics.get("eval_cer")
            logger.info(f"  FINAL val WER={results['final_val_wer']:.4f} CER={results['final_val_cer']:.4f}")
            if results["final_val_wer"] is not None and results["final_val_wer"] >= 0.999:
                logger.error("  !! Val WER is ~1.0 - the adapter did NOT learn. Do not trust this run.")

        save_adapter_flat(trainer.model if trainer is not None else model, name)

        # Held-out (unseen) speakers
        heldout_df = prepare_frame(df[df["speaker_group"] == "heldout_test"], processor, False,
                                   SMOKE_HELDOUT_SAMPLES if smoke else None)
        if len(heldout_df) > 0:
            logger.info(f"\n  Evaluating on held-out speakers: {heldout_df['speaker_id'].unique().tolist()}")
            heldout_ds = to_dataset(heldout_df, processor)
            ev = trainer.evaluate(heldout_ds)
            results["heldout_wer"] = ev.get("eval_wer")
            results["heldout_cer"] = ev.get("eval_cer")
            results["heldout_samples"] = len(heldout_df)
            results["heldout_speakers"] = int(heldout_df["speaker_id"].nunique())
            logger.info(f"  Held-out WER: {results['heldout_wer']:.4f}")
            logger.info(f"  Held-out CER: {results['heldout_cer']:.4f}")
        return results
    finally:
        del model, trainer
        free_gpu()


def train_speaker_adapter(speaker_id, frames, processor, base_adapter_path=None, smoke: bool = False):
    """Train a LoRA adapter for one speaker, warm-started from the generalized adapter."""
    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING SPEAKER ADAPTER: {speaker_id}" + (" [SMOKE TEST]" if smoke else ""))
    logger.info(f"{'='*70}")

    if "train" not in frames or len(frames["train"]) == 0:
        logger.warning(f"  No training data for speaker {speaker_id}, skipping.")
        return None

    if base_adapter_path is None and INIT_SPEAKER_FROM_GENERAL:
        default_gen = os.path.join(ADAPTER_DIR, "generalized_adapter")
        if adapter_is_complete(default_gen):
            base_adapter_path = default_gen
    if base_adapter_path:
        logger.info(f"  Warm-starting from: {base_adapter_path}")

    train_df = prepare_frame(frames["train"], processor, True, SMOKE_TRAIN_SAMPLES if smoke else None)
    val_df = prepare_frame(frames["val"], processor, True, SMOKE_VAL_SAMPLES if smoke else None) \
        if "val" in frames else pd.DataFrame()
    test_df = prepare_frame(frames["test"], processor, False, SMOKE_VAL_SAMPLES if smoke else None) \
        if "test" in frames else pd.DataFrame()

    if len(train_df) == 0:
        logger.warning(f"  No usable training clips for {speaker_id} after cleaning, skipping.")
        return None
    log_label_sanity(train_df, processor, n=100)

    model = trainer = None
    try:
        model, _ = create_speaker_adapter_model(speaker_id, base_adapter_path=base_adapter_path)
        harden_ctc(model)

        train_ds = to_dataset(train_df, processor, augment=True)
        eval_ds = to_dataset(val_df, processor, augment=False) if len(val_df) > 0 else None

        name = f"_smoke_speaker_{speaker_id}" if smoke else f"speaker_{speaker_id}"
        trainer = build_trainer(model, processor, train_ds, eval_ds,
                                os.path.join(CHECKPOINTS_DIR, name), speaker_hp(smoke))

        logger.info(f"  Training samples: {len(train_ds)}"
                    + (f" | Validation samples: {len(eval_ds)}" if eval_ds is not None else ""))

        train_result = trainer.train()
        logger.info(f"  Training loss: {train_result.training_loss:.4f}")

        save_adapter_flat(trainer.model if trainer is not None else model, name)

        results = {
            "speaker_id": speaker_id,
            "train_loss": float(train_result.training_loss),
            "train_samples": len(train_ds),
        }
        if len(test_df) > 0:
            test_ds = to_dataset(test_df, processor)
            ev = trainer.evaluate(test_ds)
            results["test_wer"] = ev.get("eval_wer")
            results["test_cer"] = ev.get("eval_cer")
            results["test_samples"] = len(test_df)
            logger.info(f"  Test WER: {results['test_wer']:.4f}")
            logger.info(f"  Test CER: {results['test_cer']:.4f}")
        return results
    finally:
        del model, trainer
        free_gpu()


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_full_training(mode="all", target_speaker=None, smoke=False, skip_existing=False):
    """
    Phase 1: generalized adapter on pooled train-speaker data (+ held-out eval)
    Phase 2: per-speaker adapters, warm-started from the generalized adapter

    mode: 'all' | 'general' | 'speaker'
    """
    logger.info("=" * 70)
    logger.info("STEP 5: TRAINING PIPELINE")
    logger.info(f"Mode: {mode.upper()} | Target Speaker: {target_speaker or 'ALL'} | "
                f"Smoke: {smoke} | Skip existing: {skip_existing}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | "
                    f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("CUDA NOT AVAILABLE - training on CPU will be extremely slow.")
    logger.info("=" * 70)

    processor = load_processor()

    manifest_path = os.path.join(MANIFEST_DIR, "split_manifest.csv")
    if not os.path.exists(manifest_path):
        logger.error(f"Split manifest not found: {manifest_path}. Run step3_split_speakers.py first!")
        return

    df = pd.read_csv(manifest_path)
    train_speakers = sorted(df[df["speaker_group"] == "train"]["speaker_id"].unique().tolist())
    heldout_speakers = sorted(df[df["speaker_group"] == "heldout_test"]["speaker_id"].unique().tolist())
    logger.info(f"Train speakers ({len(train_speakers)}): {train_speakers}")
    logger.info(f"Held-out speakers ({len(heldout_speakers)}): {heldout_speakers}")

    generalized_adapter_path = os.path.join(ADAPTER_DIR, "generalized_adapter")
    phase1_ok = True

    # ─── PHASE 1 ────────────────────────────────────────────────────────
    if mode in ("all", "general"):
        logger.info("\n" + "#" * 70)
        logger.info("#  PHASE 1: TRAIN GENERALIZED ADAPTER (Pooled Train-Speaker Data)")
        logger.info("#" * 70)
        try:
            general_results = train_generalized_adapter(processor, smoke=smoke)
            fname = "generalized_adapter_results_smoke.json" if smoke else "generalized_adapter_results.json"
            out = os.path.join(LOGS_DIR, fname)
            with open(out, "w") as f:
                json.dump(general_results, f, indent=2, default=str)
            logger.info(f"\nGeneralized adapter results saved to: {out}")
        except Exception as e:
            phase1_ok = False
            logger.error(f"Failed to train generalized adapter: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # ─── PHASE 2 ────────────────────────────────────────────────────────
    if mode in ("all", "speaker"):
        if mode == "all" and not phase1_ok:
            logger.error("Skipping Phase 2 because Phase 1 failed (speaker adapters would warm-start from nothing).")
        else:
            logger.info("\n" + "#" * 70)
            logger.info("#  PHASE 2: TRAIN PER-SPEAKER ADAPTERS (Warm-Started from Generalized)")
            logger.info("#" * 70)

            speakers_to_train = [target_speaker] if target_speaker else train_speakers
            warm_start_path = generalized_adapter_path if adapter_is_complete(generalized_adapter_path) else None
            if warm_start_path:
                logger.info(f"[OK] Using generalized adapter for warm-starting: {warm_start_path}")
            else:
                logger.warning("Generalized adapter not found - speaker adapters start from fresh LoRA weights")

            all_results, failed = [], []
            for speaker_id in speakers_to_train:
                if speaker_id not in df["speaker_id"].unique():
                    logger.warning(f"Speaker '{speaker_id}' not found in manifest, skipping.")
                    continue

                spk_path = os.path.join(ADAPTER_DIR, f"speaker_{speaker_id}")
                if skip_existing and not smoke and adapter_is_complete(spk_path):
                    logger.info(f"Skipping {speaker_id}: adapter already exists at {spk_path}")
                    continue

                try:
                    result = train_speaker_adapter(
                        speaker_id=speaker_id,
                        frames=load_speaker_frames(speaker_id),
                        processor=processor,
                        base_adapter_path=warm_start_path,
                        smoke=smoke,
                    )
                    if result:
                        result["severity"] = SPEAKER_SEVERITY.get(speaker_id, "Unknown")
                        all_results.append(result)
                except Exception as e:
                    failed.append(speaker_id)
                    logger.error(f"Failed to train adapter for {speaker_id}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    free_gpu()

            if all_results:
                fname = "speaker_adapter_results_smoke.csv" if smoke else "speaker_adapter_results.csv"
                out = os.path.join(LOGS_DIR, fname)
                # Merge with earlier results so resumed runs don't overwrite finished speakers
                new_df = pd.DataFrame(all_results)
                if os.path.exists(out) and not smoke:
                    old_df = pd.read_csv(out)
                    new_df = pd.concat([old_df[~old_df["speaker_id"].isin(new_df["speaker_id"])], new_df])
                new_df.to_csv(out, index=False)
                logger.info(f"\nPer-speaker results saved to: {out}")
            if failed:
                logger.error(f"Speakers that FAILED: {failed}  (re-run with --mode speaker --skip_existing)")

    # ─── Final Summary ──────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("TRAINING PIPELINE FINISHED (check the WER numbers above - finished != successful)")
    logger.info(f"{'='*70}")
    logger.info(f"  Generalized adapter:  {ADAPTER_DIR}/generalized_adapter")
    logger.info(f"  Per-speaker adapters: {ADAPTER_DIR}/speaker_<SPEAKER_ID>")
    logger.info("  python step6_inference.py --audio <audio.wav>                    # Generalized")
    logger.info("  python step6_inference.py --audio <audio.wav> --speaker_id <id>  # Personalized")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dysarthric ASR Training Pipeline")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "general", "speaker"],
                        help="Training mode: 'all', 'general', or 'speaker'")
    parser.add_argument("--speaker_id", type=str, default=None,
                        help="Target speaker ID when mode is 'speaker'")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Tiny 1-epoch run on a small subset to verify WER drops below 1.0")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip speakers whose adapter already exists (resume after a crash)")
    args = parser.parse_args()

    run_full_training(mode=args.mode, target_speaker=args.speaker_id,
                      smoke=args.smoke_test, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()