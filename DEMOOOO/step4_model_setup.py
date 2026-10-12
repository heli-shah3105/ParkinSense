import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Fix Windows/Anaconda OpenMP conflict

"""
=============================================================================
Step 4: Model Setup — Base Wav2Vec2 + LoRA Adapter Injection
=============================================================================
1. Loads the pretrained Wav2Vec2 model (facebook/wav2vec2-base-960h)
2. Freezes ALL base model weights (encoder + feature extractor)
3. Injects LoRA adapters into attention/projection layers using PEFT
4. Keeps the CTC head trainable for task adaptation
5. Provides functions to create per-speaker and generalized adapters

Adapter architecture:
  - LoRA rank 16, alpha 32, applied to q_proj, k_proj, v_proj, out_proj,
    intermediate_dense, output_dense in the Wav2Vec2 encoder transformer layers
  - CTC linear head remains trainable
"""

import sys
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import torch
    from transformers import (
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
    )
except ImportError:
    conda_py = r"C:\Users\Heli P Shah\ANACONDA\envs\dysarthric\python.exe"
    if os.path.exists(conda_py) and sys.executable.lower() != conda_py.lower():
        import subprocess
        res = subprocess.run([conda_py] + sys.argv)
        sys.exit(res.returncode)
    else:
        print("[!] Required ML libraries not found. Run: conda activate dysarthric")
        sys.exit(1)

import logging

from config import (
    BASE_MODEL_NAME, FREEZE_BASE_WEIGHTS,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    ADAPTER_DIR, LOGS_DIR, CHECKPOINTS_DIR,
)

# ─── Logging ────────────────────────────────────────────────────────────────
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(ADAPTER_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "04_model_setup.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_processor():
    """Load the Wav2Vec2 processor (feature extractor + tokenizer)."""
    logger.info(f"Loading processor from: {BASE_MODEL_NAME}")
    processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL_NAME)
    return processor


def load_base_model():
    """
    Load the pretrained Wav2Vec2ForCTC model.
    This includes the encoder + CTC head.
    """
    logger.info(f"Loading base model from: {BASE_MODEL_NAME}")
    model = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL_NAME,
        ctc_loss_reduction="mean",
        pad_token_id=0,
    )
    model = Wav2Vec2ForCTC.from_pretrained(
    BASE_MODEL_NAME,
    ctc_loss_reduction="mean",
    ctc_zero_infinity=True,   # prevents inf/NaN loss on very short clips
    pad_token_id=0,
)
    logger.info(f"  Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def freeze_base_weights(model):
    """ 
    Freeze ALL base model weights:
    - Feature extractor (CNN layers)
    - Encoder (Transformer layers)
    
    The CTC head (lm_head) is intentionally left trainable.
    """
    if not FREEZE_BASE_WEIGHTS:
        logger.info("FREEZE_BASE_WEIGHTS is False — skipping freeze")
        return model

    logger.info("Freezing base model weights...")

    # Freeze feature extractor
    for param in model.wav2vec2.feature_extractor.parameters():
        param.requires_grad = False

    # Freeze feature projection
    for param in model.wav2vec2.feature_projection.parameters():
        param.requires_grad = False

    # Freeze encoder (transformer layers)
    for param in model.wav2vec2.encoder.parameters():
        param.requires_grad = False

    # Keep CTC head (lm_head) trainable
    for param in model.lm_head.parameters():
        param.requires_grad = True

    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    logger.info(f"  Frozen parameters: {frozen}")
    logger.info(f"  Trainable parameters (CTC head): {trainable}")

    return model


def inject_lora_adapters(model, adapter_name="default"):
    """
    Inject LoRA adapters into the Wav2Vec2 encoder.
    
    This unfreezes the adapter parameters while keeping the base weights frozen.
    LoRA adds low-rank decomposition matrices (A and B) to the target modules.
    
    Args:
        model: The Wav2Vec2ForCTC model (with frozen base weights)
        adapter_name: Name for this adapter configuration
    
    Returns:
        PEFT model with LoRA adapters
    """
    logger.info(f"Injecting LoRA adapters (adapter_name='{adapter_name}')...")
    logger.info(f"  LoRA config: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    logger.info(f"  Target modules: {LORA_TARGET_MODULES}")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        modules_to_save=["lm_head"],  # Keep CTC head fully trainable
    )

    peft_model = get_peft_model(model, lora_config)

    # Log parameter summary
    total_params = sum(p.numel() for p in peft_model.parameters())
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    logger.info(f"\n  PEFT Model Summary:")
    logger.info(f"    Total parameters:     {total_params:>12,}")
    logger.info(f"    Trainable parameters: {trainable_params:>12,}")
    logger.info(f"    Frozen parameters:    {frozen_params:>12,}")
    logger.info(f"    Trainable %:          {100 * trainable_params / total_params:.2f}%")

    return peft_model


def create_speaker_adapter_model(speaker_id, base_adapter_path=None):
    """
    Create a model with LoRA adapters for a specific speaker.
    
    If base_adapter_path is provided and exists, initializes adapter weights
    from the generalized adapter checkpoint (warm-starting transfer learning).
    Otherwise initializes fresh LoRA matrices.
    
    Returns: (peft_model, processor)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Creating speaker adapter for: {speaker_id}")
    if base_adapter_path and os.path.exists(base_adapter_path):
        logger.info(f"  -> Initializing from generalized adapter: {base_adapter_path}")
    else:
        logger.info(f"  -> Initializing with fresh LoRA weights")
    logger.info(f"{'='*70}")

    model = load_base_model()
    model = freeze_base_weights(model)
    processor = load_processor()

    if base_adapter_path and os.path.exists(base_adapter_path):
        from peft import PeftModel
        peft_model = PeftModel.from_pretrained(
            model,
            base_adapter_path,
            is_trainable=True,
            adapter_name=f"speaker_{speaker_id}",
        )
        logger.info(f"✓ Successfully warm-started speaker adapter from generalized weights")
    else:
        peft_model = inject_lora_adapters(model, adapter_name=f"speaker_{speaker_id}")

    return peft_model, processor


def create_generalized_adapter_model():
    """
    Create a model with LoRA adapters for the generalized (multi-speaker) adapter.
    
    Returns: (peft_model, processor)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Creating GENERALIZED adapter model")
    logger.info(f"{'='*70}")

    model = load_base_model()
    model = freeze_base_weights(model)
    peft_model = inject_lora_adapters(model, adapter_name="generalized")
    processor = load_processor()

    return peft_model, processor


def save_adapter(peft_model, adapter_save_dir, adapter_name):
    """Save only the adapter weights (not the full model)."""
    save_path = os.path.join(adapter_save_dir, adapter_name)
    os.makedirs(save_path, exist_ok=True)
    peft_model.save_pretrained(save_path)
    logger.info(f"Saved adapter '{adapter_name}' to: {save_path}")


def load_adapter(adapter_path, base_model=None):
    """Load a saved adapter on top of the base model."""
    if base_model is None:
        base_model = load_base_model()

    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    logger.info(f"Loaded adapter from: {adapter_path}")
    return peft_model


# ─────────────────────────────────────────────────────────────────────────────
# Main: Verify model setup
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("STEP 4: MODEL SETUP VERIFICATION")
    logger.info("=" * 70)

    # Test creating a speaker adapter
    model, processor = create_speaker_adapter_model("TEST_SPEAKER")

    # Print model architecture summary
    logger.info(f"\nModel architecture:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"  TRAINABLE: {name:60s} {str(param.shape):>20s}")

    # Verify processor
    logger.info(f"\nProcessor vocab size: {processor.tokenizer.vocab_size}")
    logger.info(f"Feature extractor sample rate: {processor.feature_extractor.sampling_rate}")

    # Test save/load cycle
    test_save_dir = os.path.join(ADAPTER_DIR, "_test")
    save_adapter(model, ADAPTER_DIR, "_test")

    # Cleanup test
    import shutil
    if os.path.exists(test_save_dir):
        shutil.rmtree(test_save_dir)
        logger.info(f"Cleaned up test adapter directory")

    logger.info("\n✓ Model setup verification complete!")
