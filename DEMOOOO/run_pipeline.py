import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Fix Windows/Anaconda OpenMP conflict

"""
=============================================================================
MASTER RUNNER — Dysarthric Speech-to-Text Pipeline
=============================================================================
Runs all steps in sequence:
  Step 1: Build unified manifest from all 3 data folders
  Step 2: Audio preprocessing & transcript cleaning
  Step 3: Speaker-level train/test splits
  Step 4: Model setup verification (Wav2Vec2 + LoRA)
  Step 5: Train per-speaker adapters + generalized adapter
  Step 6: [Inference — run separately via CLI]

Usage:
  python run_pipeline.py              # Run all steps
  python run_pipeline.py --step 1     # Run only step 1
  python run_pipeline.py --step 1-3   # Run steps 1 through 3
  python run_pipeline.py --step 5     # Run only training (assumes steps 1-3 done)
"""

import argparse
import sys
import os
import time
import logging

try:
    import torch
except ImportError:
    conda_py = r"C:\Users\Heli P Shah\ANACONDA\envs\dysarthric\python.exe"
    if os.path.exists(conda_py) and sys.executable.lower() != conda_py.lower():
        import subprocess
        print(f"[*] Re-launching pipeline with conda 'dysarthric' environment: {conda_py}")
        res = subprocess.run([conda_py] + sys.argv)
        sys.exit(res.returncode)
    else:
        print("[!] PyTorch is not installed in current environment.")
        print("[!] Please run: conda activate dysarthric")
        sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_step(step_num):
    """Run a specific step of the pipeline."""
    start_time = time.time()

    if step_num == 1:
        logger.info("\n" + "█" * 70)
        logger.info("█  STEP 1: BUILD UNIFIED MANIFEST")
        logger.info("█" * 70)
        from step1_build_manifest import build_unified_manifest
        build_unified_manifest()

    elif step_num == 2:
        logger.info("\n" + "█" * 70)
        logger.info("█  STEP 2: AUDIO PREPROCESSING & TRANSCRIPT CLEANING")
        logger.info("█" * 70)
        from step2_preprocess import preprocess_dataset
        preprocess_dataset()

    elif step_num == 3:
        logger.info("\n" + "█" * 70)
        logger.info("█  STEP 3: SPEAKER-LEVEL TRAIN/TEST SPLITS")
        logger.info("█" * 70)
        from step3_split_speakers import create_speaker_splits
        create_speaker_splits()

    elif step_num == 4:
        logger.info("\n" + "█" * 70)
        logger.info("█  STEP 4: MODEL SETUP VERIFICATION")
        logger.info("█" * 70)
        from step4_model_setup import create_speaker_adapter_model, load_processor
        model, processor = create_speaker_adapter_model("VERIFICATION")
        logger.info("✓ Model setup verified successfully")

    elif step_num == 5:
        logger.info("\n" + "█" * 70)
        logger.info("█  STEP 5: TRAINING (Generalized first, then Per-Speaker warm-started)")
        logger.info("█" * 70)
        from step5_train import run_full_training
        run_full_training()

    elif step_num == 6:
        logger.info("\n" + "█" * 70)
        logger.info("█  STEP 6: INFERENCE / DEMO VERIFICATION")
        logger.info("█" * 70)
        from step6_inference import DysarthricASR
        asr = DysarthricASR()
        info = asr.list_available_adapters()
        logger.info(f"✓ Model ready. Adapters: {info['count_speaker_adapters']} speaker adapters, generalized={'Yes' if info['generalized_adapter']['available'] else 'No'}")

    elif step_num == 7:
        logger.info("\n" + "█" * 70)
        logger.info("█  STEP 7: CLINICAL BENCHMARKING (Baseline vs. Generalized vs. Personal)")
        logger.info("█" * 70)
        from benchmark import DysarthricASRBenchmark
        bench = DysarthricASRBenchmark()
        bench.run_benchmark()

    else:
        logger.error(f"Unknown step: {step_num}")
        return

    elapsed = time.time() - start_time
    logger.info(f"\n  Step {step_num} completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")


def main():
    parser = argparse.ArgumentParser(description="Dysarthric Speech Pipeline Runner")
    parser.add_argument("--step", type=str, default=None,
                       help="Step number (e.g., '1', '3', '1-3', '1-7'). Default: run steps 1-5.")
    parser.add_argument("--web", action="store_true",
                       help="Launch interactive VocalAid AI web application")
    parser.add_argument("--port", type=int, default=8080,
                       help="Port for web application (default: 8080)")

    args = parser.parse_args()

    if args.web:
        from app import run_server
        run_server(port=args.port)
        return

    if args.step is None:
        # Run standard pipeline steps
        steps = [1, 2, 3, 4, 5]
    elif "-" in args.step:
        start, end = args.step.split("-")
        steps = list(range(int(start), int(end) + 1))
    else:
        steps = [int(args.step)]

    logger.info("=" * 70)
    logger.info("DYSARTHRIC SPEECH-TO-TEXT PIPELINE")
    logger.info(f"Steps to run: {steps}")
    logger.info("=" * 70)

    total_start = time.time()

    for step in steps:
        try:
            run_step(step)
        except Exception as e:
            logger.error(f"\n❌ Step {step} FAILED: {e}")
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)

    total_elapsed = time.time() - total_start
    logger.info(f"\n{'='*70}")
    logger.info(f"ALL STEPS COMPLETE! Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
