"""
=============================================================================
Step 3: Speaker-Level Train/Test Splits
=============================================================================
Two-level splitting strategy:

1. SPEAKER-LEVEL SPLIT: 
   - ~80% of speakers → "train speakers" (used for both per-speaker and generalized adapters)
   - ~20% of speakers → "held-out test speakers" (ONLY for generalized adapter evaluation)
   - Stratified by dysarthria status to ensure both groups have dys + control speakers

2. INTRA-SPEAKER SPLIT (within each train speaker):
   - 80% train / 10% val / 10% test (for per-speaker adapter training)

Output:
  - output/manifests/split_manifest.csv  (with 'speaker_group' and 'split' columns)
  - output/logs/03_splits.log
"""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    MANIFEST_DIR, LOGS_DIR,
    HELD_OUT_SPEAKER_FRACTION,
    INTRA_SPEAKER_TRAIN_RATIO,
    INTRA_SPEAKER_VAL_RATIO,
    INTRA_SPEAKER_TEST_RATIO,
    RANDOM_SEED,
    SPEAKER_SEVERITY,
)

# ─── Logging ────────────────────────────────────────────────────────────────
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "03_splits.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def create_speaker_splits():
    """
    Create speaker-level and intra-speaker splits.
    Stratifies speakers by clinical dysarthria severity:
      - Control (non-dysarthric)
      - Mild dysarthria
      - Moderate dysarthria
      - Severe dysarthria
    Ensures every severity level is represented in both train and held-out test groups!
    """
    logger.info("=" * 70)
    logger.info("STEP 3: SPEAKER-LEVEL TRAIN/TEST SPLITS (SEVERITY STRATIFIED)")
    logger.info("=" * 70)

    # Load cleaned manifest
    cleaned_path = os.path.join(MANIFEST_DIR, "cleaned_manifest.csv")
    if not os.path.exists(cleaned_path):
        logger.error(f"Cleaned manifest not found: {cleaned_path}")
        logger.error("Run step2_preprocess.py first!")
        return

    df = pd.read_csv(cleaned_path)
    logger.info(f"Loaded cleaned manifest: {len(df)} samples, {df['speaker_id'].nunique()} speakers")

    # Add severity column to manifest
    df["severity"] = df["speaker_id"].map(SPEAKER_SEVERITY).fillna("Unknown")

    # ─────────────────────────────────────────────────────────────────────
    # 1. SPEAKER-LEVEL SPLIT (Stratified by Severity Level)
    # ─────────────────────────────────────────────────────────────────────
    speaker_info = df.groupby("speaker_id").agg(
        is_dysarthria=("is_dysarthria", "first"),
        severity=("severity", "first"),
        gender=("gender", "first"),
        count=("audio_path", "count"),
    ).reset_index()

    logger.info(f"\nSpeaker Severity Breakdown:")
    for sev, group in speaker_info.groupby("severity"):
        logger.info(f"  [{sev:8s}] {len(group)} speakers: {group['speaker_id'].tolist()}")

    # Stratify by severity level
    train_speakers = []
    heldout_speakers = []

    for sev, group in speaker_info.groupby("severity"):
        spks = sorted(group["speaker_id"].tolist())
        if len(spks) == 1:
            train_speakers.extend(spks)
        elif len(spks) == 2:
            # 1 for train, 1 for heldout
            train_speakers.append(spks[0])
            heldout_speakers.append(spks[1])
        else:
            n_held = max(1, int(round(len(spks) * HELD_OUT_SPEAKER_FRACTION)))
            t_spks, h_spks = train_test_split(
                spks, test_size=n_held, random_state=RANDOM_SEED
            )
            train_speakers.extend(t_spks)
            heldout_speakers.extend(h_spks)

    train_speakers = sorted(train_speakers)
    heldout_speakers = sorted(heldout_speakers)

    logger.info(f"\n{'─'*50}")
    logger.info(f"SPEAKER-LEVEL SPLIT (SEVERITY STRATIFIED):")
    logger.info(f"  Train speakers ({len(train_speakers)}):")
    for s in train_speakers:
        logger.info(f"    - {s:12s} ({SPEAKER_SEVERITY.get(s, 'Unknown')})")
    logger.info(f"  Held-out test speakers ({len(heldout_speakers)}):")
    for s in heldout_speakers:
        logger.info(f"    - {s:12s} ({SPEAKER_SEVERITY.get(s, 'Unknown')})")

    # Assign speaker_group column
    df["speaker_group"] = df["speaker_id"].apply(
        lambda x: "train" if x in train_speakers else "heldout_test"
    )

    # ─────────────────────────────────────────────────────────────────────
    # 2. INTRA-SPEAKER SPLIT (within train speakers)
    # ─────────────────────────────────────────────────────────────────────
    df["split"] = ""

    # For held-out speakers: ALL their data is "test" (generalized adapter eval)
    df.loc[df["speaker_group"] == "heldout_test", "split"] = "test"

    # For train speakers: split their data into train/val/test
    for speaker_id in train_speakers:
        speaker_mask = df["speaker_id"] == speaker_id
        speaker_indices = df[speaker_mask].index.tolist()
        n_samples = len(speaker_indices)

        if n_samples < 5:
            # Too few samples: all go to train
            df.loc[speaker_indices, "split"] = "train"
            continue

        # Shuffle deterministically
        np.random.seed(RANDOM_SEED)
        shuffled_indices = np.random.permutation(speaker_indices)

        n_test = max(1, int(n_samples * INTRA_SPEAKER_TEST_RATIO))
        n_val = max(1, int(n_samples * INTRA_SPEAKER_VAL_RATIO))
        n_train = n_samples - n_val - n_test

        test_indices = shuffled_indices[:n_test]
        val_indices = shuffled_indices[n_test:n_test + n_val]
        train_indices = shuffled_indices[n_test + n_val:]

        df.loc[test_indices, "split"] = "test"
        df.loc[val_indices, "split"] = "val"
        df.loc[train_indices, "split"] = "train"

    # ─────────────────────────────────────────────────────────────────────
    # Log split statistics
    # ─────────────────────────────────────────────────────────────────────
    logger.info(f"\n{'─'*50}")
    logger.info(f"INTRA-SPEAKER SPLIT SUMMARY:")

    for speaker_id in sorted(df["speaker_id"].unique()):
        speaker_data = df[df["speaker_id"] == speaker_id]
        group = speaker_data["speaker_group"].iloc[0]
        is_dys = speaker_data["is_dysarthria"].iloc[0]

        train_count = len(speaker_data[speaker_data["split"] == "train"])
        val_count = len(speaker_data[speaker_data["split"] == "val"])
        test_count = len(speaker_data[speaker_data["split"] == "test"])

        dys_label = "DYS" if is_dys else "CTL"
        logger.info(
            f"  {speaker_id:20s} [{dys_label}] [{group:12s}] "
            f"train={train_count:4d} val={val_count:4d} test={test_count:4d} "
            f"total={len(speaker_data):4d}"
        )

    # Overall summary
    logger.info(f"\n{'─'*50}")
    logger.info(f"OVERALL SPLIT SUMMARY:")
    split_summary = df.groupby(["speaker_group", "split"]).agg(
        count=("audio_path", "count"),
        n_speakers=("speaker_id", "nunique"),
    )
    logger.info(f"\n{split_summary.to_string()}")

    # ─────────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────────
    output_path = os.path.join(MANIFEST_DIR, "split_manifest.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"\nSaved split manifest to: {output_path}")

    # Also save speaker-group mapping for reference
    all_speakers = sorted(set(train_speakers) | set(heldout_speakers))
    speaker_groups = pd.DataFrame({
        "speaker_id": all_speakers,
        "speaker_group": [
            "train" if s in train_speakers else "heldout_test"
            for s in all_speakers
        ],
        "severity": [SPEAKER_SEVERITY.get(s, "Unknown") for s in all_speakers],
    })
    speaker_groups_path = os.path.join(MANIFEST_DIR, "speaker_groups.csv")
    speaker_groups.to_csv(speaker_groups_path, index=False)
    logger.info(f"Saved speaker groups to: {speaker_groups_path}")

    return df


if __name__ == "__main__":
    create_speaker_splits()
