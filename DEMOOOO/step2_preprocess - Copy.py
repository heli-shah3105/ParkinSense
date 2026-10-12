"""
=============================================================================
Step 2: Audio Preprocessing & Transcript Cleaning
=============================================================================
Takes the raw manifest, then:
  1. Validates each audio file exists and is readable
  2. Resamples to 16kHz mono
  3. Trims leading/trailing silence
  4. Normalizes volume to target dBFS
  5. Computes duration and SNR, filters outliers
  6. Cleans transcripts (lowercase, remove punctuation, filter non-speech prompts)
  7. Removes corrupt/empty/bad-quality pairs

Output: output/manifests/cleaned_manifest.csv
        output/processed_audio/{speaker_id}/{filename}.wav
        output/logs/02_preprocessing.log
"""

import os
import re
import string
import warnings
import numpy as np
import pandas as pd
import soundfile as sf
import logging
from pathlib import Path

try:
    import librosa
except ImportError:
    raise ImportError("librosa is required. Install with: pip install librosa")

from config import (
    SAMPLE_RATE, MIN_DURATION_SEC, MAX_DURATION_SEC,
    SILENCE_THRESHOLD_DB, MIN_SNR_DB, NORMALIZE_TARGET_DB,
    LOWERCASE_TRANSCRIPTS, REMOVE_PUNCTUATION, NON_SPEECH_PATTERNS,
    PROCESSED_AUDIO_DIR, MANIFEST_DIR, LOGS_DIR,
)

warnings.filterwarnings("ignore")

# ─── Logging ────────────────────────────────────────────────────────────────
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PROCESSED_AUDIO_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "02_preprocessing.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Audio Processing Functions
# ─────────────────────────────────────────────────────────────────────────────

def load_and_resample(audio_path, target_sr=SAMPLE_RATE):
    """Load audio file, convert to mono, resample to target SR."""
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return audio, target_sr
    except Exception as e:
        logger.warning(f"Failed to load {audio_path}: {e}")
        return None, None


def trim_silence(audio, sr, threshold_db=SILENCE_THRESHOLD_DB):
    """Trim leading and trailing silence from audio."""
    try:
        trimmed, _ = librosa.effects.trim(audio, top_db=abs(threshold_db))
        return trimmed
    except Exception:
        return audio


def normalize_volume(audio, target_db=NORMALIZE_TARGET_DB):
    """Normalize audio to target dBFS (decibels relative to full scale)."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return audio  # Silent audio, don't normalize

    current_db = 20 * np.log10(rms + 1e-10)
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20)
    normalized = audio * gain_linear

    # Clip to prevent overflow
    normalized = np.clip(normalized, -1.0, 1.0)
    return normalized


def compute_snr(audio):
    """
    Estimate SNR using a simple energy-based approach.
    Splits audio into frames, uses the quietest frames as noise estimate.
    """
    if len(audio) < 1600:  # Less than 0.1s at 16kHz
        return 0.0

    frame_length = 1600  # 100ms frames at 16kHz
    hop_length = 800
    n_frames = (len(audio) - frame_length) // hop_length + 1

    if n_frames < 3:
        return 0.0

    frame_energies = []
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + frame_length]
        energy = np.mean(frame ** 2)
        frame_energies.append(energy)

    frame_energies = np.array(frame_energies)
    frame_energies = frame_energies[frame_energies > 0]

    if len(frame_energies) < 3:
        return 0.0

    # Bottom 10% of frames as noise estimate
    n_noise = max(1, len(frame_energies) // 10)
    sorted_energies = np.sort(frame_energies)
    noise_energy = np.mean(sorted_energies[:n_noise])
    signal_energy = np.mean(sorted_energies)

    if noise_energy < 1e-20:
        return 60.0  # Very clean

    snr = 10 * np.log10(signal_energy / (noise_energy + 1e-20))
    return snr


def compute_duration(audio, sr):
    """Compute duration in seconds."""
    return len(audio) / sr


# ─────────────────────────────────────────────────────────────────────────────
# Transcript Cleaning Functions
# ─────────────────────────────────────────────────────────────────────────────

def is_non_speech_prompt(text):
    """Check if transcript is a non-speech instruction (should be filtered)."""
    for pattern in NON_SPEECH_PATTERNS:
        if re.match(pattern, text.strip()):
            return True
    return False


def clean_transcript(text):
    """
    Clean a transcript string:
    1. Strip whitespace
    2. Lowercase (optional)
    3. Remove punctuation (optional)
    4. Normalize whitespace
    """
    text = str(text).strip()

    if LOWERCASE_TRANSCRIPTS:
        text = text.lower()

    if REMOVE_PUNCTUATION:
        # Keep apostrophes within words (e.g., "don't"), remove everything else
        text = re.sub(r"[^\w\s']", " ", text)
        # Remove standalone apostrophes
        text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ─────────────────────────────────────────────────────────────────────────────
# Main Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_dataset():
    """Run the full preprocessing pipeline on the raw manifest."""
    logger.info("=" * 70)
    logger.info("STEP 2: AUDIO PREPROCESSING & TRANSCRIPT CLEANING")
    logger.info("=" * 70)

    # Load raw manifest
    raw_manifest_path = os.path.join(MANIFEST_DIR, "raw_manifest.csv")
    if not os.path.exists(raw_manifest_path):
        logger.error(f"Raw manifest not found: {raw_manifest_path}")
        logger.error("Run step1_build_manifest.py first!")
        return

    df = pd.read_csv(raw_manifest_path)
    logger.info(f"Loaded raw manifest: {len(df)} rows")

    # Tracking stats
    stats = {
        "total_raw": len(df),
        "file_not_found": 0,
        "load_failed": 0,
        "non_speech_prompt": 0,
        "empty_transcript": 0,
        "too_short": 0,
        "too_long": 0,
        "low_snr": 0,
        "kept": 0,
    }

    cleaned_records = []

    for idx, row in df.iterrows():
        if idx % 500 == 0:
            logger.info(f"  Processing {idx}/{len(df)} ({100*idx/len(df):.1f}%)")

        audio_path = str(row["audio_path"])
        transcript = str(row["transcript"])
        speaker_id = str(row["speaker_id"])

        # ── Check file exists ───────────────────────────────────────────
        if not os.path.exists(audio_path):
            stats["file_not_found"] += 1
            continue

        # ── Filter non-speech prompts ───────────────────────────────────
        if is_non_speech_prompt(transcript):
            stats["non_speech_prompt"] += 1
            continue

        # ── Clean transcript ────────────────────────────────────────────
        cleaned_text = clean_transcript(transcript)
        if not cleaned_text or cleaned_text == "nan":
            stats["empty_transcript"] += 1
            continue

        # ── Load and process audio ──────────────────────────────────────
        audio, sr = load_and_resample(audio_path)
        if audio is None:
            stats["load_failed"] += 1
            continue

        # Trim silence
        audio = trim_silence(audio, sr)

        # Check duration
        duration = compute_duration(audio, sr)
        if duration < MIN_DURATION_SEC:
            stats["too_short"] += 1
            continue
        if duration > MAX_DURATION_SEC:
            stats["too_long"] += 1
            continue

        # Check SNR
        snr = compute_snr(audio)
        if snr < MIN_SNR_DB:
            stats["low_snr"] += 1
            continue

        # Normalize volume
        audio = normalize_volume(audio)

        # ── Save processed audio ────────────────────────────────────────
        speaker_output_dir = os.path.join(PROCESSED_AUDIO_DIR, speaker_id)
        os.makedirs(speaker_output_dir, exist_ok=True)

        output_filename = f"{speaker_id}_{idx:06d}.wav"
        output_path = os.path.join(speaker_output_dir, output_filename)

        sf.write(output_path, audio, sr)

        # ── Record cleaned entry ────────────────────────────────────────
        cleaned_records.append({
            "audio_path": output_path,
            "transcript": cleaned_text,
            "speaker_id": speaker_id,
            "is_dysarthria": row["is_dysarthria"],
            "gender": row["gender"],
            "source_folder": row["source_folder"],
            "duration_sec": round(duration, 3),
            "snr_db": round(snr, 2),
        })
        stats["kept"] += 1

    # ── Build cleaned manifest ──────────────────────────────────────────────
    cleaned_df = pd.DataFrame(cleaned_records)

    # Save cleaned manifest
    cleaned_path = os.path.join(MANIFEST_DIR, "cleaned_manifest.csv")
    cleaned_df.to_csv(cleaned_path, index=False)

    # ── Log statistics ──────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info(f"PREPROCESSING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total raw samples:        {stats['total_raw']}")
    logger.info(f"File not found:           {stats['file_not_found']}")
    logger.info(f"Load failed:              {stats['load_failed']}")
    logger.info(f"Non-speech prompt:        {stats['non_speech_prompt']}")
    logger.info(f"Empty transcript:         {stats['empty_transcript']}")
    logger.info(f"Too short (<{MIN_DURATION_SEC}s):      {stats['too_short']}")
    logger.info(f"Too long (>{MAX_DURATION_SEC}s):       {stats['too_long']}")
    logger.info(f"Low SNR (<{MIN_SNR_DB}dB):          {stats['low_snr']}")
    logger.info(f"KEPT:                     {stats['kept']}")

    if len(cleaned_df) > 0:
        logger.info(f"\nCleaned dataset stats:")
        logger.info(f"  Unique speakers: {cleaned_df['speaker_id'].nunique()}")
        logger.info(f"  Dysarthric samples: {cleaned_df['is_dysarthria'].sum()}")
        logger.info(f"  Control samples: {(~cleaned_df['is_dysarthria']).sum()}")
        logger.info(f"  Avg duration: {cleaned_df['duration_sec'].mean():.2f}s")
        logger.info(f"  Avg SNR: {cleaned_df['snr_db'].mean():.1f} dB")

        # Per-speaker stats
        speaker_stats = cleaned_df.groupby("speaker_id").agg(
            count=("audio_path", "count"),
            is_dys=("is_dysarthria", "first"),
            avg_duration=("duration_sec", "mean"),
            avg_snr=("snr_db", "mean"),
            total_duration=("duration_sec", "sum"),
        ).sort_values("count", ascending=False)

        logger.info(f"\nPer-speaker statistics:")
        logger.info(f"\n{speaker_stats.to_string()}")

        # Save speaker stats
        speaker_stats_path = os.path.join(LOGS_DIR, "speaker_stats.csv")
        speaker_stats.to_csv(speaker_stats_path)
        logger.info(f"\nSpeaker stats saved to: {speaker_stats_path}")

    logger.info(f"\nCleaned manifest saved to: {cleaned_path}")
    return cleaned_df


if __name__ == "__main__":
    preprocess_dataset()
