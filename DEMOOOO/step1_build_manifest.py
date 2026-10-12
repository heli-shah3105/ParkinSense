"""
=============================================================================
Step 1: Data Discovery & Unified Manifest Creation
=============================================================================
Scans all three dataset folders, extracts audio paths and transcripts,
assigns speaker IDs, and merges everything into a single unified manifest CSV.

Output: output/manifests/raw_manifest.csv
Columns: audio_path, transcript, speaker_id, is_dysarthria, gender, source_folder, mic_type
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import glob
import pandas as pd
import logging

from config import (
    FOLDER1_ROOT, FOLDER2_ROOT, FOLDER2_DATASET, FOLDER2_CSV,
    FOLDER3_ROOT, MANIFEST_DIR, LOGS_DIR, UASPEECH_WORDS
)

# ─── Logging ────────────────────────────────────────────────────────────────
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MANIFEST_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "01_build_manifest.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def parse_uaspeech_filename(filepath):
    """
    Parse UASpeech-style filenames like CF02_B1_CW10_M2.wav
    Returns: (speaker_id, block, word_code, mic_id) or None if unparseable.
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    # Pattern: {Speaker}_{Block}_{WordCode}_{Mic}
    match = re.match(r"^([A-Z]+\d+)_(B\d+)_([A-Z]+\d+)_(M\d+)$", basename)
    if match:
        return match.group(1), match.group(2), match.group(3), match.group(4)
    return None


def scan_folder1():
    """
    Scan Folder 1: UASpeech noise-reduced control speakers (CF02, CF03, CF04).
    These are CONTROL (non-dysarthric) speakers. Transcripts come from word codes.
    Only take mic M8 (the array microphone, highest quality) to avoid duplicates.
    """
    logger.info("=" * 70)
    logger.info("Scanning Folder 1: UASpeech Control (noise-reduced)")
    records = []

    speaker_dirs = sorted(glob.glob(os.path.join(FOLDER1_ROOT, "*")))
    for speaker_dir in speaker_dirs:
        if not os.path.isdir(speaker_dir):
            continue
        speaker_id = os.path.basename(speaker_dir)
        wav_files = glob.glob(os.path.join(speaker_dir, "*.wav"))

        for wav_path in sorted(wav_files):
            parsed = parse_uaspeech_filename(wav_path)
            if parsed is None:
                continue

            spk, block, word_code, mic = parsed

            # Only keep one mic per utterance to avoid duplicate transcripts
            # M8 = array mic (best quality for UASpeech noise-reduced)
            if mic != "M8":
                continue

            # Look up transcript from word code
            transcript = UASPEECH_WORDS.get(word_code, None)
            if transcript is None:
                logger.warning(f"Unknown word code {word_code} in {wav_path}")
                continue

            records.append({
                "audio_path": wav_path,
                "transcript": transcript,
                "speaker_id": f"UA_{spk}",   # Prefix to avoid collisions
                "is_dysarthria": False,        # Control speakers
                "gender": "Female" if spk.startswith("CF") else "Male",
                "source_folder": "folder1_uaspeech_control",
                "mic_type": "array_noisereduced",
            })

    logger.info(f"  Folder 1: Found {len(records)} samples from {len(speaker_dirs)} speaker dirs")
    return records


def scan_folder2():
    """
    Scan Folder 2: Dysarthria and Non-Dysarthria dataset with CSV metadata.
    Only keeps records if the audio file actually exists on disk.
    """
    logger.info("=" * 70)
    logger.info("Scanning Folder 2: Dysarthria and Non Dysarthria")
    records = []

    if not os.path.exists(FOLDER2_CSV):
        logger.error(f"CSV file not found: {FOLDER2_CSV}")
        return records

    df = pd.read_csv(FOLDER2_CSV)
    logger.info(f"  CSV has {len(df)} rows, columns: {list(df.columns)}")

    kaggle_prefix = "/kaggle/input/dysarthria-and-nondysarthria-speech-dataset/"

    for idx, row in df.iterrows():
        wav_kaggle_path = str(row.get("Wav_path", ""))
        transcript = str(row.get("Prompts", ""))
        gender = str(row.get("Gender", "Unknown"))
        is_dys = str(row.get("Is_dysarthria", "")).strip().lower() == "yes"

        # Remap Kaggle path to local path
        if wav_kaggle_path.startswith(kaggle_prefix):
            relative_path = wav_kaggle_path[len(kaggle_prefix):]
        else:
            relative_path = wav_kaggle_path.lstrip("/")

        local_wav_path = os.path.join(FOLDER2_ROOT, relative_path)
        if not os.path.exists(local_wav_path):
            continue

        # Extract speaker ID from path
        # Path pattern: .../Dataset/{Category}/{SpeakerID}/Session{N}/Wav/{file}.wav
        path_parts = relative_path.replace("/", os.sep).split(os.sep)
        speaker_id = "UNKNOWN"
        for i, part in enumerate(path_parts):
            if part == "Dataset" and i + 2 < len(path_parts):
                speaker_id = path_parts[i + 2]
                break

        records.append({
            "audio_path": local_wav_path,
            "transcript": transcript,
            "speaker_id": f"F2_{speaker_id}",   # Prefix for folder 2
            "is_dysarthria": is_dys,
            "gender": gender,
            "source_folder": "folder2_dysarthria_nondysarthria",
            "mic_type": "unknown",
        })

    logger.info(f"  Folder 2: Found {len(records)} existing audio samples on disk")
    return records


def scan_folder3():
    """
    Scan Folder 3: TORGO dataset (F_Dys, M_Dys, F_Con, M_Con).
    Accurately matches each audio file to its ground-truth prompt in FOLDER2_CSV
    using the (speaker_id, session_num, utterance_num) composite key.
    Only headMic audio is used (highest quality, no duplicates).
    """
    logger.info("=" * 70)
    logger.info("Scanning Folder 3: TORGO-style (F_Dys, M_Dys, F_Con, M_Con)")
    records = []

    # 1. Build ground-truth prompt lookup from FOLDER2_CSV
    # Key: (speaker_id.upper(), session_int, utterance_int) -> (prompt, is_dysarthria, gender)
    csv_lookup = {}
    spk_sessions = {}

    if os.path.exists(FOLDER2_CSV):
        df = pd.read_csv(FOLDER2_CSV)
        csv_re = re.compile(
            r"Dataset/(?P<cat>[^/]+)/(?P<spk>[^/]+)/Session(?P<sess>\d+)/Wav/(?P<utt>\d+)\.wav",
            re.I
        )
        for _, row in df.iterrows():
            wav_p = str(row.get("Wav_path", "")).replace("\\", "/")
            m = csv_re.search(wav_p)
            if not m:
                continue
            spk = m["spk"].upper()
            sess = int(m["sess"])
            utt = int(m["utt"])
            prompt = str(row.get("Prompts", "")).strip()
            gender = str(row.get("Gender", "Unknown")).strip()
            is_dys = str(row.get("Is_dysarthria", "")).strip().lower() == "yes"

            csv_lookup[(spk, sess, utt)] = (prompt, is_dys, gender)
            if spk not in spk_sessions:
                spk_sessions[spk] = set()
            spk_sessions[spk].add(sess)
        logger.info(f"  Loaded {len(csv_lookup)} prompt mappings from {FOLDER2_CSV}")
    else:
        logger.warning(f"  FOLDER2_CSV not found: {FOLDER2_CSV}")

    categories = {
        "F_Dys": {"is_dys": True, "gender": "Female"},
        "M_Dys": {"is_dys": True, "gender": "Male"},
        "F_Con": {"is_dys": False, "gender": "Female"},
        "M_Con": {"is_dys": False, "gender": "Male"},
    }

    f3_dir_re = re.compile(r"wav_(?P<mic>arrayMic|headMic)_(?P<spk>[A-Za-z]+\d+)(?:S(?P<sess>\d+))?$", re.I)

    matched_count = 0
    unmatched_count = 0

    for category, meta in categories.items():
        cat_dir = os.path.join(FOLDER3_ROOT, category)
        if not os.path.isdir(cat_dir):
            logger.warning(f"  Category dir not found: {cat_dir}")
            continue

        session_dirs = sorted(os.listdir(cat_dir))
        for session_name in session_dirs:
            session_path = os.path.join(cat_dir, session_name)
            if not os.path.isdir(session_path):
                continue

            m = f3_dir_re.match(session_name)
            if not m:
                continue

            mic_type = m.group("mic")
            speaker_id = m.group("spk").upper()
            raw_sess = m.group("sess")
            sess_num = int(raw_sess) if raw_sess is not None else None

            # Only use headMic to avoid duplicate utterances and get cleanest audio
            if mic_type != "headMic":
                continue

            # If session is not in directory name (e.g. wav_headMic_F01),
            # determine session from CSV candidate sessions for this speaker
            if sess_num is None:
                cands = sorted(spk_sessions.get(speaker_id, [1]))
                sess_num = cands[0] if cands else 1

            wav_files = glob.glob(os.path.join(session_path, "*.wav"))
            for wav_path in sorted(wav_files):
                stem = os.path.splitext(os.path.basename(wav_path))[0]
                nums = re.findall(r"\d+", stem)
                if not nums:
                    continue
                utt_num = int(nums[-1])

                key = (speaker_id, sess_num, utt_num)
                if key in csv_lookup:
                    prompt, is_dys, gender = csv_lookup[key]
                    matched_count += 1
                else:
                    prompt = ""
                    is_dys = meta["is_dys"]
                    gender = meta["gender"]
                    unmatched_count += 1

                # Only keep samples that have a non-empty transcript
                if not prompt or prompt.lower() == "nan":
                    continue

                records.append({
                    "audio_path": wav_path,
                    "transcript": prompt,
                    "speaker_id": f"F3_{speaker_id}",
                    "is_dysarthria": is_dys,
                    "gender": gender,
                    "source_folder": "folder3_torgo",
                    "mic_type": mic_type,
                })

    logger.info(f"  Folder 3: Matched {matched_count} headMic clips (unmatched: {unmatched_count})")
    logger.info(f"  Folder 3: Created {len(records)} verified records with ground truth")
    return records


def build_unified_manifest():
    """Merge all folders into a single manifest CSV."""
    logger.info("=" * 70)
    logger.info("BUILDING UNIFIED MANIFEST")
    logger.info("=" * 70)

    # Scan all three folders
    records_f1 = scan_folder1()
    records_f2 = scan_folder2()
    records_f3 = scan_folder3()

    all_records = records_f1 + records_f2 + records_f3
    df = pd.DataFrame(all_records)

    logger.info(f"\n{'='*70}")
    logger.info(f"TOTAL RAW RECORDS: {len(df)}")
    logger.info(f"  Folder 1 (UASpeech Control): {len(records_f1)}")
    logger.info(f"  Folder 2 (Dysarthria/NonDys): {len(records_f2)}")
    logger.info(f"  Folder 3 (TORGO-style):      {len(records_f3)}")
    logger.info(f"  Unique speakers: {df['speaker_id'].nunique()}")
    logger.info(f"  Dysarthric samples: {df['is_dysarthria'].sum()}")
    logger.info(f"  Control samples: {(~df['is_dysarthria']).sum()}")

    # Per-speaker summary
    speaker_summary = df.groupby("speaker_id").agg(
        count=("audio_path", "count"),
        is_dysarthria=("is_dysarthria", "first"),
        gender=("gender", "first"),
        source=("source_folder", "first"),
    ).sort_values("count", ascending=False)

    logger.info(f"\nPer-speaker sample counts:")
    logger.info(f"\n{speaker_summary.to_string()}")

    # Save raw manifest
    output_path = os.path.join(MANIFEST_DIR, "raw_manifest.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"\nSaved raw manifest to: {output_path}")

    return df


if __name__ == "__main__":
    build_unified_manifest()
