"""
=============================================================================
Dysarthric Speech-to-Text: Central Configuration
=============================================================================
All paths, hyperparameters, adapter settings, and data pipeline parameters
are defined here for easy tuning and reproducibility.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# BASE PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = r"c:\Users\Heli P Shah\OneDrive\Desktop\DEMOOOO"

# Dataset root directories
FOLDER1_ROOT = os.path.join(BASE_DIR, "dysartia(1)", "noisereduced-uaspeech-control")
FOLDER2_ROOT = os.path.join(BASE_DIR, "dysartia(2)")
FOLDER2_DATASET = os.path.join(BASE_DIR, "dysartia(2)", "Dysarthria and Non Dysarthria", "Dataset")
FOLDER2_CSV = os.path.join(BASE_DIR, "dysartia(2)", "data_with_path.csv")
FOLDER3_ROOT = os.path.join(BASE_DIR, "dysarthia(3)", "dysarthia(3)")

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PROCESSED_AUDIO_DIR = os.path.join(OUTPUT_DIR, "processed_audio")
MANIFEST_DIR = os.path.join(OUTPUT_DIR, "manifests")
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "adapters")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000           # Target sample rate in Hz
AUDIO_CHANNELS = 1            # Mono
MIN_DURATION_SEC = 0.3        # Minimum audio duration to keep (seconds)
MAX_DURATION_SEC = 30.0       # Maximum audio duration to keep (seconds)
SILENCE_THRESHOLD_DB = -40    # dB threshold for silence trimming
MIN_SNR_DB = 5.0              # Minimum SNR to keep a sample
NORMALIZE_TARGET_DB = -20.0   # Target dBFS for volume normalization
HIGH_PASS_FILTER_HZ = 80.0    # 80Hz high-pass to filter AC hum, mic pop, desk rumble
MAX_NORMALIZATION_GAIN_DB = 24.0 # Maximum gain boost to avoid blowing up background noise floor
ENABLE_LEXICON_RESCORING = True # Dysarthria-aware lexicon and language correction

# ─────────────────────────────────────────────────────────────────────────────
# TRANSCRIPT CLEANING
# ─────────────────────────────────────────────────────────────────────────────
LOWERCASE_TRANSCRIPTS = True
REMOVE_PUNCTUATION = True
# Patterns that indicate non-speech prompts (to be filtered out)
NON_SPEECH_PATTERNS = [
    r"^\[.*\]?$",             # [say 'Ah-P-Eee' repeatedly], [relax your mouth...], or unclosed bracket
    r"^\[.*",                 # Any prompt starting with bracket instruction
    r"^\(.*\)$",              # Parenthesized instructions / annotations
    r"^input/images/",        # Image references like input/images/story_starters_15.jpg
    r"^x+$",                  # xxx, xxxx unintelligible tokens in TORGO
    r"^\[?xxx\]?$",           # [xxx] annotations
    r"^nan$",                 # NaN strings
    r"^\s*$",                 # Empty transcripts
]

# ─────────────────────────────────────────────────────────────────────────────
# UASpeech WORD LABELS (Folder 1 transcript lookup)
# The UASpeech corpus uses coded word IDs. These are the standard mappings.
# ─────────────────────────────────────────────────────────────────────────────
UASPEECH_WORDS = {
    # Common words (CW1-CW100) - these are the 100 most common English words
    "CW1": "the", "CW2": "of", "CW3": "and", "CW4": "a", "CW5": "to",
    "CW6": "in", "CW7": "is", "CW8": "you", "CW9": "that", "CW10": "it",
    "CW11": "he", "CW12": "was", "CW13": "for", "CW14": "on", "CW15": "are",
    "CW16": "as", "CW17": "with", "CW18": "his", "CW19": "they", "CW20": "i",
    "CW21": "at", "CW22": "be", "CW23": "this", "CW24": "have", "CW25": "from",
    "CW26": "or", "CW27": "one", "CW28": "had", "CW29": "by", "CW30": "but",
    "CW31": "not", "CW32": "what", "CW33": "all", "CW34": "were", "CW35": "we",
    "CW36": "when", "CW37": "your", "CW38": "can", "CW39": "said", "CW40": "there",
    "CW41": "each", "CW42": "which", "CW43": "she", "CW44": "do", "CW45": "how",
    "CW46": "their", "CW47": "if", "CW48": "will", "CW49": "up", "CW50": "other",
    "CW51": "about", "CW52": "out", "CW53": "many", "CW54": "then", "CW55": "them",
    "CW56": "these", "CW57": "so", "CW58": "some", "CW59": "her", "CW60": "would",
    "CW61": "make", "CW62": "like", "CW63": "him", "CW64": "into", "CW65": "time",
    "CW66": "has", "CW67": "look", "CW68": "two", "CW69": "more", "CW70": "go",
    "CW71": "see", "CW72": "number", "CW73": "no", "CW74": "way", "CW75": "could",
    "CW76": "people", "CW77": "my", "CW78": "than", "CW79": "first", "CW80": "water",
    "CW81": "been", "CW82": "call", "CW83": "who", "CW84": "oil", "CW85": "its",
    "CW86": "now", "CW87": "find", "CW88": "long", "CW89": "down", "CW90": "day",
    "CW91": "did", "CW92": "get", "CW93": "come", "CW94": "made", "CW95": "may",
    "CW96": "part", "CW97": "over", "CW98": "new", "CW99": "sound", "CW100": "take",
    # Uncommon words (C1-C19) 
    "C1": "command", "C2": "computer", "C3": "backspace", "C4": "escape",
    "C5": "return", "C6": "tab", "C7": "insert", "C8": "delete",
    "C9": "home", "C10": "end", "C11": "page up", "C12": "page down",
    "C13": "caps lock", "C14": "num lock", "C15": "scroll lock",
    "C16": "print screen", "C17": "pause", "C18": "break", "C19": "help",
    # Digits (D0-D9)
    "D0": "zero", "D1": "one", "D2": "two", "D3": "three", "D4": "four",
    "D5": "five", "D6": "six", "D7": "seven", "D8": "eight", "D9": "nine",
    # Letters (LA-LZ)
    "LA": "a", "LB": "b", "LC": "c", "LD": "d", "LE": "e",
    "LF": "f", "LG": "g", "LH": "h", "LI": "i", "LJ": "j",
    "LK": "k", "LL": "l", "LM": "m", "LN": "n", "LO": "o",
    "LP": "p", "LQ": "q", "LR": "r", "LS": "s", "LT": "t",
    "LU": "u", "LV": "v", "LW": "w", "LX": "x", "LY": "y", "LZ": "z",
}

# ─────────────────────────────────────────────────────────────────────────────
# SPEAKER SEVERITY MAPPING (Clinical assessment stratification)
# ─────────────────────────────────────────────────────────────────────────────
SPEAKER_SEVERITY = {
    # Dysarthric speakers (TORGO clinical assessments)
    "F3_F01": "Moderate",
    "F3_F03": "Severe",
    "F3_F04": "Moderate",
    "F3_M01": "Mild",
    "F3_M02": "Severe",
    "F3_M03": "Moderate",
    "F3_M04": "Severe",
    "F3_M05": "Mild",
    # Controls (TORGO + UASpeech)
    "F3_FC01": "Control",
    "F3_FC02": "Control",
    "F3_FC03": "Control",
    "F3_MC01": "Control",
    "F3_MC02": "Control",
    "F3_MC03": "Control",
    "F3_MC04": "Control",
    "UA_CF02": "Control",
    "UA_CF03": "Control",
    "UA_CF04": "Control",
}

# ─────────────────────────────────────────────────────────────────────────────
# SPEAKER SPLITS
# ─────────────────────────────────────────────────────────────────────────────
# Fraction of speakers held out for generalized adapter testing
HELD_OUT_SPEAKER_FRACTION = 0.2
# Within each speaker: train/val/test split ratios
INTRA_SPEAKER_TRAIN_RATIO = 0.8
INTRA_SPEAKER_VAL_RATIO = 0.1
INTRA_SPEAKER_TEST_RATIO = 0.1
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Base model checkpoint (Wav2Vec2 pretrained on 960h LibriSpeech)
BASE_MODEL_NAME = "facebook/wav2vec2-base-960h"
FREEZE_BASE_WEIGHTS = True

# ─────────────────────────────────────────────────────────────────────────────
# LoRA ADAPTER CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
LORA_R = 16                   # LoRA rank
LORA_ALPHA = 32               # LoRA alpha (scaling factor)
LORA_DROPOUT = 0.1            # LoRA dropout
# Target modules in Wav2Vec2 encoder for adapter injection
LORA_TARGET_MODULES = [
    "q_proj",                 # Query projection in attention
    "k_proj",                 # Key projection in attention
    "v_proj",                 # Value projection in attention
    "out_proj",               # Output projection in attention
    "intermediate_dense",     # Feed-forward intermediate
    "output_dense",           # Feed-forward output
]

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# -----------------------------------------------------------------------------
# NOTE (memory): batch sizes below were reduced from the original config
# (general=16, speaker=8) because training was hitting CUDA OOM on a 6GB
# laptop GPU. Effective batch size is preserved via gradient accumulation
# (effective_batch = batch_size * gradient_accumulation), so training
# dynamics stay roughly the same -- it just takes more accumulation steps
# per optimizer update, which costs time, not memory.
#
# If you still hit CUDA OOM after this change:
#   1. Make sure gradient_checkpointing=True is set in the TrainingArguments
#      inside step5_train.py (see train_generalized_adapter() and
#      train_speaker_adapter()) -- this is NOT a config.py setting, it must
#      be added directly in step5_train.py's TrainingArguments(...) call.
#   2. Drop GENERAL_ADAPTER_BATCH_SIZE / SPEAKER_ADAPTER_BATCH_SIZE further
#      (batch_size=1 is the floor).
#   3. Lower MAX_DURATION_SEC above (e.g. to 15.0) so unusually long
#      utterances don't spike memory on their own regardless of batch size.
#   4. Lower EVAL_BATCH_SIZE below too -- eval can OOM independently of
#      training if left too high.
# ─────────────────────────────────────────────────────────────────────────────
# Adapter training order & transfer
INIT_SPEAKER_FROM_GENERAL = True  # Initialize speaker adapters from trained generalized adapter

# Generalized adapter training (uses ALL train-split speakers)
GENERAL_ADAPTER_EPOCHS = 15
GENERAL_ADAPTER_BATCH_SIZE = 4              # batch 4 utilizes GPU tensor cores efficiently
GENERAL_ADAPTER_LR = 1e-4
GENERAL_ADAPTER_WEIGHT_DECAY = 0.01
GENERAL_ADAPTER_WARMUP_STEPS = 250
GENERAL_ADAPTER_GRADIENT_ACCUMULATION = 8   # 4 * 8 = 32 effective batch size
GENERAL_ADAPTER_MAX_GRAD_NORM = 1.0

# Per-speaker adapter training (fine-tuned starting from generalized adapter)
SPEAKER_ADAPTER_EPOCHS = 20
SPEAKER_ADAPTER_BATCH_SIZE = 4              # batch 4
SPEAKER_ADAPTER_LR = 8e-5
SPEAKER_ADAPTER_WEIGHT_DECAY = 0.01
SPEAKER_ADAPTER_WARMUP_STEPS = 50
SPEAKER_ADAPTER_GRADIENT_ACCUMULATION = 4   # 4 * 4 = 16 effective batch size
SPEAKER_ADAPTER_MAX_GRAD_NORM = 1.0

# CTC head training
CTC_HEAD_LR_MULTIPLIER = 1.0   # Relative LR for CTC head vs adapters

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
EVAL_BATCH_SIZE = 4
LOG_EVERY_N_STEPS = 25
SAVE_EVERY_N_STEPS = 250
EARLY_STOPPING_PATIENCE = 3
