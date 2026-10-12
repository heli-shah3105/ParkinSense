# VocalAid AI: Parkinson's Dysarthric Speech-to-Text Platform

A complete end-to-end deep learning system for transcribing speech from individuals with Parkinson's Disease and hypokinetic dysarthria. Uses **Wav2Vec2** acoustic representations with hierarchical **LoRA (Low-Rank Adaptation)** adapters.

---

## 🎯 Key Architectural Highlights

1. **Frozen Base Acoustic Representations**:
   - Base model: `facebook/wav2vec2-base-960h` (Feature extractor CNN and 12-layer Transformer encoder frozen).
2. **LoRA Adapters ($r=16, \alpha=32$)**:
   - Injected into self-attention projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`) and feedforward layers (`intermediate_dense`, `output_dense`).
   - Keeps the CTC linear projection head (`lm_head`) fully trainable.
3. **Two-Tier Sequential Transfer Learning**:
   - **Tier 1 (Generalized Adapter)**: Trained on pooled train data from 15 speakers across all severity levels. Evaluated on unseen held-out speakers.
   - **Tier 2 (Personalized Adapters)**: Initialized **from the trained generalized adapter checkpoint**, then fine-tuned on individual patient data to tailor the acoustic model to personal vocal tremor and articulation patterns.
4. **Clinical Severity Stratification**:
   - Stratified across 4 clinical levels: **Mild**, **Moderate**, **Severe**, and **Control**.
   - Held-out test set includes severe dysarthria (`F3_F03`) and control speakers (`F3_FC02`, `UA_CF03`) for strict zero-shot evaluation.

---

## 📂 Project Directory Structure

```
DEMOOOO/
├── config.py                 # Central configuration: paths, audio, LoRA & training hyperparams
├── step1_build_manifest.py   # Scans UASpeech + TORGO and merges into raw_manifest.csv
├── step2_preprocess.py       # Audio standardization (16kHz mono, -20dBFS norm, SNR/silence filter)
├── step3_split_speakers.py   # Severity-stratified speaker splits + intra-speaker train/val/test
├── step4_model_setup.py      # Wav2Vec2 + LoRA injection & generalized warm-start loading
├── step5_train.py            # Phase 1 (Generalized) -> Phase 2 (Per-speaker warm-started)
├── step6_inference.py        # Production transcriber with smart routing & word timestamps
├── model.py                  # High-level SpeechToTextModel Python API
├── benchmark.py              # Clinical evaluation comparing Baseline vs Gen vs Personal
├── app.py                    # Interactive web platform with audio recorder & dashboard
├── run_pipeline.py           # Master runner orchestrating steps 1 through 7
├── requirements.txt          # Python dependencies
└── output/
    ├── manifests/            # raw_manifest.csv, cleaned_manifest.csv, split_manifest.csv
    ├── processed_audio/      # 8,516 standardized 16kHz mono audio files
    ├── adapters/             # generalized_adapter and speaker_* checkpoints
    └── logs/                 # Clinical logs, benchmark reports, speaker stats
```

---

## 🚀 Quick Start Guide

### 1. Direct Python Speech-to-Text API

```python
from model import SpeechToTextModel

# Auto-detects personal adapter if speaker_id matches, else uses generalized adapter
asr = SpeechToTextModel()

# Transcribe any audio file (.wav, .mp3, numpy array, or tensor)
result = asr.transcribe("output/processed_audio/F3_F01/F3_F01_009250.wav")
print("Transcription:", result["transcription"])
print("Confidence:   ", result["confidence"])
print("Adapter:      ", result["adapter_used"])

# Transcribe with word-level timestamps (for clinical speech rate metrics)
detailed = asr.transcribe_with_timestamps("sample.wav")
for word in detailed["words"]:
    print(f"[{word['start_time']}s - {word['end_time']}s] {word['word']} (conf: {word['confidence']})")
```

---

### 2. Interactive Web Platform

Launch the local web application:
```bash
python app.py
```
Open **`http://localhost:8080`** in your browser to:
- Select an adapter (unseen patient or specific speaker).
- Record live from microphone or upload audio files.
- Listen to preprocessed samples with synchronized word highlighting.
- Inspect real-time clinical metrics (Words Per Minute, pause ratio, articulation score).

---

### 3. Command-Line Inference

```bash
# Transcribe single file
python step6_inference.py --audio "output/processed_audio/F3_M01/F3_M01_000001.wav"

# Transcribe with a specific patient's personal adapter
python step6_inference.py --audio "sample.wav" --speaker_id F3_M01

# Transcribe with word timestamps
python step6_inference.py --audio "sample.wav" --timestamps

# List all available trained adapters on disk
python step6_inference.py --list_adapters
```

---

### 4. Training Pipeline

```bash
# Run full training (Phase 1 Generalized -> Phase 2 Per-Speaker Adapters)
python step5_train.py --mode all

# Train only Phase 1 (Generalized Adapter)
python step5_train.py --mode general

# Fine-tune a specific patient's adapter from the generalized checkpoint
python step5_train.py --mode speaker --speaker_id F3_M01
```

---

### 5. Clinical Benchmarking

Run evaluation comparing Baseline (Zero-Shot) vs. Generalized Adapter vs. Personalized Adapters:
```bash
python benchmark.py
```
Generates Markdown and CSV reports at `output/logs/clinical_benchmark_report.md` detailing Relative Error Reduction (RER %) by clinical severity.

---

### 6. Master Pipeline Runner

Run the entire pipeline or specific stages:
```bash
python run_pipeline.py              # Runs standard pipeline (steps 1-5)
python run_pipeline.py --step 4-6   # Model setup, training & inference check
python run_pipeline.py --step 7     # Run clinical benchmark
python run_pipeline.py --web        # Launch web application
```
