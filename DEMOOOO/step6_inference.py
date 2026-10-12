"""
=============================================================================
Step 6: Inference & Production ASR Model — Dysarthric Speech-to-Text
=============================================================================
Provides a direct, production-ready speech-to-text model for people with
Parkinson's disease and dysarthric speech.

Features:
  1. Smart Adapter Routing:
     - Automatically routes to personalized speaker adapter if available.
     - Gracefully falls back to the generalized adapter for unseen speakers.
     - Fallbacks to base pretrained Wav2Vec2 checkpoint if no adapters exist yet.
  2. Multi-Format Input Support:
     - Audio file paths (.wav, .flac, .mp3, .ogg)
     - Raw NumPy arrays (1D float32 or int16)
     - PyTorch Tensors
  3. Word-Level Timestamps:
     - Computes word start/end times and confidence scores from CTC activations.
  4. Real-Time Dynamic Switching:
     - Switch adapters between speakers on-the-fly in memory without reloading the base model.
  5. Built-in Evaluation:
     - Compute WER and CER across test splits or held-out speakers.

Python API Usage:
  >>> from step6_inference import DysarthricSpeechTranscriber
  >>> asr = DysarthricASR(speaker_id="F3_M01")  # Auto-selects adapter
  >>> result = asr.transcribe("path/to/speech.wav")
  >>> print(result["transcription"])

CLI Usage:
  python step6_inference.py --audio sample.wav
  python step6_inference.py --audio sample.wav --speaker_id F3_M01
  python step6_inference.py --audio sample.wav --timestamps
  python step6_inference.py --audio_dir output/processed_audio/F3_F01/
  python step6_inference.py --list_adapters
  python step6_inference.py --eval --split test
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from peft import PeftModel
except ImportError:
    conda_py = r"C:\Users\Heli P Shah\ANACONDA\envs\dysarthric\python.exe"
    if os.path.exists(conda_py) and sys.executable.lower() != conda_py.lower():
        import subprocess
        res = subprocess.run([conda_py] + sys.argv)
        sys.exit(res.returncode)
    else:
        print("[!] Required ML libraries not found. Run: conda activate dysarthric")
        sys.exit(1)

from config import (
    BASE_MODEL_NAME, SAMPLE_RATE, ADAPTER_DIR,
    MANIFEST_DIR, LOGS_DIR, SPEAKER_SEVERITY,
    SILENCE_THRESHOLD_DB, NORMALIZE_TARGET_DB,
    HIGH_PASS_FILTER_HZ, MAX_NORMALIZATION_GAIN_DB,
    ENABLE_LEXICON_RESCORING
)
from lexicon_decoder import DysarthricLexiconDecoder

# ─── Logging ────────────────────────────────────────────────────────────────
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class DysarthricSpeechTranscriber:
    """
    Speech-to-Text Model for Parkinson's / Dysarthric Speech.
    Uses Wav2Vec2-base with LoRA adapters (generalized + personalized).
    """

    def __init__(
        self,
        adapter_path: Optional[str] = None,
        speaker_id: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the transcriber.

        Args:
            adapter_path: Direct path to a trained LoRA adapter directory.
            speaker_id: Speaker ID (e.g., 'F3_M01') for automatic adapter routing.
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing Dysarthric Speech Transcriber on device: {self.device}")

        # Load Wav2Vec2 processor
        self.processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL_NAME)

        # Load base Wav2Vec2 model
        logger.info(f"Loading base acoustic model: {BASE_MODEL_NAME}")
        self.base_model = Wav2Vec2ForCTC.from_pretrained(BASE_MODEL_NAME)
        self.base_model.eval()

        self.current_adapter_name = "base_model"
        self.current_adapter_path = None
        self.model = self.base_model

        # Determine and load best available adapter
        target_adapter = adapter_path or self._resolve_adapter_path(speaker_id)
        if target_adapter and os.path.exists(target_adapter):
            self._load_adapter(target_adapter)
        # Initialize pure Python Dysarthric Lexicon & Language Decoder
        try:
            self.lexicon_decoder = DysarthricLexiconDecoder()
            logger.info("✓ Dysarthric Lexicon & Language Decoder initialized")
        except Exception as e:
            logger.warning(f"Could not initialize DysarthricLexiconDecoder: {e}")
            self.lexicon_decoder = None

        self.model.to(self.device)

    def _resolve_adapter_path(self, speaker_id: Optional[str] = None) -> Optional[str]:
        """
        Determine the best adapter path according to hierarchy:
          1. Personalized speaker adapter (if speaker_id provided and exists)
          2. Generalized multi-speaker adapter (if exists)
          3. None (fall back to base model)
        """
        if speaker_id:
            spk_path = os.path.join(ADAPTER_DIR, f"speaker_{speaker_id}")
            if os.path.exists(spk_path):
                severity = SPEAKER_SEVERITY.get(speaker_id, "Unknown")
                logger.info(f"✓ Found personalized adapter for speaker: {speaker_id} [Severity: {severity}]")
                return spk_path
            else:
                logger.info(f"Personal adapter not found for '{speaker_id}', checking generalized adapter...")

        gen_path = os.path.join(ADAPTER_DIR, "generalized_adapter")
        if os.path.exists(gen_path):
            logger.info("✓ Found generalized dysarthric adapter")
            return gen_path

        return None

    def _load_adapter(self, adapter_path: str):
        """Load a LoRA adapter on top of the base model."""
        try:
            logger.info(f"Loading adapter weights from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
            self.model.eval()
            self.current_adapter_path = adapter_path
            self.current_adapter_name = os.path.basename(adapter_path.rstrip("/\\"))
            logger.info(f"✓ Active adapter set to: {self.current_adapter_name}")
        except Exception as e:
            logger.warning(f"Failed to load adapter from {adapter_path}: {e}")
            logger.info("Falling back to base model.")
            self.model = self.base_model
            self.current_adapter_name = "base_model"
            self.current_adapter_path = None

    def switch_adapter(self, speaker_id: Optional[str] = None, adapter_path: Optional[str] = None):
        """
        Dynamically switch active adapter without reloading base model weights.
        """
        if speaker_id == "__base__" or adapter_path == "__base__":
            logger.info("Switching to Base Wav2Vec2 Model (No Adapter)")
            self.model = self.base_model
            self.current_adapter_name = "base_model"
            self.current_adapter_path = None
            return

        target_path = adapter_path or self._resolve_adapter_path(speaker_id)
        if target_path and target_path != self.current_adapter_path:
            self._load_adapter(target_path)
            self.model.to(self.device)
        elif not target_path:
            logger.info("Reverting to base model (no adapter)")
            self.model = self.base_model
            self.current_adapter_name = "base_model"
            self.current_adapter_path = None

    def _prepare_audio(self, audio_input: Union[str, Path, np.ndarray, torch.Tensor], sr: Optional[int] = None) -> np.ndarray:
        """
        Standardize audio input to 16kHz mono float32 array, matching training preprocessing:
        1. Resamples to 16kHz
        2. Converts to Mono Float32
        3. Removes DC bias
        4. Trims leading and trailing silence
        5. Normalizes volume to target -20 dBFS RMS
        """
        if isinstance(audio_input, (str, Path)):
            audio_path = str(audio_input)
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            try:
                audio, orig_sr = sf.read(audio_path)
            except Exception as sf_err:
                try:
                    import torchaudio
                    wav_t, orig_sr = torchaudio.load(audio_path)
                    audio = wav_t.squeeze().cpu().numpy()
                except Exception:
                    try:
                        import librosa
                        audio, orig_sr = librosa.load(audio_path, sr=None)
                    except Exception:
                        raise RuntimeError(f"Could not decode audio from '{audio_path}': {sf_err}")
            if orig_sr != SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
        elif isinstance(audio_input, torch.Tensor):
            audio = audio_input.detach().cpu().numpy()
            if sr and sr != SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        elif isinstance(audio_input, np.ndarray):
            audio = audio_input.copy()
            if sr and sr != SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Convert to float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # 1. Remove DC offset and sub-bass microphone rumble (< 80 Hz)
        if len(audio) > 0:
            audio = audio - np.mean(audio)
        try:
            from scipy.signal import butter, filtfilt
            cutoff = HIGH_PASS_FILTER_HZ
            nyq = SAMPLE_RATE / 2.0
            b, a = butter(4, cutoff / nyq, btype="high")
            audio = filtfilt(b, a, audio).astype(np.float32)
        except Exception:
            pass

        # 2. Optional stationary noise reduction for consumer live microphones
        try:
            import noisereduce as nr
            audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE, stationary=True, prop_decrease=0.6)
        except Exception:
            pass

        # 3. Non-destructive silence trimming (40 dB threshold to protect soft hypophonic speech)
        try:
            import librosa
            trim_db = abs(SILENCE_THRESHOLD_DB)
            trimmed, _ = librosa.effects.trim(audio, top_db=trim_db)
            # Guard against over-trimming soft/hypophonic Parkinsonian speech:
            if len(trimmed) >= int(len(audio) * 0.40) and len(trimmed) >= int(SAMPLE_RATE * 0.25):
                audio = trimmed
        except Exception:
            pass

        # 4. Clamped RMS volume normalization (prevents exploding room hiss into clipping)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 1e-6:
            target_rms = 10.0 ** (NORMALIZE_TARGET_DB / 20.0)  # ~0.10 (-20 dBFS)
            gain = target_rms / rms
            max_gain_linear = 10.0 ** (MAX_NORMALIZATION_GAIN_DB / 20.0)  # ~15.8x (+24 dB max boost)
            gain = min(gain, max_gain_linear)
            audio = audio * gain

        # 5. Clip to [-1.0, 1.0] to prevent clipping
        audio = np.clip(audio, -1.0, 1.0)

        return audio

    def transcribe(
        self,
        audio_input: Union[str, Path, np.ndarray, torch.Tensor],
        speaker_id: Optional[str] = None,
        sample_rate: Optional[int] = None,
        use_lexicon: bool = ENABLE_LEXICON_RESCORING,
    ) -> Dict:
        """
        Transcribe audio to text.

        Args:
            audio_input: File path, NumPy array, or PyTorch tensor.
            speaker_id: Optional speaker ID to switch adapter if needed.
            sample_rate: Sampling rate if passing raw audio array.
            use_lexicon: Whether to apply dysarthric language and lexicon rescoring.

        Returns:
            dict containing:
              - 'transcription': Final transcribed text string (lexicon-rescored)
              - 'raw_transcription': Raw greedy CTC character output
              - 'confidence': Mean probability of predicted tokens (0.0 to 1.0)
              - 'adapter_used': Name of the adapter used for transcription
              - 'duration_sec': Audio length in seconds
              - 'lexicon_applied': Boolean flag indicating if lexicon correction was active
        """
        if speaker_id:
            self.switch_adapter(speaker_id=speaker_id)

        audio = self._prepare_audio(audio_input, sr=sample_rate)
        duration_sec = len(audio) / SAMPLE_RATE

        # Preprocess features
        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(input_values).logits

        # Greedy CTC decoding
        predicted_ids = torch.argmax(logits, dim=-1)
        raw_transcription = self.processor.batch_decode(predicted_ids)[0].strip().lower()

        # Apply Dysarthric Lexicon & Language Rescorer
        if use_lexicon and self.lexicon_decoder is not None:
            final_transcription = self.lexicon_decoder.decode_text(raw_transcription)
        else:
            final_transcription = raw_transcription

        # Calculate token-level confidence
        probs = torch.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values
        confidence = float(max_probs.mean().item())

        source_info = str(audio_input) if isinstance(audio_input, (str, Path)) else "raw_audio_array"

        return {
            "transcription": final_transcription,
            "raw_transcription": raw_transcription,
            "confidence": round(confidence, 4),
            "adapter_used": self.current_adapter_name,
            "duration_sec": round(duration_sec, 3),
            "source": source_info,
            "lexicon_applied": (use_lexicon and final_transcription != raw_transcription),
        }

    def transcribe_with_timestamps(
        self,
        audio_input: Union[str, Path, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
        use_lexicon: bool = ENABLE_LEXICON_RESCORING,
    ) -> Dict:
        """
        Transcribe speech and return word-level timestamps and confidences.
        Crucial for clinical speech pathology analysis (rate, pauses, prolongations).
        """
        audio = self._prepare_audio(audio_input, sr=sample_rate)
        duration_sec = len(audio) / SAMPLE_RATE

        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits

        probs = torch.softmax(logits, dim=-1)[0]
        predicted_ids = torch.argmax(probs, dim=-1).cpu().numpy()
        tokens = self.processor.tokenizer.convert_ids_to_tokens(predicted_ids)

        # Wav2Vec2 downsamples 320x (16kHz / 320 = 50 frames/sec = 20ms per frame)
        time_stride = 0.020

        words = []
        current_word_chars = []
        current_word_start = None
        current_word_char_probs = []
        prev_token = None

        for frame_idx, (token, token_id) in enumerate(zip(tokens, predicted_ids)):
            frame_time = frame_idx * time_stride
            frame_prob = float(probs[frame_idx, token_id].item())

            if token == "<pad>":
                prev_token = "<pad>"
                continue

            if token == "|":
                if current_word_chars:
                    word_str = "".join(current_word_chars).lower()
                    words.append({
                        "word": word_str,
                        "start_time": round(current_word_start, 3),
                        "end_time": round(frame_time, 3),
                        "confidence": round(float(np.mean(current_word_char_probs)), 3),
                    })
                    current_word_chars = []
                    current_word_char_probs = []
                    current_word_start = None
                prev_token = "|"
                continue

            # Non-blank character token
            if current_word_start is None:
                current_word_start = frame_time

            # CTC collapse: identical consecutive frames represent the same phoneme
            if token == prev_token and current_word_char_probs:
                current_word_char_probs[-1] = max(current_word_char_probs[-1], frame_prob)
            else:
                current_word_chars.append(token)
                current_word_char_probs.append(frame_prob)

            prev_token = token

        # Flush trailing word
        if current_word_chars:
            word_str = "".join(current_word_chars).lower()
            words.append({
                "word": word_str,
                "start_time": round(current_word_start or 0.0, 3),
                "end_time": round(duration_sec, 3),
                "confidence": round(float(np.mean(current_word_char_probs)), 3),
            })

        raw_text = " ".join(w["word"] for w in words)

        # Apply Dysarthric Lexicon Decoder to words & timestamps
        if use_lexicon and self.lexicon_decoder is not None:
            corrected_words = self.lexicon_decoder.decode_word_timestamps(words)
            final_text = " ".join(w["word"] for w in corrected_words)
        else:
            corrected_words = words
            final_text = raw_text

        overall_conf = (
            round(float(np.mean([w["confidence"] for w in corrected_words])), 4)
            if corrected_words else round(float(probs.max(dim=-1).values.mean().item()), 4)
        )

        return {
            "transcription": final_text,
            "raw_transcription": raw_text,
            "words": corrected_words,
            "confidence": overall_conf,
            "duration_sec": round(duration_sec, 3),
            "adapter_used": self.current_adapter_name,
            "lexicon_applied": (use_lexicon and final_text != raw_text),
        }

    def transcribe_batch(
        self,
        audio_paths: List[str],
        speaker_id: Optional[str] = None,
    ) -> List[Dict]:
        """Transcribe multiple audio files."""
        results = []
        for i, path in enumerate(audio_paths):
            try:
                res = self.transcribe(path, speaker_id=speaker_id)
                results.append(res)
                if (i + 1) % 10 == 0:
                    logger.info(f"  Transcribed {i+1}/{len(audio_paths)} files")
            except Exception as e:
                logger.error(f"Error transcribing {path}: {e}")
                results.append({
                    "transcription": "[ERROR]",
                    "confidence": 0.0,
                    "adapter_used": self.current_adapter_name,
                    "source": path,
                    "error": str(e),
                })
        return results

    @staticmethod
    def list_available_adapters() -> Dict:
        """Scan and return all available adapters on disk."""
        gen_path = os.path.join(ADAPTER_DIR, "generalized_adapter")
        has_gen = os.path.exists(gen_path)

        speaker_adapters = {}
        if os.path.exists(ADAPTER_DIR):
            for entry in os.listdir(ADAPTER_DIR):
                if entry.startswith("speaker_"):
                    spk_id = entry[len("speaker_"):]
                    speaker_adapters[spk_id] = {
                        "path": os.path.join(ADAPTER_DIR, entry),
                        "severity": SPEAKER_SEVERITY.get(spk_id, "Unknown"),
                    }

        return {
            "generalized_adapter": {"path": gen_path, "available": has_gen},
            "speaker_adapters": speaker_adapters,
            "count_speaker_adapters": len(speaker_adapters),
        }

    def evaluate(
        self,
        manifest_path: Optional[str] = None,
        split: str = "test",
        speaker_id: Optional[str] = None,
    ) -> Dict:
        """
        Compute WER and CER over a manifest split.
        """
        import pandas as pd
        import evaluate
        wer_metric = evaluate.load("wer")
        cer_metric = evaluate.load("cer")

        manifest_file = manifest_path or os.path.join(MANIFEST_DIR, "split_manifest.csv")
        if not os.path.exists(manifest_file):
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")

        df = pd.read_csv(manifest_file)
        if split:
            df = df[df["split"] == split]
        if speaker_id:
            df = df[df["speaker_id"] == speaker_id]

        df = df[df["audio_path"].apply(os.path.exists)]
        logger.info(f"Evaluating {len(df)} samples (split='{split}', speaker='{speaker_id or 'ALL'}')...")

        predictions = []
        references = []

        for _, row in df.iterrows():
            res = self.transcribe(row["audio_path"])
            predictions.append(res["transcription"].strip())
            references.append(str(row["transcript"]).strip())

        wer = wer_metric.compute(predictions=predictions, references=references)
        cer = cer_metric.compute(predictions=predictions, references=references)

        return {
            "eval_split": split,
            "speaker_id": speaker_id or "ALL",
            "samples_evaluated": len(df),
            "wer": round(float(wer), 4),
            "cer": round(float(cer), 4),
            "adapter_used": self.current_adapter_name,
        }


# Direct alias for clean user imports
DysarthricASR = DysarthricSpeechTranscriber
SpeechToTextModel = DysarthricSpeechTranscriber


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dysarthric Speech-to-Text Model")
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to audio file (.wav)")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Directory of .wav audio files to transcribe in batch")
    parser.add_argument("--speaker_id", type=str, default=None,
                        help="Speaker ID (e.g., 'F3_M01') for personalized adapter routing")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Explicit path to adapter directory")
    parser.add_argument("--timestamps", action="store_true",
                        help="Output word-level timestamps and confidences")
    parser.add_argument("--list_adapters", action="store_true",
                        help="List all available trained adapters")
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation on the test split")
    parser.add_argument("--split", type=str, default="test",
                        help="Split to evaluate: 'test', 'val', or 'train'")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save output results JSON")

    args = parser.parse_args()

    # List adapters if requested
    if args.list_adapters:
        info = DysarthricASR.list_available_adapters()
        print("\n" + "=" * 60)
        print("AVAILABLE ADAPTERS:")
        print("=" * 60)
        print(f"Generalized Adapter: {'Available' if info['generalized_adapter']['available'] else 'Not trained yet'}")
        print(f"Personalized Speaker Adapters: {info['count_speaker_adapters']}")
        for spk, meta in sorted(info["speaker_adapters"].items()):
            print(f"  - {spk:12s} [{meta['severity']:8s}] -> {meta['path']}")
        print("=" * 60)
        return

    # Initialize transcriber
    transcriber = DysarthricASR(adapter_path=args.adapter, speaker_id=args.speaker_id)

    # Run evaluation if requested
    if args.eval:
        res = transcriber.evaluate(split=args.split, speaker_id=args.speaker_id)
        print("\n" + "=" * 60)
        print(f"EVALUATION RESULTS (Split: {args.split})")
        print("=" * 60)
        print(f"  Samples:  {res['samples_evaluated']}")
        print(f"  WER:      {res['wer'] * 100:.2f}%")
        print(f"  CER:      {res['cer'] * 100:.2f}%")
        print(f"  Adapter:  {res['adapter_used']}")
        print("=" * 60)
        return

    # Transcribe single file
    if args.audio:
        if args.timestamps:
            res = transcriber.transcribe_with_timestamps(args.audio)
            print("\n" + "=" * 60)
            print(f"Transcription: {res['transcription']}")
            print(f"Duration:      {res['duration_sec']}s")
            print(f"Adapter:       {res['adapter_used']}")
            print("Words:")
            for w in res["words"]:
                print(f"  [{w['start_time']:5.2f}s - {w['end_time']:5.2f}s] {w['word']:15s} (conf: {w['confidence']:.2f})")
            print("=" * 60)
        else:
            res = transcriber.transcribe(args.audio)
            print("\n" + "=" * 60)
            print(f"Audio File:    {os.path.basename(args.audio)}")
            print(f"Transcription: {res['transcription']}")
            if res.get("lexicon_applied"):
                print(f"Raw CTC Output:{res.get('raw_transcription')} (Lexicon Corrected)")
            print(f"Confidence:    {res['confidence'] * 100:.1f}%")
            print(f"Adapter Used:  {res['adapter_used']}")
            print("=" * 60)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(res, f, indent=2)

    # Transcribe directory
    elif args.audio_dir:
        wavs = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
        if not wavs:
            logger.error(f"No .wav files found in: {args.audio_dir}")
            return
        results = transcriber.transcribe_batch(wavs, speaker_id=args.speaker_id)
        print("\n" + "=" * 60)
        for r in results:
            print(f"[{os.path.basename(r['source'])}] {r['transcription']} (conf: {r.get('confidence', 0):.2f})")
        print("=" * 60)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved results to: {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
