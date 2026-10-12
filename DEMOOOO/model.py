"""
=============================================================================
Dysarthric Speech-to-Text Model for Parkinson's Disease (Direct API)
=============================================================================
Top-level model module providing direct access to the Parkinson's ASR system.

Usage:
    from model import SpeechToTextModel

    # 1. Initialize model (automatically detects best adapter or base model)
    asr = SpeechToTextModel()

    # 2. Or initialize with a specific speaker's personalized adapter:
    # asr = SpeechToTextModel(speaker_id="F3_M01")

    # 3. Transcribe speech audio (supports file path, numpy array, or tensor):
    result = asr.transcribe("sample.wav")
    print(f"Transcribed Text: {result['transcription']}")
    print(f"Confidence:       {result['confidence']}")
    print(f"Adapter:          {result['adapter_used']}")

    # 4. Transcribe with word-level timestamps (for clinical speech rate analysis):
    detailed = asr.transcribe_with_timestamps("sample.wav")
    for word_info in detailed['words']:
        print(f"[{word_info['start_time']}s - {word_info['end_time']}s] {word_info['word']}")
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from step6_inference import DysarthricSpeechTranscriber, DysarthricASR, SpeechToTextModel

__all__ = ["SpeechToTextModel", "DysarthricASR", "DysarthricSpeechTranscriber"]


if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("Dysarthric Speech-to-Text Model Initialized")
    print("=" * 60)
    model = SpeechToTextModel()
    adapters = model.list_available_adapters()
    print(f"Base Acoustic Model:  facebook/wav2vec2-base-960h (16kHz)")
    print(f"Generalized Adapter:  {'Ready' if adapters['generalized_adapter']['available'] else 'Not trained yet (using base)'}")
    print(f"Personalized Adapters: {adapters['count_speaker_adapters']} speakers")
    print("\nReady for transcription. Call model.transcribe('audio.wav')")
    print("=" * 60)
