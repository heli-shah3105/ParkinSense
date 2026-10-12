"""
=============================================================================
check_labels.py - Run this BEFORE training (takes ~2-5 minutes)
=============================================================================
Transcribes a few clips per speaker with the UNMODIFIED pretrained model and
compares them to the manifest transcripts.

How to read the result:
  * CONTROL speakers (non-dysarthric) should get a clearly-below-1.0 WER.
    If a control speaker has ~100% WER and the hypotheses look like real words
    that simply differ from the reference, the TRANSCRIPTS ARE WRONG for that
    speaker (audio and text are not aligned) and no amount of training will fix it.
  * Dysarthric speakers will be worse than controls - that is expected.

Usage:
  python check_labels.py              # 20 clips per speaker
  python check_labels.py --n 40
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import sys
import argparse
import numpy as np
import pandas as pd
import soundfile as sf

try:
    import torch
    import jiwer
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
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

from config import BASE_MODEL_NAME, SAMPLE_RATE, MANIFEST_DIR, SPEAKER_SEVERITY, RANDOM_SEED


def sanitize(text, allowed):
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    t = "".join(ch if (ch in allowed or ch == " ") else " " for ch in str(text).upper())
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="clips per speaker")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(BASE_MODEL_NAME).to(device).eval()
    allowed = {t for t in processor.tokenizer.get_vocab() if len(t) == 1 and t != "|"}

    df = pd.read_csv(os.path.join(MANIFEST_DIR, "split_manifest.csv"))
    df = df[df["audio_path"].apply(os.path.exists)]

    # ---- Manifest-level red flags (no model needed) ----------------------
    print("\n=== MANIFEST OVERVIEW ===")
    overview = df.groupby("source_folder").agg(
        clips=("audio_path", "count"),
        speakers=("speaker_id", "nunique"),
        unique_transcripts=("transcript", "nunique"),
    )
    print(overview.to_string())
    print("\n(If a folder has MANY clips but very FEW unique transcripts per speaker group, "
          "labels were probably assigned by a lookup that does not match the audio.)")
    print("\nSpeakers per source folder:")
    for folder, g in df.groupby("source_folder"):
        print(f"  {folder}: {sorted(g['speaker_id'].unique())}")
    if not df["source_folder"].str.contains("folder2").any():
        print("\n!! WARNING: no speakers from folder 2 are in the manifest at all. "
              "Check output/logs/01_build_manifest.log and 02_preprocessing.log "
              "('File not found' count) - that data is currently unused.")

    # ---- Zero-shot WER per speaker ---------------------------------------
    print("\n=== ZERO-SHOT WER PER SPEAKER (pretrained model, no adapter) ===")
    rows, shown = [], 0
    for spk, g in df.groupby("speaker_id"):
        g = g.sample(n=min(args.n, len(g)), random_state=RANDOM_SEED)
        refs, hyps = [], []
        for _, r in g.iterrows():
            ref = sanitize(r["transcript"], allowed)
            if not ref:
                continue
            audio, sr = sf.read(r["audio_path"])
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            inp = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(device)
            with torch.no_grad():
                ids = torch.argmax(model(inp).logits, dim=-1)
            hyps.append(processor.batch_decode(ids)[0].strip())
            refs.append(ref)
        if not refs:
            continue
        wer = jiwer.wer(refs, hyps)
        cer = jiwer.cer(refs, hyps)
        rows.append((spk, SPEAKER_SEVERITY.get(spk, "Unknown"), len(refs), wer, cer))
        print(f"\n{spk:10s} [{SPEAKER_SEVERITY.get(spk, 'Unknown'):8s}] n={len(refs):3d}  WER={wer:.2f}  CER={cer:.2f}")
        for ref, hyp in list(zip(refs, hyps))[:4]:
            print(f"    REF: {ref!r:40s} HYP: {hyp!r}")

    print("\n=== SUMMARY ===")
    out = pd.DataFrame(rows, columns=["speaker", "severity", "n", "WER", "CER"]).sort_values("WER")
    print(out.to_string(index=False))
    ctrl = out[out["severity"] == "Control"]
    if len(ctrl):
        print(f"\nMean control WER: {ctrl['WER'].mean():.2f}")
        if ctrl["WER"].mean() > 0.8:
            print("!! Control speakers should NOT be this bad. Suspect mislabeled transcripts "
                  "(check how transcripts are assigned in step1_build_manifest.py) before training.")


if __name__ == "__main__":
    main()