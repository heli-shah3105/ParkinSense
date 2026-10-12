"""
=============================================================================
Clinical Benchmarking & Comparative Evaluation — Dysarthric ASR
=============================================================================
Evaluates and compares:
  1. Baseline Pretrained Wav2Vec2 (Zero-shot, no adaptation)
  2. Generalized Adapter (Trained on pooled dysarthric speech)
  3. Personalized Adapters (Fine-tuned per speaker)

Produces:
  - Stratified metrics by Clinical Severity (Mild, Moderate, Severe, Control)
  - Relative Error Reduction (RER %)
  - Word Error Rate (WER) & Character Error Rate (CER)
  - Markdown report: output/logs/clinical_benchmark_report.md
  - CSV report: output/logs/clinical_benchmark_report.csv

Usage:
  python benchmark.py                 # Full benchmark across all test splits
  python benchmark.py --heldout_only   # Evaluate only unseen held-out speakers
  python benchmark.py --speaker F3_M01 # Evaluate specific speaker
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
try:
    import evaluate
except ImportError:
    evaluate = None
import jiwer

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from peft import PeftModel

from config import (
    BASE_MODEL_NAME, SAMPLE_RATE, ADAPTER_DIR,
    MANIFEST_DIR, LOGS_DIR, SPEAKER_SEVERITY
)
from step6_inference import DysarthricSpeechTranscriber

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class DysarthricASRBenchmark:
    """
    Comprehensive benchmarking suite for dysarthric speech recognition.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing Benchmark Suite on: {self.device}")

        # Metrics
        self.evaluate = evaluate

        # Load processor
        self.processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL_NAME)

        # Transcriber instance
        self.transcriber = DysarthricSpeechTranscriber(device=self.device)

        # Manifest
        manifest_path = os.path.join(MANIFEST_DIR, "split_manifest.csv")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run steps 1-3 first!")
        self.df = pd.read_csv(manifest_path)
        self.df["severity"] = self.df["speaker_id"].map(SPEAKER_SEVERITY).fillna("Unknown")

    def evaluate_dataset(self, df_subset: pd.DataFrame, desc: str = "") -> Dict:
        """Evaluate transcriber on a given dataframe subset."""
        if len(df_subset) == 0:
            return {"wer": None, "cer": None, "samples": 0}

        predictions = []
        references = []

        for _, row in df_subset.iterrows():
            audio_path = row["audio_path"]
            if not os.path.exists(audio_path):
                continue
            ref = str(row["transcript"]).strip().lower()
            if not ref:
                continue

            res = self.transcriber.transcribe(audio_path)
            pred = res["transcription"].strip().lower()

            predictions.append(pred)
            references.append(ref)

        if not references:
            return {"wer": 1.0, "cer": 1.0, "samples": 0}

        try:
            if self.evaluate is not None:
                wer = self.evaluate.load("wer").compute(predictions=predictions, references=references)
                cer = self.evaluate.load("cer").compute(predictions=predictions, references=references)
            else:
                wer = jiwer.wer(reference=references, hypothesis=predictions)
                cer = jiwer.cer(reference=references, hypothesis=predictions)
        except Exception:
            wer = jiwer.wer(reference=references, hypothesis=predictions)
            cer = jiwer.cer(reference=references, hypothesis=predictions)

        return {
            "wer": round(float(wer), 4),
            "cer": round(float(cer), 4),
            "samples": len(references),
            "description": desc,
        }

    def run_benchmark(self, heldout_only: bool = False, target_speaker: Optional[str] = None) -> pd.DataFrame:
        """Run full comparative benchmark across baseline, generalized, and personalized models."""
        logger.info("=" * 70)
        logger.info("STARTING CLINICAL DYSARTHRIC ASR BENCHMARK")
        logger.info("=" * 70)

        # Filter test samples
        test_df = self.df[self.df["split"] == "test"].copy()
        if target_speaker:
            test_df = test_df[test_df["speaker_id"] == target_speaker]
        elif heldout_only:
            test_df = test_df[test_df["speaker_group"] == "heldout_test"]

        speakers = sorted(test_df["speaker_id"].unique())
        logger.info(f"Evaluating {len(test_df)} test samples across {len(speakers)} speakers:")
        for spk in speakers:
            n_spk = len(test_df[test_df["speaker_id"] == spk])
            sev = SPEAKER_SEVERITY.get(spk, "Unknown")
            grp = test_df[test_df["speaker_id"] == spk]["speaker_group"].iloc[0]
            logger.info(f"  - {spk:12s} [{sev:8s}] [{grp:12s}] {n_spk} test samples")

        gen_adapter_path = os.path.join(ADAPTER_DIR, "generalized_adapter")
        has_gen_adapter = os.path.exists(gen_adapter_path)

        results = []

        for spk in speakers:
            spk_df = test_df[test_df["speaker_id"] == spk]
            sev = SPEAKER_SEVERITY.get(spk, "Unknown")
            grp = spk_df["speaker_group"].iloc[0]
            n_samples = len(spk_df)

            # 1. Evaluate BASELINE (Pretrained Wav2Vec2 Zero-Shot - No Adapter)
            self.transcriber.switch_adapter(adapter_path="__base__")
            base_metrics = self.evaluate_dataset(spk_df, desc=f"{spk} - Baseline")
            base_wer = base_metrics["wer"]
            base_cer = base_metrics["cer"]

            # 2. Evaluate GENERALIZED ADAPTER
            if has_gen_adapter:
                self.transcriber.switch_adapter(adapter_path=gen_adapter_path)
                gen_metrics = self.evaluate_dataset(spk_df, desc=f"{spk} - Generalized")
                gen_wer = gen_metrics["wer"]
                gen_cer = gen_metrics["cer"]
            else:
                gen_wer = None
                gen_cer = None

            # 3. Evaluate PERSONALIZED ADAPTER
            spk_adapter_path = os.path.join(ADAPTER_DIR, f"speaker_{spk}")
            if os.path.exists(spk_adapter_path):
                self.transcriber.switch_adapter(adapter_path=spk_adapter_path)
                pers_metrics = self.evaluate_dataset(spk_df, desc=f"{spk} - Personalized")
                pers_wer = pers_metrics["wer"]
                pers_cer = pers_metrics["cer"]
            else:
                pers_wer = None
                pers_cer = None

            # Compute Relative Error Reduction (RER)
            best_wer = pers_wer if pers_wer is not None else gen_wer
            rer = None
            if base_wer is not None and best_wer is not None and base_wer > 0:
                rer = round(((base_wer - best_wer) / base_wer) * 100, 2)

            results.append({
                "speaker_id": spk,
                "group": grp,
                "severity": sev,
                "test_samples": n_samples,
                "baseline_wer": base_wer,
                "baseline_cer": base_cer,
                "generalized_wer": gen_wer,
                "generalized_cer": gen_cer,
                "personalized_wer": pers_wer,
                "personalized_cer": pers_cer,
                "best_adapter_wer": best_wer,
                "wer_reduction_pct": rer,
            })

            logger.info(
                f"[{spk}] Sev: {sev:8s} | Base WER: {base_wer*100 if base_wer else 0:5.1f}% | "
                f"Gen WER: {gen_wer*100 if gen_wer else 0:5.1f}% | "
                f"Pers WER: {pers_wer*100 if pers_wer else 0:5.1f}% | "
                f"RER: {f'{rer:+.1f}%' if rer else 'N/A'}"
            )

        report_df = pd.DataFrame(results)

        # Save CSV
        csv_path = os.path.join(LOGS_DIR, "clinical_benchmark_report.csv")
        report_df.to_csv(csv_path, index=False)
        logger.info(f"\nSaved CSV report to: {csv_path}")

        # Generate Markdown Report
        self._generate_markdown_report(report_df)

        return report_df

    def _generate_markdown_report(self, df: pd.DataFrame):
        """Format results into a clean clinical benchmark markdown table."""
        md_lines = [
            "# Clinical Dysarthric Speech Recognition Benchmark Report",
            "",
            "## 1. Executive Summary",
            "",
            "This report quantifies speech-to-text recognition accuracy across dysarthric speakers with Parkinson's Disease.",
            "Performance is evaluated across three model tiers:",
            "- **Tier 1 (Baseline)**: Pretrained Wav2Vec2 Base (zero-shot, no dysarthria adaptation).",
            "- **Tier 2 (Generalized Adapter)**: Shared LoRA adapter trained across pooled dysarthric speakers.",
            "- **Tier 3 (Personalized Adapter)**: Patient-specific LoRA adapter fine-tuned starting from generalized weights.",
            "",
            "## 2. Per-Speaker Comparative Metrics",
            "",
            "| Speaker ID | Group | Severity | Samples | Baseline WER | Generalized WER | Personal WER | WER Reduction (RER) |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
        ]

        for _, row in df.iterrows():
            b_wer = f"{row['baseline_wer']*100:.1f}%" if pd.notnull(row['baseline_wer']) else "N/A"
            g_wer = f"{row['generalized_wer']*100:.1f}%" if pd.notnull(row['generalized_wer']) else "N/A"
            p_wer = f"{row['personalized_wer']*100:.1f}%" if pd.notnull(row['personalized_wer']) else "N/A"
            rer_str = f"**{row['wer_reduction_pct']:+.1f}%**" if pd.notnull(row['wer_reduction_pct']) else "N/A"

            md_lines.append(
                f"| `{row['speaker_id']}` | {row['group']} | **{row['severity']}** | {row['test_samples']} | "
                f"{b_wer} | {g_wer} | {p_wer} | {rer_str} |"
            )

        # Severity stratification summary
        md_lines.extend([
            "",
            "## 3. Stratified Accuracy by Clinical Severity",
            "",
            "| Severity Level | Speakers | Total Test Samples | Mean Baseline WER | Mean Best Adapter WER | Mean Relative Error Reduction |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])

        for sev, grp in df.groupby("severity"):
            spk_cnt = len(grp)
            total_samples = grp["test_samples"].sum()
            mean_base = grp["baseline_wer"].dropna().mean()
            mean_best = grp["best_adapter_wer"].dropna().mean()
            mean_rer = grp["wer_reduction_pct"].dropna().mean()

            m_base_str = f"{mean_base*100:.1f}%" if pd.notnull(mean_base) else "N/A"
            m_best_str = f"{mean_best*100:.1f}%" if pd.notnull(mean_best) else "N/A"
            m_rer_str = f"**{mean_rer:+.1f}%**" if pd.notnull(mean_rer) else "N/A"

            md_lines.append(
                f"| **{sev}** | {spk_cnt} | {total_samples} | {m_base_str} | {m_best_str} | {m_rer_str} |"
            )

        md_path = os.path.join(LOGS_DIR, "clinical_benchmark_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        logger.info(f"Saved Markdown report to: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Dysarthric ASR Clinical Benchmark")
    parser.add_argument("--heldout_only", action="store_true",
                        help="Benchmark only unseen held-out speakers")
    parser.add_argument("--speaker", type=str, default=None,
                        help="Benchmark a specific speaker ID")
    args = parser.parse_args()

    benchmark = DysarthricASRBenchmark()
    report = benchmark.run_benchmark(heldout_only=args.heldout_only, target_speaker=args.speaker)
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETED")
    print(f"Results saved to: {os.path.join(LOGS_DIR, 'clinical_benchmark_report.md')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
