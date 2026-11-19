"""
Runs all systematic ablations, collects evaluation metrics and probe accuracies,
and exports results/ablation_table.csv.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import pandas as pd
from train import run_training
from eval import evaluate_checkpoint
from probe import run_emotion_probe


ABLATION_CONFIGS = [
    ("Ablation 1: No Emotion Supervision", "configs/ablation1_no_emotion.yaml"),
    ("Ablation 2: Multi-task without GRL", "configs/ablation2_no_grl.yaml"),
    ("Proposed: GRL Scheduled (lambda=0.5)", "configs/proposed_grl_scheduled.yaml"),
    ("Proposed: GRL Fixed (lambda=0.5)", "configs/proposed_grl_fixed.yaml"),
    ("Ablation 4: Mel Spectrogram Only", "configs/ablation4_mel_only.yaml"),
    ("Ablation 4: MFCC Only", "configs/ablation4_mfcc_only.yaml"),
    ("Ablation 6: No Augmentation", "configs/ablation6_no_aug.yaml"),
    ("Baseline: RawNet2", "configs/baseline_rawnet2.yaml"),
    ("Upgrade: SSL-AASIST + GRL", "configs/upgrade_ssl_aasist.yaml"),
]


def main():
    records = []
    manifest_p = Path("data/manifest.csv")

    for name, cfg_path in ABLATION_CONFIGS:
        p = Path(cfg_path)
        if not p.exists():
            continue

        print(f"\n=======================================================")
        print(f"RUNNING EXPERIMENT: {name}")
        print(f"=======================================================")

        with open(p, "r") as f:
            cfg = yaml.safe_load(f)

        out_dir = Path(cfg.get("output_dir", f"runs/{p.stem}"))
        best_ckpt = out_dir / "best.pt"

        # Check if already trained or train now
        if not best_ckpt.exists():
            cfg.setdefault("train", {})["epochs"] = 12
            run_training(cfg, output_dir=out_dir, seed=42)

        # Evaluate
        eval_res = evaluate_checkpoint(best_ckpt, manifest_p, split="test")

        # Probe (if model extracts embeddings)
        try:
            probe_res = run_emotion_probe(best_ckpt, manifest_p, train_split="train", test_split="test")
            probe_acc = probe_res["probe_accuracy"]
            probe_bal = probe_res["balanced_accuracy"]
        except Exception as e:
            print(f"Notice: Probe not applicable for {name}: {e}")
            probe_acc = None
            probe_bal = None

        rec = {
            "Experiment": name,
            "Architecture": cfg.get("architecture"),
            "Features": cfg.get("data", {}).get("feature_type", "raw"),
            "Use GRL": cfg.get("model", {}).get("use_grl", False),
            "Lambda Max": cfg.get("train", {}).get("lambda_max", 0.0),
            "Augmentation": cfg.get("data", {}).get("aug_noise", False),
            "Global EER (%)": eval_res["global_eer"],
            "EER 95% CI": f"[{eval_res['ci_95'][0]:.2f}, {eval_res['ci_95'][1]:.2f}]",
            "Emotion Mean EER (%)": eval_res.get("emotion_stats", {}).get("mean", 0.0),
            "Emotion Sigma (%)": eval_res.get("emotion_stats", {}).get("sigma", 0.0),
            "Emotion Range (%)": eval_res.get("emotion_stats", {}).get("range", 0.0),
            "Probe Accuracy (%)": probe_acc if probe_acc is not None else "N/A"
        }
        # Add per-emotion
        for emo in ["neutral", "happy", "angry", "sad", "surprise"]:
            rec[f"EER_{emo.capitalize()} (%)"] = eval_res.get("per_emotion", {}).get(emo, 0.0)

        records.append(rec)

    # Also add AASIST baseline from results/aasist_baseline_eval.json
    import json
    aasist_res_p = Path("results/aasist_baseline_eval.json")
    if aasist_res_p.exists():
        with open(aasist_res_p) as f:
            a_res = json.load(f)
        a_rec = {
            "Experiment": "Baseline: Pretrained AASIST (clovaai)",
            "Architecture": "aasist",
            "Features": "raw",
            "Use GRL": False,
            "Lambda Max": 0.0,
            "Augmentation": False,
            "Global EER (%)": a_res["global_eer"],
            "EER 95% CI": f"[{a_res['ci_95'][0]:.2f}, {a_res['ci_95'][1]:.2f}]",
            "Emotion Mean EER (%)": a_res.get("emotion_stats", {}).get("mean", 0.0),
            "Emotion Sigma (%)": a_res.get("emotion_stats", {}).get("sigma", 0.0),
            "Emotion Range (%)": a_res.get("emotion_stats", {}).get("range", 0.0),
            "Probe Accuracy (%)": "N/A"
        }
        for emo in ["neutral", "happy", "angry", "sad", "surprise"]:
            a_rec[f"EER_{emo.capitalize()} (%)"] = a_res.get("per_emotion", {}).get(emo, 0.0)
        records.insert(0, a_rec)

    df_abl = pd.DataFrame(records)
    Path("results").mkdir(exist_ok=True)
    out_csv = Path("results/ablation_table.csv")
    df_abl.to_csv(out_csv, index=False)
    print(f"\n=======================================================")
    print(f"ABLATION SUMMARY TABLE SAVED TO: {out_csv}")
    print(f"=======================================================")
    print(df_abl[["Experiment", "Global EER (%)", "Emotion Sigma (%)", "Probe Accuracy (%)"]])


if __name__ == "__main__":
    main()
