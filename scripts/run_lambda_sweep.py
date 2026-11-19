"""
Script to execute the lambda_max sweep in {0.05, 0.1, 0.3, 0.5, 1.0}
and fixed vs scheduled comparison.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import yaml
import pandas as pd

from train import run_training
from eval import evaluate_checkpoint
from probe import run_emotion_probe


def run_sweep(base_config_path: Path, lambda_values=[0.05, 0.1, 0.3, 0.5, 1.0], epochs=15, seed=42):
    with open(base_config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    results_records = []
    manifest_p = Path(base_cfg.get("data", {}).get("manifest_path", "data/manifest.csv"))

    for l_max in lambda_values:
        for scheduled in [True, False]:
            mode_str = "sched" if scheduled else "fixed"
            run_name = f"grl_lambda_{l_max}_{mode_str}"
            out_dir = Path(f"runs/{run_name}")
            
            cfg = yaml.safe_load(yaml.dump(base_cfg))
            cfg["output_dir"] = str(out_dir)
            cfg.setdefault("train", {})["epochs"] = epochs
            cfg["train"]["lambda_max"] = l_max
            cfg["train"]["schedule_lambda"] = scheduled
            if not scheduled:
                cfg["train"]["fixed_lambda"] = l_max

            print(f"\n=======================================================")
            print(f"RUNNING: Lambda={l_max} | Scheduled={scheduled}")
            print(f"=======================================================")

            # 1. Train
            run_training(cfg, output_dir=out_dir, seed=seed)

            # 2. Evaluate on test split
            best_ckpt = out_dir / "best.pt"
            eval_res = evaluate_checkpoint(best_ckpt, manifest_p, split="test")

            # 3. Run emotion probe
            probe_res = run_emotion_probe(best_ckpt, manifest_p, train_split="train", test_split="test")

            record = {
                "lambda_max": l_max,
                "scheduled": scheduled,
                "global_eer": eval_res["global_eer"],
                "ci_95_low": eval_res["ci_95"][0],
                "ci_95_high": eval_res["ci_95"][1],
                "emotion_mean": eval_res.get("emotion_stats", {}).get("mean", 0.0),
                "emotion_range": eval_res.get("emotion_stats", {}).get("range", 0.0),
                "emotion_sigma": eval_res.get("emotion_stats", {}).get("sigma", 0.0),
                "probe_acc": probe_res["probe_accuracy"],
                "probe_balanced_acc": probe_res["balanced_accuracy"]
            }
            # Add per-emotion EERs
            for emo, eer in eval_res.get("per_emotion", {}).items():
                record[f"eer_{emo}"] = eer
            # Add per-system EERs
            for sys_id, eer in eval_res.get("per_system", {}).items():
                record[f"eer_{sys_id}"] = eer

            results_records.append(record)
            print(f"--> Done: Spoof EER={record['global_eer']:.2f}% | Sigma={record['emotion_sigma']:.2f}% | Probe Acc={record['probe_acc']:.2f}%")

    df_sweep = pd.DataFrame(results_records)
    Path("results").mkdir(exist_ok=True)
    csv_out = Path("results/lambda_sweep_results.csv")
    df_sweep.to_csv(csv_out, index=False)
    print(f"\nSaved all sweep results to {csv_out}")
    print(df_sweep[["lambda_max", "scheduled", "global_eer", "emotion_sigma", "probe_acc"]])
    return df_sweep


if __name__ == "__main__":
    run_sweep(Path("configs/proposed_grl_scheduled.yaml"), lambda_values=[0.05, 0.1, 0.3, 0.5, 1.0], epochs=12)
