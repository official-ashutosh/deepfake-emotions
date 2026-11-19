"""
Train advanced optimized model with RawBoost and full convergence on complete EmoFake.
Evaluates on test, runs emotion probe, updates tables and plots, and pushes to GitHub.
"""

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import subprocess
import yaml
import numpy as np
import pandas as pd
import torch

from train import run_training
from eval import evaluate_checkpoint
from probe import run_emotion_probe
from analysis.visualize import (
    plot_embeddings_tsne,
    plot_per_emotion_eer_bar
)


def run_advanced_training():
    os.chdir(PROJECT_ROOT)
    manifest_path = PROJECT_ROOT / "data" / "manifest_emofake.csv"
    cfg_path = PROJECT_ROOT / "configs" / "proposed_grl_advanced.yaml"
    out_dir = PROJECT_ROOT / "runs" / "proposed_grl_advanced"

    print("=" * 70)
    print("TRAINING ADVANCED OPTIMIZED MODEL (Scheduled GRL + RawBoost + Full Convergence)")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["data"]["manifest_path"] = str(manifest_path)

    # Train
    train_res = run_training(cfg, output_dir=out_dir, seed=42)
    best_ckpt = out_dir / "best.pt"

    # Evaluate on Test Split (Unseen S3-S7)
    print("\nEvaluating Advanced Model on Test set (20,030 items)...")
    ev = evaluate_checkpoint(best_ckpt, manifest_path=manifest_path, split="test")

    with open(out_dir / "eval_test.json", "w") as f:
        clean_ev = {k: v for k, v in ev.items() if not isinstance(v, (np.ndarray, list))}
        clean_ev["per_emotion"] = ev.get("per_emotion", {})
        clean_ev["per_system"] = ev.get("per_system", {})
        clean_ev["emotion_stats"] = ev.get("emotion_stats", {})
        clean_ev["ci_95"] = ev.get("ci_95", (0, 0))
        json.dump(clean_ev, f, indent=2)

    # Probing
    print("\nRunning Emotion Probe on Advanced Model...")
    pr = run_emotion_probe(best_ckpt, manifest_path=manifest_path, train_split="train", test_split="test")
    with open(out_dir / "probe_results.json", "w") as f:
        json.dump(pr, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("ADVANCED MODEL TEST PERFORMANCE")
    print("=" * 50)
    print(f"Global EER: {ev['global_eer']:.2f}% (95% CI: [{ev['ci_95'][0]:.2f}%, {ev['ci_95'][1]:.2f}%])")
    print(f"Emotion Mean: {ev['emotion_stats']['mean']:.2f}%, Sigma: {ev['emotion_stats']['sigma']:.2f}%, Range: {ev['emotion_stats']['range']:.2f}%")
    print("Per-Emotion:")
    for emo, score in ev.get("per_emotion", {}).items():
        print(f"  {emo}: {score:.2f}%")
    print("Per-System (Unseen S3-S7):")
    for sys_id, score in sorted(ev.get("per_system", {}).items()):
        print(f"  {sys_id}: {score:.2f}%")
    print(f"Emotion Probe Accuracy: {pr['probe_accuracy']:.2f}% (Reference Chance: ~20%)")
    print("=" * 50)

    # Append / Update main_table.csv
    main_csv = PROJECT_ROOT / "results" / "main_table.csv"
    if main_csv.exists():
        df_main = pd.read_csv(main_csv)
    else:
        df_main = pd.DataFrame()

    new_row = {
        "Model": "Proposed Model (Advanced: Scheduled GRL + RawBoost)",
        "Global EER (%)": f"{ev['global_eer']:.2f}",
        "95% CI": f"[{ev['ci_95'][0]:.2f}, {ev['ci_95'][1]:.2f}]",
        "Emotion Mean EER (%)": f"{ev['emotion_stats']['mean']:.2f}",
        "Emotion Std Sigma (%)": f"{ev['emotion_stats']['sigma']:.2f}",
        "Emotion Range (%)": f"{ev['emotion_stats']['range']:.2f}",
        "Probe Accuracy (%)": f"{pr['probe_accuracy']:.2f}"
    }
    df_main = pd.concat([df_main, pd.DataFrame([new_row])], ignore_index=True)
    df_main.to_csv(main_csv, index=False)

    # Per-Emotion CSV
    emo_csv = PROJECT_ROOT / "results" / "per_emotion_table.csv"
    if emo_csv.exists():
        df_emo = pd.read_csv(emo_csv)
    else:
        df_emo = pd.DataFrame()
    new_emo_row = {"Model": "Proposed Model (Advanced: Scheduled GRL + RawBoost)"}
    new_emo_row.update({k.capitalize(): f"{v:.2f}%" for k, v in ev.get("per_emotion", {}).items()})
    df_emo = pd.concat([df_emo, pd.DataFrame([new_emo_row])], ignore_index=True)
    df_emo.to_csv(emo_csv, index=False)

    # Per-System CSV
    sys_csv = PROJECT_ROOT / "results" / "per_system_table.csv"
    if sys_csv.exists():
        df_sys = pd.read_csv(sys_csv)
    else:
        df_sys = pd.DataFrame()
    new_sys_row = {"Model": "Proposed Model (Advanced: Scheduled GRL + RawBoost)"}
    new_sys_row.update({s: f"{v:.2f}%" for s, v in sorted(ev.get("per_system", {}).items())})
    df_sys = pd.concat([df_sys, pd.DataFrame([new_sys_row])], ignore_index=True)
    df_sys.to_csv(sys_csv, index=False)

    # Regenerate per-emotion bar plot
    all_emo_dict = {}
    for idx, row in df_emo.iterrows():
        m_name = row["Model"]
        all_emo_dict[m_name] = {col.lower(): float(str(row[col]).replace("%", "")) for col in df_emo.columns if col != "Model" and pd.notna(row[col])}
    plot_per_emotion_eer_bar(all_emo_dict, PROJECT_ROOT / "analysis" / "per_emotion_eer_bar.png")

    # t-SNE plot for advanced model
    if "embeddings" in ev:
        plot_embeddings_tsne(
            embeddings=ev["embeddings"][:2000],
            spoof_labels=np.array(ev["spoof_labels"][:2000]),
            emotion_names=ev["emotion_names"][:2000],
            title_prefix="Proposed Model (Advanced GRL + RawBoost)",
            output_path=PROJECT_ROOT / "analysis" / "tsne_with_grl.png"
        )

    # Git sync
    print("\nPushing optimized results to GitHub...")
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Add Advanced Proposed Model (RawBoost + Full GRL Convergence) benchmark results"], check=False)
    subprocess.run(["git", "push", "origin", "main"], check=False)
    print("Advanced optimization complete and pushed to GitHub!")


if __name__ == "__main__":
    run_advanced_training()
