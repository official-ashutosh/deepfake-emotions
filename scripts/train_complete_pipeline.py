"""
End-to-end training and evaluation pipeline on complete EmoFake dataset.
Runs extraction, manifest construction, training on RTX 4060, evaluation,
emotion probing, plot generation, and git synchronization.
"""

import os
import sys
from pathlib import Path

# Enable line buffering for real-time log output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import tarfile
import shutil
import subprocess
import pandas as pd
import numpy as np
import torch

from scripts.build_manifest import build_manifest_from_emofake_splits, build_manifest_from_directory, validate_manifest
from train import run_training
from eval import evaluate_checkpoint
from probe import run_emotion_probe
from analysis.visualize import (
    plot_embeddings_tsne,
    plot_per_emotion_eer_bar,
    plot_lambda_pareto_curve,
    plot_confusion_matrix_heatmap
)


def wait_and_extract_emofake(tar_path: Path, extract_dir: Path):
    """Wait for download to finish, then extract."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already extracted
    existing_wavs = list(extract_dir.rglob("*.wav"))
    if len(existing_wavs) > 100:
        print(f"EmoFake is already extracted: found {len(existing_wavs)} audio files in {extract_dir}")
        return extract_dir

    print(f"Waiting for download of {tar_path}...")
    while not tar_path.exists():
        time.sleep(10)
        # Check if part files exist
        part_files = list(tar_path.parent.glob("*.part"))
        if part_files:
            s_mb = part_files[0].stat().st_size / (1024 * 1024)
            print(f"  Downloading in progress... {s_mb:.1f} MB", end="\r")

    print(f"\nArchive found: {tar_path} ({tar_path.stat().st_size / (1024**3):.2f} GB)")
    print(f"Extracting to {extract_dir} (this may take a few minutes)...")
    
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=extract_dir)
        
    extracted_wavs = list(extract_dir.rglob("*.wav"))
    print(f"Extracted {len(extracted_wavs)} audio files to {extract_dir}")
    return extract_dir


def prepare_manifest(data_dir: Path, manifest_path: Path):
    """Build and validate the manifest."""
    print(f"Building manifest from {data_dir} -> {manifest_path}")
    
    # Check if English subfolder exists
    lang_dir = data_dir / "English"
    if not lang_dir.exists():
        for cand in data_dir.glob("**/English"):
            if cand.is_dir():
                lang_dir = cand
                break

    if lang_dir.exists() and (lang_dir / "train").exists():
        print(f"Using official EmoFake split directories under {lang_dir.parent}")
        df = build_manifest_from_emofake_splits(lang_dir.parent, manifest_path, language="English")
    else:
        print(f"Building manifest recursively from {data_dir}")
        df = build_manifest_from_directory(data_dir, manifest_path, dev_ratio=0.15)
        
    validate_manifest(manifest_path)
    return manifest_path


def run_pipeline():
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    print("=" * 70)
    print("EMOTION-INVARIANT DEEPFAKE AUDIO DETECTION - COMPLETE PIPELINE")
    print(f"Working directory: {project_root}")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Compute Device: {device_name}")
    print("=" * 70)

    # 1. Dataset Extraction
    tar_path = project_root / "data" / "emofake_download" / "emofake.tar.gz"
    extract_dir = project_root / "data" / "emofake"
    wait_and_extract_emofake(tar_path, extract_dir)

    # 2. Manifest Creation
    manifest_path = project_root / "data" / "manifest_emofake.csv"
    prepare_manifest(extract_dir, manifest_path)

    # 3. Models to train
    models_to_train = [
        {
            "name": "Ablation 1 (No Emotion Head)",
            "config": "configs/ablation1_no_emotion.yaml",
            "output": "runs/ablation1_no_emotion"
        },
        {
            "name": "Ablation 2 (Emotion Head, No GRL)",
            "config": "configs/ablation2_no_grl.yaml",
            "output": "runs/ablation2_no_grl"
        },
        {
            "name": "Proposed Model (Scheduled GRL, lambda_max=0.5)",
            "config": "configs/proposed_grl_scheduled.yaml",
            "output": "runs/proposed_grl_scheduled"
        },
        {
            "name": "Proposed Model (Fixed GRL, lambda=0.5)",
            "config": "configs/proposed_grl_fixed.yaml",
            "output": "runs/proposed_grl_fixed"
        }
    ]

    import yaml
    eval_results = {}
    probe_results = {}

    for m in models_to_train:
        cfg_path = project_root / m["config"]
        out_dir = project_root / m["output"]
        print(f"\n>>> Starting training: {m['name']} <<<")
        
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
            
        cfg["data"]["manifest_path"] = str(manifest_path)
        
        # Train
        train_res = run_training(cfg, output_dir=out_dir, seed=42)
        best_ckpt = out_dir / "best.pt"
        
        # Evaluate on Test Split (Unseen S3-S7)
        print(f"\nEvaluating {m['name']} on Test set...")
        ev = evaluate_checkpoint(best_ckpt, manifest_path=manifest_path, split="test")
        eval_results[m["name"]] = ev
        
        # Save eval json
        with open(out_dir / "eval_test.json", "w") as f:
            clean_ev = {k: v for k, v in ev.items() if not isinstance(v, (np.ndarray, list))}
            clean_ev["per_emotion"] = ev.get("per_emotion", {})
            clean_ev["per_system"] = ev.get("per_system", {})
            clean_ev["emotion_stats"] = ev.get("emotion_stats", {})
            clean_ev["ci_95"] = ev.get("ci_95", (0, 0))
            import json
            json.dump(clean_ev, f, indent=2)

        # Probing (for models with emotion classification)
        if "No Emotion" not in m["name"]:
            print(f"\nRunning Emotion Probe on {m['name']}...")
            try:
                pr = run_emotion_probe(best_ckpt, manifest_path=manifest_path, train_split="train", test_split="test")
                probe_results[m["name"]] = pr
                with open(out_dir / "probe_results.json", "w") as f:
                    import json
                    json.dump(pr, f, indent=2)
            except Exception as e:
                print(f"Probing error for {m['name']}: {e}")

    # 4. Generate Main and Comparison Tables
    print("\nGenerating final summary tables...")
    main_rows = []
    per_emo_data = {}
    
    for name, ev in eval_results.items():
        est = ev.get("emotion_stats", {})
        pr_acc = probe_results.get(name, {}).get("probe_accuracy", float("nan"))
        row = {
            "Model": name,
            "Global EER (%)": f"{ev['global_eer']:.2f}",
            "95% CI": f"[{ev['ci_95'][0]:.2f}, {ev['ci_95'][1]:.2f}]",
            "Emotion Mean EER (%)": f"{est.get('mean', 0.0):.2f}",
            "Emotion Std Sigma (%)": f"{est.get('sigma', 0.0):.2f}",
            "Emotion Range (%)": f"{est.get('range', 0.0):.2f}",
            "Probe Accuracy (%)": f"{pr_acc:.2f}" if not np.isnan(pr_acc) else "N/A"
        }
        main_rows.append(row)
        per_emo_data[name] = ev.get("per_emotion", {})

    main_df = pd.DataFrame(main_rows)
    main_df.to_csv("results/main_table.csv", index=False)
    print("\nMain Table:")
    print(main_df.to_string(index=False))

    # Per-Emotion Table
    emo_rows = []
    for name, emos in per_emo_data.items():
        r = {"Model": name}
        r.update({k.capitalize(): f"{v:.2f}%" for k, v in emos.items()})
        emo_rows.append(r)
    pd.DataFrame(emo_rows).to_csv("results/per_emotion_table.csv", index=False)

    # Per-System Table
    sys_rows = []
    for name, ev in eval_results.items():
        r = {"Model": name}
        for s, eer in sorted(ev.get("per_system", {}).items()):
            r[s] = f"{eer:.2f}%"
        sys_rows.append(r)
    pd.DataFrame(sys_rows).to_csv("results/per_system_table.csv", index=False)

    # 5. Visualizations
    print("\nGenerating analysis plots...")
    # Bar chart
    plot_per_emotion_eer_bar(per_emo_data, Path("analysis/per_emotion_eer_bar.png"))

    # t-SNE for Proposed Scheduled Model
    prop_name = "Proposed Model (Scheduled GRL, lambda_max=0.5)"
    if prop_name in eval_results and "embeddings" in eval_results[prop_name]:
        ev = eval_results[prop_name]
        plot_embeddings_tsne(
            embeddings=ev["embeddings"][:2000],
            spoof_labels=np.array(ev["spoof_labels"][:2000]),
            emotion_names=ev["emotion_names"][:2000],
            title_prefix="Proposed Model (Scheduled GRL)",
            output_path=Path("analysis/tsne_with_grl.png")
        )

    # t-SNE for Baseline (No GRL)
    base_name = "Ablation 2 (Emotion Head, No GRL)"
    if base_name in eval_results and "embeddings" in eval_results[base_name]:
        ev = eval_results[base_name]
        plot_embeddings_tsne(
            embeddings=ev["embeddings"][:2000],
            spoof_labels=np.array(ev["spoof_labels"][:2000]),
            emotion_names=ev["emotion_names"][:2000],
            title_prefix="Ablation 2 (Without GRL)",
            output_path=Path("analysis/tsne_without_grl.png")
        )

    # 6. Git Commit & Push
    print("\nSynchronizing results to GitHub repository...")
    try:
        subprocess.run(["git", "add", "."], check=True)
        commit_msg = "Complete training run on real EmoFake with RTX 4060, results, and analysis"
        subprocess.run(["git", "commit", "-m", commit_msg], check=False)
        push_res = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True)
        print(f"Git push output: {push_res.stdout}\n{push_res.stderr}")
        if push_res.returncode == 0:
            print("Successfully pushed to https://github.com/official-ashutosh/deepfake-emotions")
        else:
            print(f"Push exited with code {push_res.returncode}")
    except Exception as e:
        print(f"Git synchronization error: {e}")

    print("\nPipeline execution complete!")


if __name__ == "__main__":
    run_pipeline()
