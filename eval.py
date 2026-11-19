"""
Evaluation script for trained anti-spoofing models.
Calculates global EER, 95% bootstrap CIs, per-emotion EERs (mean, range, sigma),
and open-set unseen system EERs (S3-S7).
"""

import sys
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.dataset import AudioSpoofEmotionDataset
from data.preprocessing import AudioPreprocessor
from models.factory import create_model
from models.metrics import evaluate_predictions


def evaluate_checkpoint(checkpoint_path: Path,
                        manifest_path: Path,
                        split: str = "test",
                        batch_size: int = 16,
                        device_str: str = "cuda") -> Dict[str, Any]:
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine config and weights
    if isinstance(ckpt, dict) and "config" in ckpt:
        config = ckpt["config"]
        state_dict = ckpt.get("model_state", ckpt)
        norm_stats = ckpt.get("norm_stats", None)
    else:
        # Pretrained baseline weights (e.g. AASIST.pth)
        config = {"architecture": "aasist", "model_config": {}}
        state_dict = ckpt
        norm_stats = None

    model_name = config.get("architecture", "proposed_cnn_bilstm")
    model = create_model(model_name, config, device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return_raw = True
    feature_type = config.get("data", {}).get("feature_type", "combined")

    prep = AudioPreprocessor(target_sr=16000, target_samples=64000)
    dataset = AudioSpoofEmotionDataset(
        manifest_path=manifest_path,
        split=split,
        preprocessor=prep,
        norm_stats=norm_stats,
        return_raw=return_raw,
        feature_type=feature_type
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Evaluating {len(dataset)} items on '{split}' split...")

    all_scores = []
    all_spoof_lbls = []
    all_emo_lbls = []
    all_systems = []
    all_paths = []
    all_embeddings = []

    with torch.no_grad():
        for batch in loader:
            if "features" in batch:
                inp = batch["features"].to(device)
            else:
                inp = batch["wav"].to(device)

            res = model(inp)
            if isinstance(res, tuple):
                if len(res) == 3:
                    z, logits_spoof, _ = res
                    all_embeddings.append(z.cpu().numpy())
                else:
                    z, logits_spoof = res
                    all_embeddings.append(z.cpu().numpy())
            else:
                logits_spoof = res

            # Score: P(bona_fide)
            probs = torch.softmax(logits_spoof, dim=1)
            scores = probs[:, 1].cpu().numpy().tolist()

            all_scores.extend(scores)
            all_spoof_lbls.extend(batch["spoof_label"].numpy().tolist())
            all_emo_lbls.extend(batch["emotion_label"].numpy().tolist())
            all_systems.extend(batch["system"])
            all_paths.extend(batch["path"])

    from data.dataset import INV_EMOTION_MAP
    emo_names = [INV_EMOTION_MAP.get(e, "unknown") for e in all_emo_lbls]

    results = evaluate_predictions(
        scores=np.array(all_scores),
        spoof_labels=np.array(all_spoof_lbls),
        emotion_labels=emo_names,
        systems=all_systems
    )

    if all_embeddings:
        results["embeddings"] = np.concatenate(all_embeddings, axis=0)
    results["paths"] = all_paths
    results["scores"] = all_scores
    results["spoof_labels"] = all_spoof_lbls
    results["emotion_names"] = emo_names
    results["systems"] = all_systems

    return results


def print_evaluation_summary(results: Dict[str, Any]):
    print("\n" + "=" * 65)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 65)
    print(f"Global EER: {results['global_eer']:.2f}%  (95% CI: [{results['ci_95'][0]:.2f}%, {results['ci_95'][1]:.2f}%])")
    print(f"Operating Threshold: {results['threshold']:.4f}")
    print(f"Samples evaluated: {results['n_bonafide']} bona fide, {results['n_spoof']} spoof")

    if "emotion_stats" in results:
        est = results["emotion_stats"]
        print(f"\nEmotion Metrics (5 emotions):")
        print(f"  Mean EER:  {est['mean']:.2f}%")
        print(f"  Range:     {est['range']:.2f}%")
        print(f"  Sigma:     {est['sigma']:.2f}%  (Lower is more emotion-invariant)")

    if results.get("per_emotion"):
        print("\nPer-Emotion Breakdown:")
        for emo, eer in sorted(results["per_emotion"].items()):
            print(f"  - {emo.capitalize():<10}: {eer:.2f}%")

    if results.get("per_system"):
        print("\nPer-System Breakdown (Unseen S3-S7):")
        for sys_id, eer in sorted(results["per_system"].items()):
            print(f"  - {sys_id:<10}: {eer:.2f}%")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate anti-spoofing model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt or .pth")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv", help="Path to manifest CSV")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate: 'test', 'dev', 'train'")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    args = parser.parse_args()

    res = evaluate_checkpoint(Path(args.checkpoint), Path(args.manifest), split=args.split)
    print_evaluation_summary(res)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Filter non-serializable arrays
        save_res = {
            "global_eer": res["global_eer"],
            "threshold": res["threshold"],
            "ci_95": res["ci_95"],
            "n_bonafide": res["n_bonafide"],
            "n_spoof": res["n_spoof"],
            "per_emotion": res.get("per_emotion", {}),
            "emotion_stats": res.get("emotion_stats", {}),
            "per_system": res.get("per_system", {})
        }
        with open(out_p, "w") as f:
            json.dump(save_res, f, indent=2)
        print(f"Saved evaluation metrics to {out_p}")


if __name__ == "__main__":
    main()
