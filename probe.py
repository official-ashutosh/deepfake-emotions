"""
Emotion Invariance Probe Script.
Freezes the encoder, extracts embeddings z, and trains a linear probe to predict the 5 emotion classes.
Validates whether adversarial GRL training reduces emotion recoverability towards ~20% chance level.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

from data.dataset import AudioSpoofEmotionDataset, INV_EMOTION_MAP
from data.preprocessing import AudioPreprocessor
from models.factory import create_model


def extract_all_embeddings_and_emotions(model: torch.nn.Module,
                                       dataset: AudioSpoofEmotionDataset,
                                       device: torch.device,
                                       batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    embeddings = []
    emotion_labels = []

    with torch.no_grad():
        for batch in loader:
            if "features" in batch:
                inp = batch["features"].to(device)
            else:
                inp = batch["wav"].to(device)

            # Extract z
            if hasattr(model, "extract_embedding"):
                z = model.extract_embedding(inp)
            else:
                res = model(inp)
                z = res[0] if isinstance(res, tuple) else res

            emo_lbl = batch["emotion_label"].numpy()
            # Only keep samples with valid emotion labels (0..4)
            valid_mask = (emo_lbl >= 0)
            if valid_mask.any():
                z_np = z.cpu().numpy()[valid_mask]
                embeddings.append(z_np)
                emotion_labels.append(emo_lbl[valid_mask])

    if not embeddings:
        return np.empty((0, 128)), np.empty((0,))
    X = np.concatenate(embeddings, axis=0)
    y = np.concatenate(emotion_labels, axis=0)
    return X, y


def run_emotion_probe(checkpoint_path: Path,
                      manifest_path: Path,
                      train_split: str = "train",
                      test_split: str = "test",
                      max_iter: int = 1000) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint for emotion probing: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "config" in ckpt:
        config = ckpt["config"]
        state_dict = ckpt.get("model_state", ckpt)
        norm_stats = ckpt.get("norm_stats", None)
    else:
        config = {"architecture": "proposed_cnn_bilstm"}
        state_dict = ckpt
        norm_stats = None

    model_name = config.get("architecture", "proposed_cnn_bilstm")
    model = create_model(model_name, config, device)
    model.load_state_dict(state_dict, strict=False)
    
    # Freeze entire model
    for p in model.parameters():
        p.requires_grad = False

    return_raw = True
    feature_type = config.get("data", {}).get("feature_type", "combined")
    prep = AudioPreprocessor(target_sr=16000, target_samples=64000)

    train_ds = AudioSpoofEmotionDataset(
        manifest_path=manifest_path,
        split=train_split,
        preprocessor=prep,
        norm_stats=norm_stats,
        return_raw=return_raw,
        feature_type=feature_type
    )

    test_ds = AudioSpoofEmotionDataset(
        manifest_path=manifest_path,
        split=test_split,
        preprocessor=prep,
        norm_stats=norm_stats,
        return_raw=return_raw,
        feature_type=feature_type
    )

    print(f"Extracting frozen embeddings from '{train_split}' split...")
    X_train, y_train = extract_all_embeddings_and_emotions(model, train_ds, device)
    print(f"Extracting frozen embeddings from '{test_split}' split...")
    X_test, y_test = extract_all_embeddings_and_emotions(model, test_ds, device)

    print(f"Train embeddings: {X_train.shape} | Test embeddings: {X_test.shape}")
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Insufficient emotion-labeled data found in train/test splits for probing.")

    # Train linear probe (Logistic Regression with L2 regularization)
    probe = LogisticRegression(max_iter=max_iter, C=1.0, random_state=42)
    probe.fit(X_train, y_train)

    preds = probe.predict(X_test)
    acc = float(accuracy_score(y_test, preds)) * 100.0
    bal_acc = float(balanced_accuracy_score(y_test, preds)) * 100.0
    conf_mat = confusion_matrix(y_test, preds).tolist()

    classes_present = sorted(list(set(y_test)))
    class_names = [INV_EMOTION_MAP.get(c, str(c)) for c in classes_present]
    report = classification_report(y_test, preds, target_names=class_names, output_dict=True, zero_division=0)

    results = {
        "probe_accuracy": acc,
        "balanced_accuracy": bal_acc,
        "chance_level": 20.0,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "confusion_matrix": conf_mat,
        "classification_report": report
    }

    print("\n" + "=" * 60)
    print("EMOTION PROBE RESULTS")
    print("=" * 60)
    print(f"Test Accuracy:          {acc:.2f}%  (Reference chance: ~20.0%)")
    print(f"Balanced Accuracy:      {bal_acc:.2f}%")
    print("\nClass breakdown:")
    for name in class_names:
        metrics = report.get(name, {})
        print(f"  - {name.capitalize():<10}: Precision={metrics.get('precision', 0):.2f}, Recall={metrics.get('recall', 0):.2f}, F1={metrics.get('f1-score', 0):.2f}")
    print("=" * 60 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run emotion invariance probe")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv", help="Path to manifest CSV")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--output", type=str, default=None, help="Save probe metrics JSON")
    args = parser.parse_args()

    res = run_emotion_probe(
        checkpoint_path=Path(args.checkpoint),
        manifest_path=Path(args.manifest),
        train_split=args.train_split,
        test_split=args.test_split
    )

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved probe results to {out_p}")


if __name__ == "__main__":
    main()
