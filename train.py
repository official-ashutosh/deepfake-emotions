"""
Unified Training Script for Emotion-Invariant Audio Anti-Spoofing.
Supports Phase 1 (neutral baseline) and Phase 2 (dual-head adversarial training with GRL).
Includes AMP, gradient clipping, AdamW + cosine schedule, and balanced sampling.
"""

import os
import sys
import yaml
import json
import argparse
import random
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import AudioSpoofEmotionDataset, BalancedBatchSampler, compute_dataset_norm_stats
from data.preprocessing import AudioPreprocessor
from data.augmentation import AudioAugmenter
from models.factory import create_model
from models.grl import compute_dann_lambda
from models.metrics import evaluate_predictions, compute_eer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scaler: torch.cuda.amp.GradScaler,
                    device: torch.device,
                    epoch: int,
                    total_epochs: int,
                    lambda_max: float,
                    schedule_lambda: bool,
                    fixed_lambda: Optional[float],
                    grad_clip: float = 1.0,
                    phase: str = "phase2") -> Dict[str, float]:
    model.train()
    spoof_loss_fn = nn.CrossEntropyLoss()
    emotion_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    total_loss = 0.0
    total_ls = 0.0
    total_le = 0.0
    n_batches = len(loader)

    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        
        # Calculate dynamic or fixed lambda
        if schedule_lambda:
            p = (epoch * n_batches + i) / max(1, total_epochs * n_batches)
            curr_lambda = compute_dann_lambda(p, lambda_max=lambda_max)
        elif fixed_lambda is not None:
            curr_lambda = float(fixed_lambda)
        else:
            curr_lambda = float(lambda_max)

        if hasattr(model, "set_lambda"):
            model.set_lambda(curr_lambda)

        # Forward pass with AMP
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            if "features" in batch:
                inp = batch["features"].to(device)
            else:
                inp = batch["wav"].to(device)

            targets_spoof = batch["spoof_label"].to(device)
            targets_emo = batch["emotion_label"].to(device)

            # Check model output format
            res = model(inp)
            if isinstance(res, tuple):
                if len(res) == 3:
                    z, logits_spoof, logits_emo = res
                else:
                    _, logits_spoof = res
                    logits_emo = None
            else:
                logits_spoof = res
                logits_emo = None

            # Loss computation
            # NOTE: The GRL already multiplies emotion gradients by -lambda during
            # backprop. The forward loss is simply L_spoof + L_emotion.
            # The GRL handles the adversarial direction and weighting internally.
            loss_spoof = spoof_loss_fn(logits_spoof, targets_spoof)

            if phase == "phase2" and logits_emo is not None and (targets_emo != -1).any():
                loss_emo = emotion_loss_fn(logits_emo, targets_emo)
                loss = loss_spoof + loss_emo
                total_le += loss_emo.item()
            else:
                loss = loss_spoof
                loss_emo = torch.tensor(0.0)

        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_ls += loss_spoof.item()

        if (i + 1) % 200 == 0 or (i + 1) == n_batches:
            print(f"  [Epoch {epoch:02d}] Batch [{i+1:04d}/{n_batches:04d}] Loss: {loss.item():.4f} (Spoof: {loss_spoof.item():.4f}, Emo: {loss_emo.item():.4f}, Lambda: {curr_lambda:.3f})", flush=True)

    avg_loss = total_loss / max(1, n_batches)
    avg_ls = total_ls / max(1, n_batches)
    avg_le = total_le / max(1, n_batches)
    return {"loss": avg_loss, "loss_spoof": avg_ls, "loss_emotion": avg_le, "lambda": curr_lambda}


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    all_scores = []
    all_spoof_lbls = []
    all_emo_lbls = []
    all_systems = []

    with torch.no_grad():
        for batch in loader:
            if "features" in batch:
                inp = batch["features"].to(device)
            else:
                inp = batch["wav"].to(device)

            res = model(inp)
            if isinstance(res, tuple):
                logits_spoof = res[1]
            else:
                logits_spoof = res

            # Score is probability or log-probability of bona fide class (index 1)
            probs = torch.softmax(logits_spoof, dim=1)
            scores = probs[:, 1].cpu().numpy().tolist()

            all_scores.extend(scores)
            all_spoof_lbls.extend(batch["spoof_label"].numpy().tolist())
            all_emo_lbls.extend(batch["emotion_label"].numpy().tolist())
            all_systems.extend(batch["system"])

    # Map emotion integer IDs back to names for metric aggregation
    from data.dataset import INV_EMOTION_MAP
    emo_names = [INV_EMOTION_MAP.get(e, "unknown") for e in all_emo_lbls]

    results = evaluate_predictions(
        scores=np.array(all_scores),
        spoof_labels=np.array(all_spoof_lbls),
        emotion_labels=emo_names,
        systems=all_systems,
        compute_ci=False
    )
    return results


def run_training(config: Dict[str, Any], output_dir: Path, seed: int = 42) -> Dict[str, Any]:
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tb_logs"))

    # Save used config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device} | Seed: {seed}")

    # Build datasets (all models use fast on-GPU feature extraction from raw audio)
    data_cfg = config.get("data", {})
    manifest_path = data_cfg.get("manifest_path", "data/manifest.csv")
    feature_type = data_cfg.get("feature_type", "combined")
    model_name = config.get("architecture", "proposed_cnn_bilstm")
    return_raw = True

    prep = AudioPreprocessor(target_sr=16000, target_samples=64000)
    aug = AudioAugmenter(
        enable_noise=data_cfg.get("aug_noise", True),
        enable_gain=data_cfg.get("aug_gain", True),
        enable_codec=data_cfg.get("aug_codec", True),
        enable_rawboost=data_cfg.get("aug_rawboost", False)
    )

    train_ds = AudioSpoofEmotionDataset(
        manifest_path=manifest_path,
        split="train",
        preprocessor=prep,
        augmenter=aug,
        return_raw=return_raw,
        feature_type=feature_type
    )

    # Compute training normalization stats if feature-based
    norm_stats = None
    if not return_raw and data_cfg.get("normalize_features", True):
        norm_stats = compute_dataset_norm_stats(train_ds, max_samples=200)
        train_ds.norm_stats = norm_stats

    dev_ds = AudioSpoofEmotionDataset(
        manifest_path=manifest_path,
        split="dev",
        preprocessor=prep,
        norm_stats=norm_stats,
        return_raw=return_raw,
        feature_type=feature_type
    )

    # DataLoaders
    batch_size = config.get("train", {}).get("batch_size", 16)
    use_balanced = data_cfg.get("balanced_batch", True)

    if use_balanced and len(train_ds) >= batch_size:
        sampler = BalancedBatchSampler(train_ds, batch_size=batch_size)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize model
    model = create_model(model_name, config, device)

    # Load Phase 1 checkpoint if specified
    pretrained_phase1 = config.get("train", {}).get("pretrained_phase1_ckpt", None)
    if pretrained_phase1 and Path(pretrained_phase1).exists():
        print(f"Loading Phase 1 checkpoint from: {pretrained_phase1}")
        state = torch.load(pretrained_phase1, map_location=device, weights_only=False)
        model.load_state_dict(state.get("model_state", state), strict=False)

    # Optimizer & Scheduler
    t_cfg = config.get("train", {})
    lr = float(t_cfg.get("lr", 1e-3))
    weight_decay = float(t_cfg.get("weight_decay", 1e-4))
    epochs = int(t_cfg.get("epochs", 25))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # GRL parameters
    lambda_max = float(t_cfg.get("lambda_max", 1.0))
    schedule_lambda = bool(t_cfg.get("schedule_lambda", True))
    fixed_lambda = t_cfg.get("fixed_lambda", None)
    phase = t_cfg.get("phase", "phase2")
    patience = int(t_cfg.get("patience", 7))  # Early stopping patience
    warmup_epochs = int(t_cfg.get("warmup_epochs", 3))

    best_combined_score = float("inf")
    best_train_spoof_loss = float("inf")
    best_metrics = {}
    history = []
    no_improve_count = 0

    print(f"Starting training: {epochs} epochs | Phase: {phase} | Lambda: {lambda_max} (Schedule: {schedule_lambda})")
    print(f"Early stopping patience: {patience} epochs")

    for epoch in range(1, epochs + 1):
        train_res = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            lambda_max=lambda_max,
            schedule_lambda=schedule_lambda,
            fixed_lambda=fixed_lambda,
            phase=phase
        )
        scheduler.step()

        # Evaluate on dev
        dev_res = evaluate_model(model, dev_loader, device)
        dev_eer = dev_res["global_eer"]
        emo_stats = dev_res.get("emotion_stats", {"sigma": 0.0})
        dev_sigma = emo_stats.get("sigma", 0.0)

        # Early stopping criterion: dev spoof EER + std of per-emotion EER
        combined_score = dev_eer + dev_sigma

        # Logging
        writer.add_scalar("Train/Loss", train_res["loss"], epoch)
        writer.add_scalar("Train/Loss_Spoof", train_res["loss_spoof"], epoch)
        writer.add_scalar("Train/Loss_Emotion", train_res["loss_emotion"], epoch)
        writer.add_scalar("Train/Lambda", train_res["lambda"], epoch)
        writer.add_scalar("Dev/EER", dev_eer, epoch)
        writer.add_scalar("Dev/Sigma_Emotion", dev_sigma, epoch)
        writer.add_scalar("Dev/Combined_Score", combined_score, epoch)

        row = {
            "epoch": epoch,
            "train_loss": train_res["loss"],
            "train_loss_spoof": train_res["loss_spoof"],
            "train_loss_emotion": train_res["loss_emotion"],
            "train_lambda": train_res["lambda"],
            "dev_eer": dev_eer,
            "dev_sigma": dev_sigma,
            "combined_score": combined_score
        }
        history.append(row)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_res['loss']:.4f} | "
              f"Dev EER: {dev_eer:.2f}% | Dev Sigma: {dev_sigma:.2f}% | Combined: {combined_score:.2f}%", flush=True)

        # Save best model
        improved = False
        if combined_score < best_combined_score:
            improved = True
            best_combined_score = combined_score
            best_train_spoof_loss = train_res["loss_spoof"]
        elif abs(combined_score - best_combined_score) < 1e-5 and train_res["loss_spoof"] < best_train_spoof_loss:
            improved = True
            best_train_spoof_loss = train_res["loss_spoof"]

        if improved:
            best_metrics = dev_res
            no_improve_count = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config,
                "dev_metrics": dev_res,
                "norm_stats": norm_stats
            }, output_dir / "best.pt")
            print(f"  * New best model saved (Combined Score: {best_combined_score:.2f}%, Spoof Loss: {best_train_spoof_loss:.4f})")
        else:
            no_improve_count += 1
            if no_improve_count >= patience and epoch > warmup_epochs:
                print(f"  Early stopping triggered after {patience} epochs without improvement.")
                break

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "config": config
        }, output_dir / "last.pt")

    # Save history CSV
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    writer.close()
    print(f"Training completed. Best Combined Dev Score: {best_combined_score:.2f}%")
    return {"best_combined_score": best_combined_score, "best_metrics": best_metrics, "history": history}


def main():
    parser = argparse.ArgumentParser(description="Train emotion-invariant spoof detector")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--output", type=str, default=None, help="Output directory for checkpoints and logs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count (overrides config)")
    parser.add_argument("--lambda-max", type=float, default=None, help="Max lambda for GRL")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.lambda_max is not None:
        cfg.setdefault("train", {})["lambda_max"] = args.lambda_max

    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    output_dir = Path(args.output or cfg.get("output_dir", f"runs/{Path(args.config).stem}"))

    run_training(cfg, output_dir=output_dir, seed=seed)


if __name__ == "__main__":
    main()
