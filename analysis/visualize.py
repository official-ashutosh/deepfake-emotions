"""
Visualization and analysis tools:
- t-SNE / UMAP plots of embeddings z colored by emotion vs spoof label (with and without GRL)
- Per-emotion EER bar charts
- Lambda vs (EER, sigma, probe accuracy) Pareto trade-off plot
- Confusion matrices
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from data.dataset import INV_EMOTION_MAP


# Set high-quality styling
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.autolayout": True, "font.family": "sans-serif"})


def plot_embeddings_tsne(embeddings: np.ndarray,
                         spoof_labels: np.ndarray,
                         emotion_names: List[str],
                         title_prefix: str,
                         output_path: Path):
    """
    Generate 2-panel figure:
    Panel A: Embeddings colored by Spoof Label (Bona fide vs Spoof)
    Panel B: Embeddings colored by Emotion (Neutral, Happy, Angry, Sad, Surprise)
    """
    print(f"Computing 2D t-SNE projection for {len(embeddings)} points...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1), random_state=42)
    proj = tsne.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Spoof Labels
    bona_mask = (spoof_labels == 1)
    axes[0].scatter(proj[bona_mask, 0], proj[bona_mask, 1], c="#2ca02c", label="Bona Fide", alpha=0.7, edgecolors="none", s=30)
    axes[0].scatter(proj[~bona_mask, 0], proj[~bona_mask, 1], c="#d62728", label="Spoof", alpha=0.7, edgecolors="none", s=30)
    axes[0].set_title(f"{title_prefix}: Colored by Spoof Label", fontsize=14, fontweight="bold")
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")

    # Panel B: Emotion Labels
    unique_emos = ["neutral", "happy", "angry", "sad", "surprise"]
    palette = {"neutral": "#7f7f7f", "happy": "#ff7f0e", "angry": "#d62728", "sad": "#1f77b4", "surprise": "#9467bd"}
    
    for emo in unique_emos:
        mask = np.array([e.lower() == emo for e in emotion_names])
        if mask.any():
            axes[1].scatter(proj[mask, 0], proj[mask, 1], c=palette.get(emo, "#333333"), label=emo.capitalize(), alpha=0.75, s=30)
    axes[1].set_title(f"{title_prefix}: Colored by Emotion", fontsize=14, fontweight="bold")
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved t-SNE plot to {output_path}")


def plot_per_emotion_eer_bar(results_dict: Dict[str, Dict[str, float]], output_path: Path):
    """
    Bar chart comparing per-emotion EER across multiple models (e.g. Baseline vs GRL).
    results_dict: {'Baseline (No GRL)': {'neutral': 9.2, ...}, 'Proposed (GRL)': {...}}
    """
    emotions = ["neutral", "happy", "angry", "sad", "surprise"]
    model_names = list(results_dict.keys())

    x = np.arange(len(emotions))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]

    for i, model in enumerate(model_names):
        eers = [results_dict[model].get(e, 0.0) for e in emotions]
        offset = (i - len(model_names) / 2.0 + 0.5) * width
        rects = ax.bar(x + offset, eers, width, label=model, color=colors[i % len(colors)], alpha=0.85)

    ax.set_ylabel("Equal Error Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Per-Emotion Audio Spoof Detection EER Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in emotions], fontsize=11)
    ax.legend(frameon=True, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved Per-Emotion EER bar chart to {output_path}")


def plot_lambda_pareto_curve(lambdas: List[float],
                             eers: List[float],
                             sigmas: List[float],
                             probe_accs: List[float],
                             output_path: Path):
    """
    Pareto plot: Lambda vs (Overall EER, Emotion Sigma, Probe Accuracy).
    Shows trade-off between spoof accuracy and emotion invariance.
    """
    fig, ax1 = plt.subplots(figsize=(9, 6))

    color_eer = "#1f77b4"
    color_sigma = "#d62728"
    color_probe = "#2ca02c"

    ax1.plot(lambdas, eers, marker="o", linewidth=2.5, color=color_eer, label="Spoof EER (%)")
    ax1.plot(lambdas, sigmas, marker="s", linewidth=2.5, linestyle="--", color=color_sigma, label=r"Emotion Std $\sigma$ (%)")
    ax1.set_xlabel(r"Gradient Reversal Strength ($\lambda_{max}$)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Error Rate / Std (%)", fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y")

    # Secondary axis for probe accuracy
    ax2 = ax1.twinx()
    ax2.plot(lambdas, probe_accs, marker="^", linewidth=2.5, linestyle="-.", color=color_probe, label="Emotion Probe Acc (%)")
    ax2.axhline(y=20.0, color="gray", linestyle=":", label="Chance Level (20%)")
    ax2.set_ylabel("Probe Accuracy (%)", fontsize=12, fontweight="bold", color=color_probe)
    ax2.tick_params(axis="y", labelcolor=color_probe)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

    plt.title(r"Impact of Adversarial Weight $\lambda_{max}$ on Accuracy and Invariance", fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved Lambda Pareto plot to {output_path}")


def plot_confusion_matrix_heatmap(cm: np.ndarray, class_names: List[str], title: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12) * 100.0
    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues", cbar=True,
                xticklabels=[c.capitalize() for c in class_names],
                yticklabels=[c.capitalize() for c in class_names], ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrix to {output_path}")
