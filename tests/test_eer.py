"""
Unit tests for EER, t-DCF, and emotion variance metrics.
"""

import numpy as np
import pytest
from models.metrics import compute_eer, bootstrap_eer_ci, compute_emotion_metrics, evaluate_predictions


def test_eer_perfect_separation():
    # Targets are high, nontargets are low
    tar = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    non = np.array([-5.0, -4.0, -3.0, -2.0, -1.0])
    eer, thresh = compute_eer(tar, non)
    assert eer == 0.0, f"Expected 0% EER for perfect separation, got {eer}"


def test_eer_inverted_separation():
    # Targets are lower than nontargets
    tar = np.array([-5.0, -4.0, -3.0, -2.0, -1.0])
    non = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    eer, thresh = compute_eer(tar, non)
    assert eer == 100.0, f"Expected 100% EER for completely inverted scores, got {eer}"


def test_eer_random_near_fifty():
    rng = np.random.default_rng(123)
    tar = rng.normal(0, 1, 5000)
    non = rng.normal(0, 1, 5000)
    eer, thresh = compute_eer(tar, non)
    assert 47.0 <= eer <= 53.0, f"Expected ~50% EER for identical distributions, got {eer}"


def test_bootstrap_ci():
    rng = np.random.default_rng(42)
    tar = rng.normal(1.0, 1.0, 500)
    non = rng.normal(-1.0, 1.0, 500)
    eer, _ = compute_eer(tar, non)
    low, high = bootstrap_eer_ci(tar, non, n_bootstraps=200, ci=0.95)
    assert low <= eer <= high, f"EER ({eer}) must lie within 95% CI [{low}, {high}]"


def test_emotion_metrics_formula():
    eers = {"neutral": 2.0, "happy": 4.0, "angry": 6.0, "sad": 8.0, "surprise": 10.0}
    stats = compute_emotion_metrics(eers)
    assert abs(stats["mean"] - 6.0) < 1e-6, f"Mean expected 6.0, got {stats['mean']}"
    assert abs(stats["range"] - 8.0) < 1e-6, f"Range expected 8.0, got {stats['range']}"
    expected_sigma = np.sqrt(np.mean([(x - 6.0)**2 for x in [2, 4, 6, 8, 10]]))
    assert abs(stats["sigma"] - expected_sigma) < 1e-6, f"Sigma expected {expected_sigma}, got {stats['sigma']}"


def test_evaluate_predictions_aggregation():
    scores = np.array([2.0, 1.5, -1.0, -2.0, 3.0, -0.5])
    spoof_labels = np.array([1, 1, 0, 0, 1, 0])
    emotions = ["neutral", "happy", "neutral", "happy", "angry", "angry"]
    systems = ["-", "-", "S1", "S1", "-", "S2"]
    
    res = evaluate_predictions(scores, spoof_labels, emotions, systems)
    assert "global_eer" in res
    assert "per_emotion" in res
    assert "per_system" in res
    assert "emotion_stats" in res
    assert "S1" in res["per_system"]
    assert "S2" in res["per_system"]


if __name__ == "__main__":
    test_eer_perfect_separation()
    test_eer_inverted_separation()
    test_eer_random_near_fifty()
    test_bootstrap_ci()
    test_emotion_metrics_formula()
    test_evaluate_predictions_aggregation()
    print("All EER & metric unit tests passed successfully!")
