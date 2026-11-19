"""
Evaluation metrics for audio anti-spoofing and emotion invariance.
Adapted from clovaai/aasist and ASVspoof official evaluation protocols.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_det_curve(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute DET curve (FRR and FAR) for given target and non-target scores.
    target_scores: scores of bona fide trials (higher should mean more bona fide)
    nontarget_scores: scores of spoof trials
    """
    target_scores = np.asarray(target_scores, dtype=np.float64)
    nontarget_scores = np.asarray(nontarget_scores, dtype=np.float64)

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0.0), tar_trial_sums / max(target_scores.size, 1)))
    far = np.concatenate((np.atleast_1d(1.0), nontarget_trial_sums / max(nontarget_scores.size, 1)))
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(target_scores: Union[np.ndarray, List[float]], 
                nontarget_scores: Union[np.ndarray, List[float]]) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and operating threshold.
    Returns:
        eer: Equal Error Rate in percentage (0 to 100) or float
        threshold: score threshold at EER
    """
    target_scores = np.asarray(target_scores, dtype=np.float64).ravel()
    nontarget_scores = np.asarray(nontarget_scores, dtype=np.float64).ravel()

    if len(target_scores) == 0 or len(nontarget_scores) == 0:
        return 0.0, 0.0

    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = float(np.mean((frr[min_index], far[min_index]))) * 100.0  # percentage
    thresh = float(thresholds[min_index])
    return eer, thresh


def bootstrap_eer_ci(target_scores: np.ndarray, nontarget_scores: np.ndarray, 
                     n_bootstraps: int = 1000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for EER.
    Returns (lower_bound, upper_bound) in percent.
    """
    rng = np.random.default_rng(seed)
    target_scores = np.asarray(target_scores)
    nontarget_scores = np.asarray(nontarget_scores)
    
    if len(target_scores) == 0 or len(nontarget_scores) == 0:
        return 0.0, 0.0

    eers = []
    n_tar = len(target_scores)
    n_non = len(nontarget_scores)

    for _ in range(n_bootstraps):
        tar_sample = rng.choice(target_scores, size=n_tar, replace=True)
        non_sample = rng.choice(nontarget_scores, size=n_non, replace=True)
        eer, _ = compute_eer(tar_sample, non_sample)
        eers.append(eer)

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(eers, 100.0 * alpha))
    high = float(np.percentile(eers, 100.0 * (1.0 - alpha)))
    return low, high


def compute_emotion_metrics(per_emotion_eers: Dict[str, float]) -> Dict[str, float]:
    """
    Compute mean, range, and standard deviation sigma across 5 emotions.
    sigma_emotion = sqrt(mean((E_e - E_bar)^2))
    """
    vals = list(per_emotion_eers.values())
    if not vals:
        return {"mean": 0.0, "range": 0.0, "sigma": 0.0}
    mean_val = float(np.mean(vals))
    range_val = float(np.max(vals) - np.min(vals))
    sigma_val = float(np.sqrt(np.mean([(v - mean_val) ** 2 for v in vals])))
    return {
        "mean": mean_val,
        "range": range_val,
        "sigma": sigma_val
    }


def evaluate_predictions(scores: np.ndarray,
                         spoof_labels: np.ndarray,
                         emotion_labels: Optional[List[str]] = None,
                         systems: Optional[List[str]] = None,
                         compute_ci: bool = True) -> Dict[str, Union[float, Dict]]:
    """
    Full comprehensive evaluation:
    - Global EER + 95% CI
    - Per-emotion EERs (for each present emotion: neutral, happy, angry, sad, surprise)
    - Emotion summary metrics (mean, range, sigma)
    - Per-system EERs (e.g. S1..S7)
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    spoof_labels = np.asarray(spoof_labels).ravel()

    # In our convention: 1 = bona_fide, 0 = spoof
    tar_mask = (spoof_labels == 1) | (spoof_labels == "bona_fide") | (spoof_labels == "bonafide") | (spoof_labels == "real")
    non_mask = (spoof_labels == 0) | (spoof_labels == "spoof") | (spoof_labels == "fake")

    tar_scores = scores[tar_mask]
    non_scores = scores[non_mask]

    global_eer, thresh = compute_eer(tar_scores, non_scores)
    if compute_ci:
        ci_low, ci_high = bootstrap_eer_ci(tar_scores, non_scores, n_bootstraps=500)
    else:
        ci_low, ci_high = 0.0, 0.0

    results = {
        "global_eer": global_eer,
        "threshold": thresh,
        "ci_95": (ci_low, ci_high),
        "n_bonafide": int(len(tar_scores)),
        "n_spoof": int(len(non_scores)),
        "per_emotion": {},
        "per_system": {}
    }

    # Per-emotion EER
    target_emotions = ["neutral", "happy", "angry", "sad", "surprise"]
    if emotion_labels is not None and len(emotion_labels) == len(scores):
        emo_arr = np.asarray([str(e).lower().strip() for e in emotion_labels])
        per_emo_dict = {}
        for emo in target_emotions:
            emo_match = (emo_arr == emo)
            e_tar = scores[tar_mask & emo_match]
            e_non = scores[non_mask & emo_match]
            if len(e_tar) > 0 and len(e_non) > 0:
                e_eer, _ = compute_eer(e_tar, e_non)
                per_emo_dict[emo] = e_eer
        results["per_emotion"] = per_emo_dict
        if len(per_emo_dict) > 0:
            results["emotion_stats"] = compute_emotion_metrics(per_emo_dict)

    # Per-system EER
    if systems is not None and len(systems) == len(scores):
        sys_arr = np.asarray([str(s).strip() for s in systems])
        unique_sys = [s for s in sorted(list(set(sys_arr))) if s and s.lower() not in ["none", "nan", "bonafide", "real", "-"]]
        per_sys_dict = {}
        for s in unique_sys:
            s_mask = (sys_arr == s)
            s_non = scores[non_mask & s_mask]
            # bona fide baseline for this system evaluation is the shared bona fide pool
            if len(tar_scores) > 0 and len(s_non) > 0:
                s_eer, _ = compute_eer(tar_scores, s_non)
                per_sys_dict[s] = s_eer
        results["per_system"] = per_sys_dict

    return results
