"""
metrics.py — Biometric evaluation metrics shared across L3 experiments
"""

import numpy as np


def compute_FAR(scores: np.ndarray, labels: np.ndarray,
                threshold: float) -> float:
    """False Accept Rate: fraction of impostor attempts (label=0) that are accepted."""
    imp = scores[labels == 0]
    return float(np.mean(imp >= threshold)) if len(imp) > 0 else 0.0


def compute_FRR(scores: np.ndarray, labels: np.ndarray,
                threshold: float) -> float:
    """False Reject Rate: fraction of genuine attempts (label=1) that are rejected."""
    gen = scores[labels == 1]
    return float(np.mean(gen < threshold)) if len(gen) > 0 else 0.0


def compute_TAR(scores: np.ndarray, labels: np.ndarray,
                threshold: float) -> float:
    """True Accept Rate = 1 - FRR."""
    return 1.0 - compute_FRR(scores, labels, threshold)


def threshold_sweep(scores: np.ndarray, labels: np.ndarray,
                    steps: int = 1000) -> tuple:
    """Sweep thresholds over [0, 1] and compute FAR + FRR at each step."""
    thresholds = np.linspace(0.0, 1.0, steps)
    fars = np.array([compute_FAR(scores, labels, t) for t in thresholds])
    frrs = np.array([compute_FRR(scores, labels, t) for t in thresholds])
    return thresholds, fars, frrs


def compute_eer(thresholds: np.ndarray, fars: np.ndarray,
                frrs: np.ndarray) -> tuple:
    """
    Equal Error Rate and the decision threshold at which it occurs.
    Uses the index where |FAR - FRR| is minimised.
    """
    idx = int(np.argmin(np.abs(fars - frrs)))
    eer = float((fars[idx] + frrs[idx]) / 2.0)
    return eer, float(thresholds[idx])
