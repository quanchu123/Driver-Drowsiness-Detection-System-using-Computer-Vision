"""Evaluation metrics for drowsiness classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics: Dict[str, Any] = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc_roc"] = None
    try:
        metrics["auc_pr"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["auc_pr"] = None
    return metrics


def threshold_rule_scores(
    X: np.ndarray,
    y: np.ndarray,
    ear_idx: int = 0,
    mar_idx: int = 1,
    ear_thr: float = 0.41,
    mar_thr: float = 0.9,
    mode: str = "physiology",
) -> Dict[str, Any]:
    """
    mode:
      - physiology: drowsy if EAR < thr (and optionally MAR > thr)
      - report_or: drowsy if EAR < thr OR MAR > thr
      - inverted: drowsy if EAR > thr (matches raw CSV orientation)
    """
    ear = X[:, ear_idx]
    mar = X[:, mar_idx]
    if mode == "physiology":
        pred = ((ear < ear_thr) | (mar > mar_thr)).astype(int)
        # use distance-as-score for AUC: lower EAR → higher drowsy score
        score = (ear_thr - ear) + 0.5 * np.maximum(mar - mar_thr, 0)
    elif mode == "report_or":
        pred = ((ear < ear_thr) | (mar > mar_thr)).astype(int)
        score = (ear_thr - ear) + 0.5 * np.maximum(mar - mar_thr, 0)
    elif mode == "inverted":
        pred = (ear > ear_thr).astype(int)
        score = ear - ear_thr
    else:
        raise ValueError(mode)

    # normalize score to [0,1]-ish for AUC
    s = score.astype(np.float64)
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    return compute_metrics(y, s, threshold=0.5) | {"rule_mode": mode, "hard_acc": float((pred == y).mean())}


def save_metrics(metrics: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # classification_report is str — keep it
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def print_metrics(name: str, metrics: Dict[str, Any]) -> None:
    print(f"\n=== {name} ===")
    for k in ("accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", "hard_acc"):
        if k in metrics and metrics[k] is not None:
            print(f"  {k:12s}: {metrics[k]:.4f}")
    if "confusion_matrix" in metrics:
        print(f"  confusion   : {metrics['confusion_matrix']}")
