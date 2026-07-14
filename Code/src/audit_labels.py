"""Label audit: check EAR/MAR correlation with Drowsy; recommend flip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-12:
        return 0.0
    return float((a * b).sum() / denom)


def audit_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    ear = df["EAR"].to_numpy(dtype=np.float64)
    mar = df["MAR"].to_numpy(dtype=np.float64)
    y = df["Drowsy"].astype(int).to_numpy()

    report: Dict[str, Any] = {
        "n_samples": int(len(df)),
        "class_counts": {
            "0": int((y == 0).sum()),
            "1": int((y == 1).sum()),
        },
        "ear_mean_by_class": {
            "0": float(ear[y == 0].mean()) if (y == 0).any() else None,
            "1": float(ear[y == 1].mean()) if (y == 1).any() else None,
        },
        "mar_mean_by_class": {
            "0": float(mar[y == 0].mean()) if (y == 0).any() else None,
            "1": float(mar[y == 1].mean()) if (y == 1).any() else None,
        },
        "corr_ear_drowsy": pearson(ear, y.astype(np.float64)),
        "corr_mar_drowsy": pearson(mar, y.astype(np.float64)),
    }

    # Physiology: drowsy should have LOWER EAR → corr should be negative
    corr = report["corr_ear_drowsy"]
    ear0 = report["ear_mean_by_class"]["0"]
    ear1 = report["ear_mean_by_class"]["1"]
    inverted = corr is not None and corr > 0.3 and ear1 is not None and ear0 is not None and ear1 > ear0

    # Threshold scan: EAR < t (physiology) vs EAR > t (inverted labels)
    best_phys = {"acc": -1.0, "t": None}
    best_inv = {"acc": -1.0, "t": None}
    for t in np.linspace(0.30, 0.55, 26):
        pred_phys = (ear < t).astype(int)
        pred_inv = (ear > t).astype(int)
        acc_p = float((pred_phys == y).mean())
        acc_i = float((pred_inv == y).mean())
        if acc_p > best_phys["acc"]:
            best_phys = {"acc": acc_p, "t": float(t)}
        if acc_i > best_inv["acc"]:
            best_inv = {"acc": acc_i, "t": float(t)}

    report["threshold_scan"] = {
        "ear_lt_best": best_phys,
        "ear_gt_best": best_inv,
    }
    report["likely_inverted_labels"] = bool(inverted)
    report["recommendation"] = (
        "FLIP labels (1 - Drowsy) so low EAR aligns with drowsy"
        if inverted
        else "Keep labels as-is; EAR-drowsiness relationship looks physiological"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit EAR/MAR vs Drowsy labels")
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to CSV (default: ../../Data/drowsiness_data_shuffled.csv relative to Code)",
    )
    parser.add_argument("--out", default="artifacts/label_audit.json")
    args = parser.parse_args()

    code_dir = Path(__file__).resolve().parents[1]
    csv_path = Path(args.csv) if args.csv else code_dir.parent / "Data" / "drowsiness_data_shuffled.csv"
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = code_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    report = audit_dataframe(df)
    report["csv_path"] = str(csv_path)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
