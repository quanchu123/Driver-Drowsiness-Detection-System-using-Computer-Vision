"""Dataset loading, split, and optional label flip."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


FEATURE_CANDIDATES = ["EAR", "MAR", "EAR_L", "EAR_R", "pitch", "yaw", "roll"]


def load_csv(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"EAR", "MAR", "Drowsy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


def apply_label_policy(df: pd.DataFrame, flip_labels: bool) -> pd.DataFrame:
    out = df.copy()
    out["Drowsy"] = out["Drowsy"].astype(int)
    if flip_labels:
        # Invert so that lower EAR → drowsy aligns with physiology
        out["Drowsy"] = 1 - out["Drowsy"]
        out.attrs["labels_flipped"] = True
    else:
        out.attrs["labels_flipped"] = False
    return out


def select_feature_columns(df: pd.DataFrame, preferred: Optional[List[str]] = None) -> List[str]:
    preferred = preferred or FEATURE_CANDIDATES
    cols = [c for c in preferred if c in df.columns]
    if "EAR" not in cols or "MAR" not in cols:
        raise ValueError("Need at least EAR and MAR feature columns")
    return cols


def stratified_splits(
    df: pd.DataFrame,
    feature_cols: List[str],
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return dict of train/val/test (X, y) with stratified splits."""
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["Drowsy"].to_numpy(dtype=np.int64)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # val_size relative to full set → convert to fraction of trainval
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=relative_val,
        random_state=seed,
        stratify=y_trainval,
    )
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


def load_and_split(
    csv_path: str | Path,
    flip_labels: bool = True,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], List[str], pd.DataFrame]:
    df = load_csv(csv_path)
    df = apply_label_policy(df, flip_labels=flip_labels)
    cols = select_feature_columns(df, feature_cols)
    splits = stratified_splits(df, cols, test_size=test_size, val_size=val_size, seed=seed)
    return splits, cols, df
