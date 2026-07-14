"""Sequence models (Phase 2) — BiLSTM stub for temporal feature windows.

Install torch to use. Geometry + XGBoost path does not require this module.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch required for sequence models: pip install torch")


class BiLSTMClassifier(nn.Module if nn is not None else object):
    """Lightweight BiLSTM over [T, F] feature sequences."""

    def __init__(self, input_dim: int, hidden: int = 64, num_layers: int = 1, dropout: float = 0.2):
        _require_torch()
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def build_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    window: int = 30,
    stride: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding windows over a single long recording-style array.
    For shuffled independent frames this is only a placeholder —
    prefer video-ordered data for real temporal training.
    """
    X, y = [], []
    n = len(features)
    for i in range(0, max(0, n - window + 1), stride):
        X.append(features[i : i + window])
        # label = majority in window
        y.append(int(np.round(labels[i : i + window].mean())))
    if not X:
        return np.zeros((0, window, features.shape[1])), np.zeros((0,), dtype=np.int64)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)
