"""Lightweight preprocess — CLAHE only when dark (saves CPU)."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

# Reuse CLAHE instance (creating every frame is wasteful)
_CLAHE: Optional[cv2.CLAHE] = None
_GAMMA_TABLES: dict = {}


def _get_clahe(clip_limit: float = 2.0, tile: int = 8) -> cv2.CLAHE:
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return _CLAHE


def _gamma_table(gamma: float) -> np.ndarray:
    key = round(gamma, 2)
    if key not in _GAMMA_TABLES:
        inv = 1.0 / max(gamma, 1e-6)
        _GAMMA_TABLES[key] = (np.linspace(0, 1, 256) ** inv * 255).astype(np.uint8)
    return _GAMMA_TABLES[key]


def estimate_brightness_fast(frame: np.ndarray, step: int = 8) -> float:
    """Mean luminance on a sparse grid — ~O(H*W/step^2)."""
    # Sample BGR → approximate gray: 0.114B+0.587G+0.299R
    sample = frame[::step, ::step]
    b = sample[:, :, 0].astype(np.float32)
    g = sample[:, :, 1].astype(np.float32)
    r = sample[:, :, 2].astype(np.float32)
    return float((0.114 * b + 0.587 * g + 0.299 * r).mean())


def apply_clahe_bgr(frame: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = _get_clahe(clip_limit, tile).apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def apply_gamma(frame: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    return cv2.LUT(frame, _gamma_table(gamma))


def resize_for_detect(
    frame: np.ndarray, max_width: int = 480
) -> Tuple[np.ndarray, float]:
    """Downscale long side for landmark inference; returns (small, scale_to_original)."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    scale = max_width / float(w)
    small = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return small, 1.0 / scale


def preprocess_frame(
    frame: np.ndarray,
    use_clahe: bool = True,
    auto_night: bool = True,
    night_brightness_threshold: float = 70.0,
    gamma_night: float = 1.5,
    always_clahe: bool = False,
) -> np.ndarray:
    """
    Optimized path:
    - Measure brightness cheaply
    - CLAHE only if dark (or always_clahe=True)
    - Gamma only when very dark
    """
    bright = estimate_brightness_fast(frame)
    out = frame
    if use_clahe and (always_clahe or bright < night_brightness_threshold):
        out = apply_clahe_bgr(out)
        # re-check after CLAHE for gamma
        if auto_night and bright < night_brightness_threshold * 0.7:
            out = apply_gamma(out, gamma=gamma_night)
    elif auto_night and bright < night_brightness_threshold * 0.55:
        out = apply_gamma(out, gamma=gamma_night)
    return out
