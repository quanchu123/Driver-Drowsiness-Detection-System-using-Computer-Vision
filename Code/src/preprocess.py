"""Image preprocessing — includes low-light path (CLAHE / gamma)."""

from __future__ import annotations

import cv2
import numpy as np


def apply_clahe_bgr(frame: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Enhance local contrast in LAB L-channel (helps night / uneven lighting)."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def apply_gamma(frame: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    """Simple gamma correction for dark frames."""
    inv = 1.0 / max(gamma, 1e-6)
    table = (np.linspace(0, 1, 256) ** inv * 255).astype(np.uint8)
    return cv2.LUT(frame, table)


def estimate_brightness(frame: np.ndarray) -> float:
    """Mean luminance in [0, 255]."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def preprocess_frame(
    frame: np.ndarray,
    use_clahe: bool = True,
    auto_night: bool = True,
    night_brightness_threshold: float = 70.0,
    gamma_night: float = 1.5,
) -> np.ndarray:
    """
    Realtime preprocess path inspired by 2026 low-light DDS literature.
    Applies CLAHE always (if enabled); extra gamma when scene is dark.
    """
    out = frame
    if use_clahe:
        out = apply_clahe_bgr(out)
    if auto_night and estimate_brightness(out) < night_brightness_threshold:
        out = apply_gamma(out, gamma=gamma_night)
    return out
