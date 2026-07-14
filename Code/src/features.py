"""Geometric & temporal drowsiness features (EAR, MAR, PERCLOS, blink, pose)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

# MediaPipe FaceMesh landmark indices (refined face mesh)
# Eyes (6-point EAR style)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Mouth for MAR
MOUTH = [61, 81, 13, 311, 291, 178, 14, 402]
# Head pose reference points
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
LEFT_MOUTH = 61
RIGHT_MOUTH = 291


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(landmarks: np.ndarray, eye_idx: Sequence[int]) -> float:
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    landmarks: (N, 2) or (N, 3) pixel coords
    """
    pts = landmarks[list(eye_idx)][:, :2]
    p1, p2, p3, p4, p5, p6 = pts
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * _dist(p1, p4) + 1e-8)


def mouth_aspect_ratio(landmarks: np.ndarray) -> float:
    """MAR using outer/inner mouth vertical vs horizontal span."""
    # Use MediaPipe mouth corners + top/bottom lip
    left = landmarks[61][:2]
    right = landmarks[291][:2]
    top = landmarks[13][:2]
    bottom = landmarks[14][:2]
    top2 = landmarks[81][:2]
    bottom2 = landmarks[178][:2]
    vert = _dist(top, bottom) + _dist(top2, bottom2)
    horiz = _dist(left, right)
    return vert / (2.0 * horiz + 1e-8)


def head_pose_approx(landmarks: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[float, float, float]:
    """
    Approximate head pose (pitch, yaw, roll) in degrees via solvePnP.
    Returns (0,0,0) if solve fails.
    """
    h, w = image_shape[:2]
    image_points = np.array(
        [
            landmarks[NOSE_TIP][:2],
            landmarks[CHIN][:2],
            landmarks[LEFT_EYE_CORNER][:2],
            landmarks[RIGHT_EYE_CORNER][:2],
            landmarks[LEFT_MOUTH][:2],
            landmarks[RIGHT_MOUTH][:2],
        ],
        dtype=np.float64,
    )
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ],
        dtype=np.float64,
    )
    focal = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array(
        [[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]], dtype=np.float64
    )
    dist_coeffs = np.zeros((4, 1))
    ok, rvec, _ = cv2_solve(image_points, model_points, camera_matrix, dist_coeffs)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = _rodrigues(rvec)
    # Pitch / yaw / roll from rotation matrix (approximate)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
    yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    roll = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
    return float(pitch), float(yaw), float(roll)


def cv2_solve(image_points, model_points, camera_matrix, dist_coeffs):
    import cv2

    return cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )


def _rodrigues(rvec):
    import cv2

    return cv2.Rodrigues(rvec)


def frame_features_from_landmarks(
    landmarks_xy: np.ndarray, image_shape: Tuple[int, int]
) -> Dict[str, float]:
    """Compute per-frame geometric features from (N,2/3) landmark array."""
    ear_l = eye_aspect_ratio(landmarks_xy, LEFT_EYE)
    ear_r = eye_aspect_ratio(landmarks_xy, RIGHT_EYE)
    ear = (ear_l + ear_r) / 2.0
    mar = mouth_aspect_ratio(landmarks_xy)
    pitch, yaw, roll = head_pose_approx(landmarks_xy, image_shape)
    return {
        "EAR": ear,
        "EAR_L": ear_l,
        "EAR_R": ear_r,
        "MAR": mar,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
    }


@dataclass
class TemporalFeatureBuffer:
    """Sliding-window temporal features: PERCLOS, blink, microsleep, yawn."""

    ear_closed_threshold: float = 0.25
    mar_yawn_threshold: float = 0.6
    window: int = 90
    blink_min_frames: int = 2
    blink_max_frames: int = 8
    microsleep_min_frames: int = 15
    yawn_min_frames: int = 10
    _ears: Deque[float] = field(default_factory=deque)
    _mars: Deque[float] = field(default_factory=deque)
    _closed_run: int = 0
    _yawn_run: int = 0
    _blink_count: int = 0
    _microsleep_count: int = 0
    _yawn_count: int = 0
    _prev_ear: Optional[float] = None

    def __post_init__(self) -> None:
        self._ears = deque(maxlen=self.window)
        self._mars = deque(maxlen=self.window)

    def reset(self) -> None:
        self._ears.clear()
        self._mars.clear()
        self._closed_run = 0
        self._yawn_run = 0
        self._blink_count = 0
        self._microsleep_count = 0
        self._yawn_count = 0
        self._prev_ear = None

    def update(self, ear: float, mar: float) -> Dict[str, float]:
        closed = ear < self.ear_closed_threshold
        yawning = mar > self.mar_yawn_threshold

        if closed:
            self._closed_run += 1
        else:
            if self.blink_min_frames <= self._closed_run <= self.blink_max_frames:
                self._blink_count += 1
            if self._closed_run >= self.microsleep_min_frames:
                self._microsleep_count += 1
            self._closed_run = 0

        if yawning:
            self._yawn_run += 1
            if self._yawn_run == self.yawn_min_frames:
                self._yawn_count += 1
        else:
            self._yawn_run = 0

        self._ears.append(ear)
        self._mars.append(mar)

        ear_arr = np.array(self._ears, dtype=np.float64)
        mar_arr = np.array(self._mars, dtype=np.float64)
        n = len(ear_arr)
        perclos = float(np.mean(ear_arr < self.ear_closed_threshold)) if n else 0.0
        ear_vel = 0.0 if self._prev_ear is None else float(ear - self._prev_ear)
        self._prev_ear = ear

        return {
            "PERCLOS": perclos,
            "EAR_mean_w": float(np.mean(ear_arr)),
            "EAR_std_w": float(np.std(ear_arr)) if n > 1 else 0.0,
            "EAR_min_w": float(np.min(ear_arr)),
            "MAR_mean_w": float(np.mean(mar_arr)),
            "MAR_max_w": float(np.max(mar_arr)),
            "blink_count_w": float(self._blink_count),
            "microsleep_count_w": float(self._microsleep_count),
            "yawn_count_w": float(self._yawn_count),
            "closed_run": float(self._closed_run),
            "EAR_vel": ear_vel,
        }


# Columns used for CSV baseline (tabular)
BASE_FEATURE_COLS = ["EAR", "MAR"]
# Extended columns if temporal extraction available
EXTENDED_FEATURE_COLS = [
    "EAR",
    "MAR",
    "EAR_L",
    "EAR_R",
    "pitch",
    "yaw",
    "roll",
    "PERCLOS",
    "EAR_mean_w",
    "EAR_std_w",
    "EAR_min_w",
    "MAR_mean_w",
    "MAR_max_w",
    "blink_count_w",
    "microsleep_count_w",
    "yawn_count_w",
    "closed_run",
    "EAR_vel",
]
