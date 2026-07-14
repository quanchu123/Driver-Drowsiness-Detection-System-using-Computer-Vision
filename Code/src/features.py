"""Geometric & temporal drowsiness features — optimized for realtime CPU."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Sequence, Tuple

import numpy as np

# MediaPipe FaceMesh landmark indices
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
# Mouth
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_TOP2, MOUTH_BOTTOM2 = 81, 178
# Head-nod proxy (cheap, no solvePnP)
NOSE_TIP, FOREHEAD, CHIN = 1, 10, 152


def _euclid2(ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = ax - bx, ay - by
    return (dx * dx + dy * dy) ** 0.5


def eye_aspect_ratio(landmarks: np.ndarray, eye_idx: Sequence[int]) -> float:
    """EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||) — pure float math."""
    p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in eye_idx)
    num = _euclid2(p2[0], p2[1], p6[0], p6[1]) + _euclid2(p3[0], p3[1], p5[0], p5[1])
    den = 2.0 * _euclid2(p1[0], p1[1], p4[0], p4[1]) + 1e-8
    return num / den


def mouth_aspect_ratio(landmarks: np.ndarray) -> float:
    left, right = landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT]
    top, bottom = landmarks[MOUTH_TOP], landmarks[MOUTH_BOTTOM]
    top2, bottom2 = landmarks[MOUTH_TOP2], landmarks[MOUTH_BOTTOM2]
    vert = _euclid2(top[0], top[1], bottom[0], bottom[1]) + _euclid2(
        top2[0], top2[1], bottom2[0], bottom2[1]
    )
    horiz = _euclid2(left[0], left[1], right[0], right[1]) + 1e-8
    return vert / (2.0 * horiz)


def head_nod_score(landmarks: np.ndarray) -> float:
    """
    Cheap head-nod proxy in [0, ~1+]: nose closer to chin vs forehead → nodding down.
    Avoids solvePnP (heavy) every frame.
    """
    nose, fore, chin = landmarks[NOSE_TIP], landmarks[FOREHEAD], landmarks[CHIN]
    up = _euclid2(nose[0], nose[1], fore[0], fore[1]) + 1e-8
    down = _euclid2(nose[0], nose[1], chin[0], chin[1]) + 1e-8
    # When head tilts down, nose moves toward chin → down/up increases
    return down / up


def frame_features_from_landmarks(
    landmarks_xy: np.ndarray,
    image_shape: Optional[Tuple[int, int]] = None,
    compute_pose: bool = False,
) -> Dict[str, float]:
    """Per-frame geometric features. Pose (solvePnP) off by default for speed."""
    ear_l = eye_aspect_ratio(landmarks_xy, LEFT_EYE)
    ear_r = eye_aspect_ratio(landmarks_xy, RIGHT_EYE)
    ear = 0.5 * (ear_l + ear_r)
    mar = mouth_aspect_ratio(landmarks_xy)
    nod = head_nod_score(landmarks_xy)
    out = {
        "EAR": ear,
        "EAR_L": ear_l,
        "EAR_R": ear_r,
        "MAR": mar,
        "nod": nod,
        "pitch": 0.0,
        "yaw": 0.0,
        "roll": 0.0,
    }
    if compute_pose and image_shape is not None:
        pitch, yaw, roll = head_pose_solvePnP(landmarks_xy, image_shape)
        out["pitch"], out["yaw"], out["roll"] = pitch, yaw, roll
    return out


def head_pose_solvePnP(
    landmarks: np.ndarray, image_shape: Tuple[int, int]
) -> Tuple[float, float, float]:
    """Optional full pose — only when explicitly enabled."""
    import cv2

    h, w = image_shape[:2]
    image_points = np.array(
        [
            landmarks[NOSE_TIP][:2],
            landmarks[CHIN][:2],
            landmarks[LEFT_EYE[0]][:2],
            landmarks[RIGHT_EYE[0]][:2],
            landmarks[MOUTH_LEFT][:2],
            landmarks[MOUTH_RIGHT][:2],
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
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        model_points, image_points, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    sy = (rmat[0, 0] ** 2 + rmat[1, 0] ** 2) ** 0.5
    pitch = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
    yaw = float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])))
    roll = float(np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2])))
    return pitch, yaw, roll


@dataclass
class EMASmoother:
    """Exponential moving average to reduce landmark jitter."""

    alpha: float = 0.35  # higher = more responsive, lower = smoother
    _ear: Optional[float] = None
    _mar: Optional[float] = None
    _nod: Optional[float] = None

    def update(self, ear: float, mar: float, nod: float = 0.0) -> Tuple[float, float, float]:
        a = self.alpha
        if self._ear is None:
            self._ear, self._mar, self._nod = ear, mar, nod
        else:
            self._ear = a * ear + (1 - a) * self._ear
            self._mar = a * mar + (1 - a) * self._mar
            self._nod = a * nod + (1 - a) * self._nod
        return self._ear, self._mar, self._nod

    def reset(self) -> None:
        self._ear = self._mar = self._nod = None


@dataclass
class TemporalFeatureBuffer:
    """
    O(1) sliding-window PERCLOS / blink / microsleep / yawn.
    Avoids reallocating numpy arrays every frame.
    """

    ear_closed_threshold: float = 0.25
    mar_yawn_threshold: float = 0.6
    nod_threshold: float = 1.35
    window: int = 90
    blink_min_frames: int = 2
    blink_max_frames: int = 8
    microsleep_min_frames: int = 15
    yawn_min_frames: int = 10
    _closed_flags: Deque[int] = field(default_factory=deque)
    _closed_sum: int = 0
    _closed_run: int = 0
    _yawn_run: int = 0
    _nod_run: int = 0
    _blink_count: int = 0
    _microsleep_count: int = 0
    _yawn_count: int = 0
    _nod_count: int = 0
    _ear_sum: float = 0.0
    _ears: Deque[float] = field(default_factory=deque)
    _prev_ear: Optional[float] = None

    def __post_init__(self) -> None:
        self._closed_flags = deque(maxlen=self.window)
        self._ears = deque(maxlen=self.window)

    def reset(self) -> None:
        self._closed_flags.clear()
        self._ears.clear()
        self._closed_sum = 0
        self._closed_run = 0
        self._yawn_run = 0
        self._nod_run = 0
        self._blink_count = 0
        self._microsleep_count = 0
        self._yawn_count = 0
        self._nod_count = 0
        self._ear_sum = 0.0
        self._prev_ear = None

    def update(self, ear: float, mar: float, nod: float = 0.0) -> Dict[str, float]:
        closed = 1 if ear < self.ear_closed_threshold else 0
        yawning = mar > self.mar_yawn_threshold
        nodding = nod > self.nod_threshold

        # Rolling PERCLOS (binary closed flags)
        if len(self._closed_flags) == self.window:
            self._closed_sum -= self._closed_flags[0]
        self._closed_flags.append(closed)
        self._closed_sum += closed

        if len(self._ears) == self.window:
            self._ear_sum -= self._ears[0]
        self._ears.append(ear)
        self._ear_sum += ear

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

        if nodding:
            self._nod_run += 1
            if self._nod_run == 12:  # ~0.4s sustained nod
                self._nod_count += 1
        else:
            self._nod_run = 0

        n = len(self._closed_flags)
        perclos = self._closed_sum / n if n else 0.0
        ear_mean = self._ear_sum / n if n else ear
        ear_vel = 0.0 if self._prev_ear is None else ear - self._prev_ear
        self._prev_ear = ear

        return {
            "PERCLOS": perclos,
            "EAR_mean_w": ear_mean,
            "EAR_vel": ear_vel,
            "blink_count_w": float(self._blink_count),
            "microsleep_count_w": float(self._microsleep_count),
            "yawn_count_w": float(self._yawn_count),
            "nod_count_w": float(self._nod_count),
            "closed_run": float(self._closed_run),
            "yawn_run": float(self._yawn_run),
            "nod_run": float(self._nod_run),
        }


def drowsiness_score(
    ear: float,
    mar: float,
    perclos: float,
    closed_run: float,
    yawn_run: float,
    nod_run: float,
    ear_thr: float,
    mar_thr: float,
    perclos_thr: float = 0.35,
    nod_run_thr: float = 10.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Multi-cue continuous score in [0, 1] — no ML model.
    Weighted fusion of eye closure, yawn, PERCLOS, head nod.
    """
    # Soft eye-closed: how far below threshold
    eye_c = 0.0
    if ear_thr > 1e-6:
        eye_c = max(0.0, min(1.0, (ear_thr - ear) / (ear_thr * 0.5 + 1e-6)))

    yawn_c = 0.0
    if mar > mar_thr:
        yawn_c = max(0.0, min(1.0, (mar - mar_thr) / (mar_thr + 1e-6)))

    perclos_c = max(0.0, min(1.0, perclos / max(perclos_thr, 1e-6)))
    # Sustained closure (microsleep path)
    micro_c = max(0.0, min(1.0, closed_run / 20.0))
    nod_c = max(0.0, min(1.0, nod_run / nod_run_thr))

    # Weights: eyes dominate, then PERCLOS/micro, yawn, nod
    score = (
        0.40 * eye_c
        + 0.25 * perclos_c
        + 0.15 * micro_c
        + 0.12 * yawn_c
        + 0.08 * nod_c
    )
    score = max(0.0, min(1.0, score))
    parts = {
        "eye": eye_c,
        "perclos": perclos_c,
        "micro": micro_c,
        "yawn": yawn_c,
        "nod": nod_c,
    }
    return score, parts


BASE_FEATURE_COLS = ["EAR", "MAR"]
