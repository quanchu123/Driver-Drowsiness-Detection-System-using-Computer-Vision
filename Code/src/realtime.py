"""
Realtime drowsiness detection — optimized geometry path (no ML training).

Pipeline:
  capture → (optional downscale) → night-aware preprocess → FaceLandmarker VIDEO
  → EMA-smoothed EAR/MAR/nod → PERCLOS/blink window → multi-cue score
  → state machine alert
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Deque, Optional

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import (
    EMASmoother,
    TemporalFeatureBuffer,
    drowsiness_score,
    frame_features_from_landmarks,
)
from src.landmarks import FaceMeshExtractor
from src.preprocess import preprocess_frame, resize_for_detect


class State(str, Enum):
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    DROWSY = "DROWSY"
    ALERT = "ALERT"


@dataclass
class AlarmStateMachine:
    """Hysteresis + cooldown to cut false alarms from blinks."""

    suspicious_score: float = 0.35
    drowsy_score: float = 0.55
    alert_score: float = 0.70
    suspicious_frames: int = 6
    drowsy_frames: int = 15
    alert_frames: int = 22
    alert_cooldown_frames: int = 45
    # decay when evidence drops
    decay: int = 2
    state: State = State.NORMAL
    _run: int = 0
    _cooldown: int = 0

    def update(self, score: float) -> State:
        if self._cooldown > 0:
            self._cooldown -= 1
            self.state = (
                State.ALERT
                if self._cooldown > self.alert_cooldown_frames // 2
                else State.NORMAL
            )
            if self._cooldown == 0:
                self._run = 0
                self.state = State.NORMAL
            return self.state

        if score >= self.suspicious_score:
            self._run += 1
        else:
            self._run = max(0, self._run - self.decay)

        if score >= self.alert_score and self._run >= self.alert_frames:
            self.state = State.ALERT
            self._cooldown = self.alert_cooldown_frames
            self._run = 0
        elif score >= self.drowsy_score and self._run >= self.drowsy_frames:
            self.state = State.DROWSY
        elif score >= self.suspicious_score and self._run >= self.suspicious_frames:
            self.state = State.SUSPICIOUS
        else:
            self.state = State.NORMAL
        return self.state

    def reset(self) -> None:
        self.state = State.NORMAL
        self._run = 0
        self._cooldown = 0


@dataclass
class AdaptiveEARCalibrator:
    """
    Personal open-eye EAR → closed threshold.
    Uses robust stats; updates lightly after calibration (slow drift).
    """

    calibration_seconds: float = 20.0
    closed_ratio: float = 0.72  # thr = open_baseline * ratio
    min_samples: int = 40
    _samples: Deque[float] = field(default_factory=lambda: deque(maxlen=600))
    _start: Optional[float] = None
    open_baseline: Optional[float] = None
    threshold: Optional[float] = None

    def update(self, ear: float, now: Optional[float] = None) -> Optional[float]:
        now = now or time.time()
        if self._start is None:
            self._start = now

        # During calibration only keep higher EAR (likely open eyes)
        if self.threshold is None:
            self._samples.append(ear)
            if (
                now - self._start >= self.calibration_seconds
                and len(self._samples) >= self.min_samples
            ):
                arr = np.asarray(self._samples, dtype=np.float64)
                # Robust open-eye level: 60th percentile of samples
                self.open_baseline = float(np.percentile(arr, 60))
                self.threshold = max(0.12, min(0.35, self.open_baseline * self.closed_ratio))
            return self.threshold

        # Slow adapt: only when clearly open (ear well above thr)
        if ear > self.threshold * 1.25 and self.open_baseline is not None:
            self.open_baseline = 0.995 * self.open_baseline + 0.005 * ear
            self.threshold = max(0.12, min(0.35, self.open_baseline * self.closed_ratio))
        return self.threshold

    @property
    def done(self) -> bool:
        return self.threshold is not None


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized realtime drowsiness detection (no ML train)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-draw", action="store_true", help="Skip landmark overlay (faster)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    rt = cfg.get("realtime", {})
    feat_cfg = cfg.get("features", {})

    cam_id = args.camera if args.camera is not None else int(rt.get("camera_id", 0))
    src = args.video if args.video else cam_id
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: cannot open video source {src}")
        sys.exit(1)

    width = int(rt.get("width", 640))
    height = int(rt.get("height", 480))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Reduce camera buffer latency when supported
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    detect_width = int(rt.get("detect_width", 480))
    process_every = max(1, int(rt.get("process_every_n", 1)))
    use_clahe = bool(rt.get("use_clahe", True))
    draw = not args.no_draw and bool(rt.get("draw_landmarks", True))

    model_path = cfg.get("paths", {}).get("face_landmarker_model")
    if model_path and not Path(model_path).is_absolute():
        model_path = str((ROOT / model_path).resolve())
    extractor = FaceMeshExtractor(
        model_path=model_path if model_path and Path(model_path).is_file() else None,
        video_mode=True,
    )

    ear_thr_default = float(feat_cfg.get("ear_closed_threshold", 0.22))
    mar_thr = float(feat_cfg.get("mar_yawn_threshold", 0.55))
    perclos_thr = float(feat_cfg.get("perclos_threshold", 0.35))
    score_thr = float(feat_cfg.get("score_evidence_threshold", 0.45))

    temporal = TemporalFeatureBuffer(
        ear_closed_threshold=ear_thr_default,
        mar_yawn_threshold=mar_thr,
        nod_threshold=float(feat_cfg.get("nod_threshold", 1.35)),
        window=int(feat_cfg.get("perclos_window", 60)),
        blink_min_frames=int(feat_cfg.get("blink_min_frames", 2)),
        blink_max_frames=int(feat_cfg.get("blink_max_frames", 7)),
        microsleep_min_frames=int(feat_cfg.get("microsleep_min_frames", 12)),
        yawn_min_frames=int(feat_cfg.get("yawn_min_frames", 8)),
    )
    smoother = EMASmoother(alpha=float(rt.get("ema_alpha", 0.35)))
    sm = AlarmStateMachine(
        suspicious_score=float(rt.get("suspicious_score", 0.35)),
        drowsy_score=float(rt.get("drowsy_score", 0.55)),
        alert_score=float(rt.get("alert_score", 0.70)),
        suspicious_frames=int(rt.get("suspicious_frames", 6)),
        drowsy_frames=int(rt.get("drowsy_frames", 15)),
        alert_frames=int(rt.get("alert_frames", 22)),
        alert_cooldown_frames=int(rt.get("alert_cooldown_frames", 45)),
    )
    calibrator = AdaptiveEARCalibrator(
        calibration_seconds=float(rt.get("adaptive_calibration_seconds", 20)),
        closed_ratio=float(rt.get("ear_closed_ratio", 0.72)),
    )

    fps_t0 = time.time()
    fps_n = 0
    fps = 0.0
    frame_i = 0
    t0_wall = time.time()
    # cache last good detection when skipping frames
    last = {
        "ear": None,
        "mar": None,
        "nod": None,
        "perclos": 0.0,
        "score": 0.0,
        "parts": {},
        "pts": None,
    }

    print("Optimized realtime (geometry only — no ML train)")
    print("Controls: q=quit | r=reset | d=toggle draw")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_i += 1
            now = time.time()
            ts_ms = int((now - t0_wall) * 1000)

            run_detect = (frame_i % process_every == 0) or last["ear"] is None

            if run_detect:
                # Downscale for detector, keep full frame for display
                small, inv_scale = resize_for_detect(frame, max_width=detect_width)
                small = preprocess_frame(
                    small,
                    use_clahe=use_clahe,
                    auto_night=True,
                    always_clahe=False,
                )
                pts_small, _ = extractor.process(small, timestamp_ms=ts_ms)

                if pts_small is not None:
                    # map landmarks back to full-res if needed
                    if inv_scale != 1.0:
                        pts = pts_small.copy()
                        pts[:, 0] *= inv_scale
                        pts[:, 1] *= inv_scale
                        pts[:, 2] *= inv_scale
                    else:
                        pts = pts_small

                    ff = frame_features_from_landmarks(pts, compute_pose=False)
                    ear_r, mar_r, nod_r = ff["EAR"], ff["MAR"], ff["nod"]
                    ear, mar, nod = smoother.update(ear_r, mar_r, nod_r)

                    thr = calibrator.update(ear, now=now) or ear_thr_default
                    temporal.ear_closed_threshold = thr
                    tf = temporal.update(ear, mar, nod)

                    score, parts = drowsiness_score(
                        ear=ear,
                        mar=mar,
                        perclos=tf["PERCLOS"],
                        closed_run=tf["closed_run"],
                        yawn_run=tf["yawn_run"],
                        nod_run=tf["nod_run"],
                        ear_thr=thr,
                        mar_thr=mar_thr,
                        perclos_thr=perclos_thr,
                    )
                    last.update(
                        ear=ear,
                        mar=mar,
                        nod=nod,
                        perclos=tf["PERCLOS"],
                        score=score,
                        parts=parts,
                        pts=pts,
                        thr=thr,
                        closed_run=tf["closed_run"],
                    )
                else:
                    # face lost → decay score
                    last["score"] = max(0.0, last["score"] * 0.85)
                    last["pts"] = None

            score = float(last["score"])
            state = sm.update(score)

            # Overlay on original frame
            color = {
                State.NORMAL: (0, 200, 0),
                State.SUSPICIOUS: (0, 200, 255),
                State.DROWSY: (0, 140, 255),
                State.ALERT: (0, 0, 255),
            }[state]

            if draw and last.get("pts") is not None:
                extractor.draw_keypoints(frame, last["pts"], color=color)

            thr_s = last.get("thr", ear_thr_default)
            lines = [
                f"State: {state.value}",
                f"Score: {score:.2f}",
                f"EAR: {last['ear']:.3f}" if last["ear"] is not None else "EAR: --",
                f"MAR: {last['mar']:.3f}" if last["mar"] is not None else "MAR: --",
                f"PERCLOS: {last['perclos']:.2f}",
                f"EAR_thr: {thr_s:.3f} cal={'OK' if calibrator.done else '...'}",
                f"FPS: {fps:.1f}",
            ]
            for i, text in enumerate(lines):
                cv2.putText(
                    frame,
                    text,
                    (10, 26 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # score bar
            bar_w = int(200 * score)
            cv2.rectangle(frame, (10, frame.shape[0] - 30), (210, frame.shape[0] - 12), (40, 40, 40), -1)
            cv2.rectangle(
                frame,
                (10, frame.shape[0] - 30),
                (10 + bar_w, frame.shape[0] - 12),
                color,
                -1,
            )

            if state == State.ALERT:
                cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 5)
                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (30, frame.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

            fps_n += 1
            if time.time() - fps_t0 >= 1.0:
                fps = fps_n / (time.time() - fps_t0)
                fps_t0 = time.time()
                fps_n = 0

            if not args.no_display:
                cv2.imshow("Driver Drowsiness Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    temporal.reset()
                    smoother.reset()
                    sm.reset()
                if key == ord("d"):
                    draw = not draw
            if args.max_frames and frame_i >= args.max_frames:
                break
    finally:
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_i} frames, last FPS≈{fps:.1f}")


if __name__ == "__main__":
    main()
