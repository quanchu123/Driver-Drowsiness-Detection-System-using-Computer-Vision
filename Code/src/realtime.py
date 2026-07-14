"""Realtime webcam drowsiness detection with state machine + optional ML model."""

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
import joblib
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import TemporalFeatureBuffer, frame_features_from_landmarks
from src.landmarks import FaceMeshExtractor
from src.preprocess import preprocess_frame


class State(str, Enum):
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    DROWSY = "DROWSY"
    ALERT = "ALERT"


@dataclass
class AlarmStateMachine:
    """Reduce false alarms: require sustained evidence before ALERT."""

    suspicious_frames: int = 8
    drowsy_frames: int = 20
    alert_cooldown_frames: int = 60
    state: State = State.NORMAL
    _low_run: int = 0
    _cooldown: int = 0

    def update(self, evidence: bool) -> State:
        """
        evidence=True means current frame/window looks drowsy
        (low EAR / high model score / high PERCLOS).
        """
        if self._cooldown > 0:
            self._cooldown -= 1
            self.state = State.ALERT if self._cooldown > self.alert_cooldown_frames // 2 else State.NORMAL
            if self._cooldown == 0:
                self.state = State.NORMAL
                self._low_run = 0
            return self.state

        if evidence:
            self._low_run += 1
        else:
            self._low_run = max(0, self._low_run - 2)

        if self._low_run >= self.drowsy_frames:
            self.state = State.ALERT
            self._cooldown = self.alert_cooldown_frames
            self._low_run = 0
        elif self._low_run >= self.suspicious_frames:
            self.state = State.SUSPICIOUS
        else:
            self.state = State.NORMAL
        return self.state


@dataclass
class AdaptiveEARCalibrator:
    """Per-driver EAR baseline during first N seconds (adaptable EAR idea)."""

    calibration_seconds: float = 30.0
    _samples: Deque[float] = field(default_factory=lambda: deque(maxlen=900))
    _start: Optional[float] = None
    baseline: Optional[float] = None

    def update(self, ear: float, now: Optional[float] = None) -> Optional[float]:
        now = now or time.time()
        if self._start is None:
            self._start = now
        if self.baseline is not None:
            return self.baseline
        self._samples.append(ear)
        if now - self._start >= self.calibration_seconds and len(self._samples) > 30:
            # Use lower quantile of open-eye distribution as personal closed threshold
            arr = np.array(self._samples)
            self.baseline = float(np.percentile(arr, 35) * 0.75)
            return self.baseline
        return None

    @property
    def done(self) -> bool:
        return self.baseline is not None


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime drowsiness detection")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--video", default=None, help="Optional video file instead of camera")
    parser.add_argument("--model", default=None, help="Path to best_model.joblib (optional)")
    parser.add_argument("--no-display", action="store_true", help="Headless: process N frames then exit")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0=infinite)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    rt = cfg.get("realtime", {})
    feat_cfg = cfg.get("features", {})

    model_pack = None
    model_path = Path(args.model) if args.model else ROOT / cfg.get("paths", {}).get("model_path", "artifacts/best_model.joblib")
    if model_path.is_file():
        model_pack = joblib.load(model_path)
        print(f"Loaded model: {model_path}")
    else:
        print("No trained model found — using EAR/MAR rules only")

    cam_id = args.camera if args.camera is not None else int(rt.get("camera_id", 0))
    src = args.video if args.video else cam_id
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: cannot open video source {src}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(rt.get("width", 640)))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(rt.get("height", 480)))

    extractor = FaceMeshExtractor()
    temporal = TemporalFeatureBuffer(
        ear_closed_threshold=float(feat_cfg.get("ear_closed_threshold", 0.25)),
        mar_yawn_threshold=float(feat_cfg.get("mar_yawn_threshold", 0.6)),
        window=int(feat_cfg.get("perclos_window", 90)),
        blink_min_frames=int(feat_cfg.get("blink_min_frames", 2)),
        blink_max_frames=int(feat_cfg.get("blink_max_frames", 8)),
        microsleep_min_frames=int(feat_cfg.get("microsleep_min_frames", 15)),
        yawn_min_frames=int(feat_cfg.get("yawn_min_frames", 10)),
    )
    sm = AlarmStateMachine(
        suspicious_frames=int(rt.get("suspicious_frames", 8)),
        drowsy_frames=int(rt.get("drowsy_frames", 20)),
        alert_cooldown_frames=int(rt.get("alert_cooldown_frames", 60)),
    )
    calibrator = AdaptiveEARCalibrator(
        calibration_seconds=float(rt.get("adaptive_calibration_seconds", 30))
    )

    use_clahe = bool(rt.get("use_clahe", True))
    ear_thr = float(feat_cfg.get("ear_closed_threshold", 0.25))
    mar_thr = float(feat_cfg.get("mar_yawn_threshold", 0.6))

    fps_t0 = time.time()
    fps_n = 0
    fps = 0.0
    frame_i = 0
    print("Controls: q=quit | r=reset temporal buffer")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_i += 1
            frame = preprocess_frame(frame, use_clahe=use_clahe)

            pts, face_lm = extractor.process(frame)
            evidence = False
            ear = mar = None
            perclos = 0.0
            model_score = None

            if pts is not None:
                ff = frame_features_from_landmarks(pts, frame.shape)
                ear, mar = ff["EAR"], ff["MAR"]
                thr = calibrator.update(ear) or ear_thr
                temporal.ear_closed_threshold = thr
                tf = temporal.update(ear, mar)
                perclos = tf["PERCLOS"]

                # Rule evidence
                rule_drowsy = (ear < thr) or (mar > mar_thr) or (perclos > 0.4)
                evidence = rule_drowsy

                # Optional ML on EAR, MAR (CSV-trained features)
                if model_pack is not None:
                    cols = model_pack["feature_cols"]
                    vec = []
                    for c in cols:
                        if c == "EAR":
                            vec.append(ear)
                        elif c == "MAR":
                            vec.append(mar)
                        else:
                            vec.append(ff.get(c, tf.get(c, 0.0)))
                    X = np.array([vec], dtype=np.float64)
                    model = model_pack["model"]
                    if hasattr(model, "predict_proba"):
                        model_score = float(model.predict_proba(X)[0, 1])
                    else:
                        model_score = float(model.predict(X)[0])
                    evidence = evidence or (model_score >= 0.5)

                if face_lm is not None:
                    frame = extractor.draw(frame, face_lm)

            state = sm.update(evidence)

            # Overlay
            color = {
                State.NORMAL: (0, 200, 0),
                State.SUSPICIOUS: (0, 200, 255),
                State.DROWSY: (0, 140, 255),
                State.ALERT: (0, 0, 255),
            }[state]
            lines = [
                f"State: {state.value}",
                f"EAR: {ear:.3f}" if ear is not None else "EAR: --",
                f"MAR: {mar:.3f}" if mar is not None else "MAR: --",
                f"PERCLOS: {perclos:.2f}",
                f"EAR_thr: {temporal.ear_closed_threshold:.3f} cal={'OK' if calibrator.done else '...'}",
                f"FPS: {fps:.1f}",
            ]
            if model_score is not None:
                lines.append(f"ML score: {model_score:.2f}")
            for i, text in enumerate(lines):
                cv2.putText(
                    frame, text, (10, 28 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
                )
            if state == State.ALERT:
                cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 6)
                cv2.putText(
                    frame, "DROWSINESS ALERT!", (40, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA,
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
                    sm = AlarmStateMachine(
                        suspicious_frames=sm.suspicious_frames,
                        drowsy_frames=sm.drowsy_frames,
                        alert_cooldown_frames=sm.alert_cooldown_frames,
                    )
            elif args.max_frames and frame_i >= args.max_frames:
                break
            if args.max_frames and frame_i >= args.max_frames:
                break
    finally:
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_i} frames, last FPS≈{fps:.1f}")


if __name__ == "__main__":
    main()
