"""MediaPipe Face Landmarker — VIDEO mode + optional downscale for speed."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from mediapipe import Image as MPImage
from mediapipe import ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Key points only for overlay (much cheaper than drawing all 478)
DRAW_IDX = (
    # left eye
    33, 160, 158, 133, 153, 144,
    # right eye
    362, 385, 387, 263, 373, 380,
    # mouth
    61, 291, 13, 14, 81, 178,
    # face oval sample
    10, 152, 234, 454,
)


def resolve_model_path(explicit: Optional[str] = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("FACE_LANDMARKER_MODEL")
    if env:
        candidates.append(Path(env))
    code_dir = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            code_dir / "models" / "face_landmarker.task",
            Path.home() / ".cache" / "drowsiness-dds" / "face_landmarker.task",
        ]
    )
    for p in candidates:
        if p.is_file():
            return p

    dest = Path.home() / ".cache" / "drowsiness-dds" / "face_landmarker.task"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading FaceLandmarker model → {dest}")
    urllib.request.urlretrieve(DEFAULT_MODEL_URL, dest)
    local = code_dir / "models" / "face_landmarker.task"
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not local.exists():
            local.symlink_to(dest)
    except OSError:
        pass
    return dest


class FaceMeshExtractor:
    """
    FaceLandmarker with VIDEO running mode (temporal tracking → faster/stabler)
    or IMAGE mode for single frames.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        video_mode: bool = True,
    ) -> None:
        path = str(resolve_model_path(model_path))
        base_options = mp_python.BaseOptions(model_asset_path=path)
        mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mode,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._video_mode = video_mode
        self._ts_ms = 0

    def close(self) -> None:
        self._landmarker.close()

    def process(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[object]]:
        """
        Returns landmarks in **pixel coords of the given frame** (N, 3), and raw list.
        """
        h, w = frame_bgr.shape[:2]
        # MediaPipe expects contiguous RGB uint8
        if not frame_bgr.flags["C_CONTIGUOUS"]:
            frame_bgr = np.ascontiguousarray(frame_bgr)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)

        if self._video_mode:
            if timestamp_ms is None:
                self._ts_ms += 33  # ~30fps default step
                timestamp_ms = self._ts_ms
            else:
                self._ts_ms = timestamp_ms
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None, None
        face = result.face_landmarks[0]
        pts = np.empty((len(face), 3), dtype=np.float64)
        for i, lm in enumerate(face):
            pts[i, 0] = lm.x * w
            pts[i, 1] = lm.y * h
            pts[i, 2] = lm.z * w
        return pts, face

    def draw_keypoints(
        self,
        frame_bgr: np.ndarray,
        landmarks_xy: np.ndarray,
        indices: Sequence[int] = DRAW_IDX,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> None:
        """In-place draw of sparse keypoints (no full mesh)."""
        for i in indices:
            if i >= len(landmarks_xy):
                continue
            x, y = int(landmarks_xy[i, 0]), int(landmarks_xy[i, 1])
            cv2.circle(frame_bgr, (x, y), 2, color, -1, lineType=cv2.LINE_AA)

    def draw(self, frame_bgr: np.ndarray, face_landmarks) -> np.ndarray:
        """Backward-compatible: draw sparse points into a copy."""
        out = frame_bgr.copy()
        h, w = out.shape[:2]
        for i in DRAW_IDX:
            if i >= len(face_landmarks):
                continue
            lm = face_landmarks[i]
            cv2.circle(out, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)
        return out
