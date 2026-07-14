"""MediaPipe Face Landmarker (Tasks API) landmark extraction."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe import Image as MPImage
from mediapipe import ImageFormat

DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def resolve_model_path(explicit: Optional[str] = None) -> Path:
    """Find or download face_landmarker.task."""
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

    # download to user cache
    dest = Path.home() / ".cache" / "drowsiness-dds" / "face_landmarker.task"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading FaceLandmarker model → {dest}")
    urllib.request.urlretrieve(DEFAULT_MODEL_URL, dest)
    # best-effort local copy for repo
    local = code_dir / "models" / "face_landmarker.task"
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not local.exists():
            local.symlink_to(dest)
    except OSError:
        pass
    return dest


class FaceMeshExtractor:
    """Face landmark extractor compatible with MediaPipe Tasks FaceLandmarker."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
    ) -> None:
        path = str(resolve_model_path(model_path))
        base_options = mp_python.BaseOptions(model_asset_path=path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def process(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[object]]:
        """
        Returns:
            landmarks_xy: (N, 3) pixel coords or None
            raw_landmarks: list of normalized landmarks or None
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None, None
        face = result.face_landmarks[0]
        pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in face], dtype=np.float64)
        return pts, face

    def draw(self, frame_bgr: np.ndarray, face_landmarks) -> np.ndarray:
        """Draw simple landmark dots (Tasks API has no tesselation drawer)."""
        out = frame_bgr.copy()
        h, w = out.shape[:2]
        for lm in face_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(out, (x, y), 1, (0, 255, 0), -1)
        return out
