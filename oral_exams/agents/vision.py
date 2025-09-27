"""Vision sensing agent built around OpenCV."""
from __future__ import annotations

from datetime import datetime
from typing import Protocol

import cv2
import numpy as np

from ..models import TurnBundle, VisionActivity, VisionObservation, VisionROI


class VisionAgent(Protocol):
    def observe(self, bundle: TurnBundle) -> VisionObservation | None:
        ...


class OpenCVVisionAgent:
    """OpenCV-backed vision agent that extracts lightweight heuristics."""

    def __init__(
        self,
        smoothing: float = 0.7,
        dim_threshold: float = 40.0,
        bright_threshold: float = 200.0,
    ) -> None:
        self._prev_face: VisionROI | None = None
        self._smoothing = smoothing
        self._dim_threshold = dim_threshold
        self._bright_threshold = bright_threshold
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)

    def observe(self, bundle: TurnBundle) -> VisionObservation | None:
        frame = bundle.raw_video_frame
        if frame is None:
            return None

        timestamp = datetime.utcnow().isoformat()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        face_roi: VisionROI | None = None
        multiperson = len(faces) > 1

        if len(faces) >= 1:
            x, y, w, h = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)[0]
            smoothed = np.array([x, y, w, h], dtype=float)
            if self._prev_face is not None:
                prev = np.array(self._prev_face.bbox, dtype=float)
                smoothed = self._smoothing * prev + (1 - self._smoothing) * smoothed
            face_roi = VisionROI(bbox=smoothed.tolist())
            self._prev_face = face_roi

        lighting = "ok"
        luma = float(np.mean(gray))
        if luma < self._dim_threshold:
            lighting = "dim"
        elif luma > self._bright_threshold:
            lighting = "bright"

        activity = VisionActivity(writing=False, pointing=False, reading=False)

        cv_conf = 0.3
        if face_roi:
            cv_conf += 0.4
        if lighting == "ok":
            cv_conf += 0.2
        if not multiperson:
            cv_conf += 0.1

        return VisionObservation(
            timestamp=timestamp,
            cv_conf=min(cv_conf, 0.99),
            lighting=lighting,
            multiperson=multiperson,
            face=face_roi,
            activity=activity,
        )

