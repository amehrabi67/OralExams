"""Safety and privacy guardrails."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from ..models import SafetyResult, SafetySignal, TurnBundle


@dataclass
class SafetyConfig:
    enforce_offline: bool = True
    max_buffer_seconds: int = 10


class SafetyAgent(Protocol):
    def check(self, bundle: TurnBundle) -> SafetyResult:
        ...


class DefaultSafetyAgent:
    """Placeholder safety implementation enforcing the global contracts."""

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()

    def check(self, bundle: TurnBundle) -> SafetyResult:
        # Real implementation would inspect device state and bundle metadata.
        return SafetyResult(signal=SafetySignal.SAFE_OK)


@dataclass
class LocalSafetyConfig(SafetyConfig):
    """Thresholds for the on-device safety checks."""

    min_mean_luma: float = 40.0
    max_faces: int = 1
    face_detector: str | None = None


class LocalSafetyAgent:
    """Safety agent that inspects the current video frame for violations."""

    def __init__(self, config: LocalSafetyConfig | None = None) -> None:
        self.config = config or LocalSafetyConfig()
        cascade_path = (
            self.config.face_detector
            or cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._face_cascade = cv2.CascadeClassifier(cascade_path)
        if self._face_cascade.empty():
            raise RuntimeError(f"Failed to load face detector cascade from {cascade_path}")

    def check(self, bundle: TurnBundle) -> SafetyResult:
        frame = bundle.raw_video_frame
        if frame is None:
            # No frame provided – treat as degraded (text only) mode.
            return SafetyResult(signal=SafetySignal.SAFE_OK)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        luma = float(np.mean(gray))
        remediation: list[str] = []

        if luma < self.config.min_mean_luma:
            remediation.append("Increase lighting or tilt the page toward the light source.")

        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > self.config.max_faces:
            remediation.append("Ensure only one learner is in frame before continuing.")

        if remediation:
            return SafetyResult(signal=SafetySignal.SAFE_STOP, remediation=remediation)

        return SafetyResult(signal=SafetySignal.SAFE_OK)

