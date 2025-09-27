"""Vision sensing agent built around OpenCV."""
from __future__ import annotations

from typing import Protocol

from ..models import TurnBundle, VisionObservation


class VisionAgent(Protocol):
    def observe(self, bundle: TurnBundle) -> VisionObservation | None:
        ...


class DummyVisionAgent:
    """Stub implementation that would be replaced by OpenCV pipeline."""

    def observe(self, bundle: TurnBundle) -> VisionObservation | None:
        return bundle.vision

