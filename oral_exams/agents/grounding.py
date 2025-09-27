"""Visual grounding agent placeholder."""
from __future__ import annotations

from typing import Protocol

from ..models import GroundingObservation, TurnBundle, VisionObservation


class GroundingAgent(Protocol):
    def ground(
        self, bundle: TurnBundle, vision: VisionObservation
    ) -> GroundingObservation | None:
        ...


class DummyGroundingAgent:
    """Pass-through grounding agent for early integration tests."""

    def ground(
        self, bundle: TurnBundle, vision: VisionObservation
    ) -> GroundingObservation | None:
        return bundle.grounding

