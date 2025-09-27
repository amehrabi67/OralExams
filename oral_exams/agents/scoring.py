"""Rubric-based scoring agent interface."""
from __future__ import annotations

from typing import Protocol

from ..models import (
    GroundingObservation,
    RetrievalResult,
    ScoringResult,
    SpeechObservation,
    TurnBundle,
    VisionObservation,
)


class ScoringAgent(Protocol):
    def score(
        self,
        bundle: TurnBundle,
        speech: SpeechObservation,
        vision: VisionObservation | None,
        grounding: GroundingObservation | None,
        retrieval: RetrievalResult | None,
    ) -> ScoringResult:
        ...


class DummyScoringAgent:
    """Simple scorer that returns a neutral result for pipelines tests."""

    def score(
        self,
        bundle: TurnBundle,
        speech: SpeechObservation,
        vision: VisionObservation | None,
        grounding: GroundingObservation | None,
        retrieval: RetrievalResult | None,
    ) -> ScoringResult:
        raise NotImplementedError("Provide a concrete scoring agent for production use")

