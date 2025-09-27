"""Retrieval and RAG coverage agent."""
from __future__ import annotations

from typing import Protocol

from ..models import GroundingObservation, RetrievalResult, SpeechObservation, TurnBundle


class RetrievalAgent(Protocol):
    def retrieve(
        self,
        bundle: TurnBundle,
        speech: SpeechObservation,
        grounding: GroundingObservation | None,
    ) -> RetrievalResult | None:
        ...


class DummyRetrievalAgent:
    """Returns precomputed retrieval bundles supplied for testing."""

    def retrieve(
        self,
        bundle: TurnBundle,
        speech: SpeechObservation,
        grounding: GroundingObservation | None,
    ) -> RetrievalResult | None:
        return bundle.retrieval

