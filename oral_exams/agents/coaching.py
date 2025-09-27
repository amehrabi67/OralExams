"""Coaching decision logic."""
from __future__ import annotations

from typing import Protocol

from ..models import NextAction, RetrievalResult, ScoringResult, TurnBundle


class CoachingAgent(Protocol):
    def handle_low_coverage(
        self, bundle: TurnBundle, retrieval: RetrievalResult | None
    ) -> NextAction:
        ...

    def coach(
        self,
        bundle: TurnBundle,
        scoring: ScoringResult,
        retrieval: RetrievalResult | None,
    ) -> NextAction:
        ...


class DefaultCoachingAgent:
    """Minimal implementation that echoes the scoring recommendation."""

    def handle_low_coverage(
        self, bundle: TurnBundle, retrieval: RetrievalResult | None
    ) -> NextAction:
        payload = "Not in notes. Nearest prerequisite: variance vs variance explained."
        refs = retrieval.hits[0:1] if retrieval else []
        return NextAction(
            type="inform_gap",
            payload=payload,
            rag_refs=[hit.id for hit in refs],
        )

    def coach(
        self,
        bundle: TurnBundle,
        scoring: ScoringResult,
        retrieval: RetrievalResult | None,
    ) -> NextAction:
        return scoring.next_action

