"""Voice quality micro-evaluator agent."""
from __future__ import annotations

from typing import Protocol

from ..models import SpeechObservation, VoiceScore


class VoiceQualityAgent(Protocol):
    def evaluate(self, speech: SpeechObservation) -> VoiceScore:
        ...


class DummyVoiceQualityAgent:
    """Returns neutral voice metrics to unblock orchestrator wiring."""

    def evaluate(self, speech: SpeechObservation) -> VoiceScore:
        return VoiceScore(
            clarity=3,
            terminology=3,
            coherence=3,
            pace="ok",
            fillers_per_min=speech.fillers_per_min,
            coach_notes=[],
        )

