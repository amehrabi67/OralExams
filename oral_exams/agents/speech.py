"""Speech transcription agent wrapper."""
from __future__ import annotations

from typing import Protocol

from ..models import SpeechObservation, TurnBundle


class SpeechAgent(Protocol):
    def transcribe(self, bundle: TurnBundle) -> SpeechObservation:
        ...


class DummySpeechAgent:
    """Returns the speech observation provided in the bundle."""

    def transcribe(self, bundle: TurnBundle) -> SpeechObservation:
        return bundle.speech

