"""Rubric-based scoring agent interface."""
from __future__ import annotations

from typing import Protocol

from ..models import (
    ContentScore,
    GroundingObservation,
    NextAction,
    RetrievalResult,
    ScoringResult,
    SpeechObservation,
    TurnBundle,
    VisionObservation,
    VoiceScore,
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


class RuleBasedScoringAgent:
    """Heuristic scorer aligned with the regression tutoring rubric."""

    def score(
        self,
        bundle: TurnBundle,
        speech: SpeechObservation,
        vision: VisionObservation | None,
        grounding: GroundingObservation | None,
        retrieval: RetrievalResult | None,
    ) -> ScoringResult:
        transcript = speech.transcript.lower()

        mentions_variance = "variance" in transcript or "percent" in transcript
        mentions_r2 = "r^2" in transcript or "r-squared" in transcript or "sixty four" in transcript or "64" in transcript
        mentions_hours = "hour" in transcript
        explains_non_causal = "because" not in transcript and "cause" not in transcript

        points = sum([mentions_variance, mentions_r2, mentions_hours, explains_non_causal])
        content_score = max(0, min(4, points))

        misconceptions: list[str] = []
        if "cause" in transcript or "causes" in transcript:
            misconceptions.append("R² should not be interpreted as causal.")

        evidence = [f"{hit.id}:{hit.span}" for hit in retrieval.hits] if retrieval else []

        voice_score = _score_voice(speech)
        unit_check = {
            "slope_units_present": "per" in transcript or "each" in transcript,
            "mentions_hours": mentions_hours,
        }

        content = ContentScore(
            content_score=content_score,
            evidence=evidence,
            misconceptions=misconceptions,
            assumption_flags=[],
            unit_check=unit_check,
        )

        next_action = _next_action_from_scores(content, retrieval)

        ui_notes = {
            "need_camera_reframe": bool(vision and vision.lighting == "dim"),
            "speech_confidence": speech.asr_conf,
        }

        return ScoringResult(
            content=content,
            voice=voice_score,
            next_action=next_action,
            ui_notes=ui_notes,
        )


def _score_voice(speech: SpeechObservation) -> VoiceScore:
    wpm = speech.wpm
    pace = "ok"
    if wpm and (wpm < 90 or wpm > 170):
        pace = "slow" if wpm < 90 else "fast"

    clarity = 1 if speech.transcript.strip() == "" else 3
    terminology = 3 if "variance" in speech.transcript.lower() else 2
    coherence = 3 if len(speech.transcript.split()) > 6 else 2

    return VoiceScore(
        clarity=clarity,
        terminology=terminology,
        coherence=coherence,
        pace=pace,
        fillers_per_min=speech.fillers_per_min,
        coach_notes=[],
    )


def _next_action_from_scores(
    content: ContentScore, retrieval: RetrievalResult | None
) -> NextAction:
    refs = [hit.id for hit in retrieval.hits[:1]] if retrieval else []

    if content.misconceptions:
        return NextAction(
            type="explain_short",
            payload="R² reports variance explained. It does not guarantee causality.",
            rag_refs=refs,
        )

    if content.content_score <= 2:
        return NextAction(
            type="ask_guided_q",
            payload="Restate in words what R²=0.64 tells you about exam scores.",
            rag_refs=refs,
        )

    return NextAction(
        type="affirm",
        payload="Great explanation—highlight that 64% of score variance is explained by study time.",
        rag_refs=refs,
    )

