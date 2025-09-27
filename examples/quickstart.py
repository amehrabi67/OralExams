"""Minimal quickstart for exercising the orchestrator without hardware."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from oral_exams.agents.coaching import DefaultCoachingAgent
from oral_exams.agents.safety import DefaultSafetyAgent
from oral_exams.agents.scoring import RuleBasedScoringAgent
from oral_exams.models import (
    GroundingObservation,
    ItemContext,
    NextAction,
    RetrievalHit,
    RetrievalResult,
    SpeechObservation,
    TurnBundle,
    VisionActivity,
    VisionObservation,
)
from oral_exams.orchestrator import Orchestrator


class DemoVisionAgent:
    """Return a deterministic observation that clears the confidence gate."""

    def observe(self, bundle: TurnBundle) -> VisionObservation:
        return VisionObservation(
            timestamp=datetime.utcnow().isoformat(),
            cv_conf=0.72,
            lighting="ok",
            multiperson=False,
            activity=VisionActivity(writing=True, pointing=False, reading=False),
        )


class DemoGroundingAgent:
    """Pretend the camera spotted the R² statement on the page."""

    def ground(self, bundle: TurnBundle, vision: VisionObservation) -> GroundingObservation:
        return GroundingObservation(
            vlm_conf=0.78,
            grounding=[{"region": "board", "tag": "R^2=0.64", "bbox": [10, 10, 120, 40]}],
            math_fragments=["R^2=0.64"],
        )


class DemoSpeechAgent:
    """Stub speech agent that returns a canned transcript."""

    def transcribe(self, bundle: TurnBundle) -> SpeechObservation:
        transcript = (
            "Sixty four percent of the score variance is explained by hours studied, "
            "so more study time is associated with higher scores."
        )
        return SpeechObservation(
            transcript=transcript,
            asr_conf=0.95,
            wpm=128.0,
            fillers_per_min=1.0,
        )


class DemoRetrievalAgent:
    """Return a retrieval result with healthy coverage and a single citation."""

    def retrieve(
        self,
        bundle: TurnBundle,
        speech: SpeechObservation,
        grounding: GroundingObservation | None,
    ) -> RetrievalResult:
        hit = RetrievalHit(
            id="rag#12",
            span="lines 10-28",
            snippet="R^2 is the proportion of variance explained by the predictors.",
        )
        return RetrievalResult(hits=[hit], rag_cov=0.83, conflicts=[])


class DemoCoachingAgent(DefaultCoachingAgent):
    """Use the default coaching logic but show a friendlier action label."""

    def coach(self, bundle: TurnBundle, scoring, retrieval):
        action = super().coach(bundle, scoring, retrieval)
        # Rename the affirm action to make the demo output more readable.
        if action.type == "affirm":
            return NextAction(
                type="coach_affirm",
                payload=action.payload,
                rag_refs=action.rag_refs,
            )
        return action


@dataclass
class DemoAgents:
    safety: DefaultSafetyAgent
    vision: DemoVisionAgent
    grounding: DemoGroundingAgent
    speech: DemoSpeechAgent
    retrieval: DemoRetrievalAgent
    scoring: RuleBasedScoringAgent
    coaching: DemoCoachingAgent


def build_orchestrator() -> Orchestrator:
    agents = DemoAgents(
        safety=DefaultSafetyAgent(),
        vision=DemoVisionAgent(),
        grounding=DemoGroundingAgent(),
        speech=DemoSpeechAgent(),
        retrieval=DemoRetrievalAgent(),
        scoring=RuleBasedScoringAgent(),
        coaching=DemoCoachingAgent(),
    )
    return Orchestrator(**agents.__dict__)


def main() -> None:
    orchestrator = build_orchestrator()
    bundle = TurnBundle(
        item=ItemContext(
            item_id="slr_r2_demo",
            stem="Interpret R^2 = 0.64 for predicting exam scores from hours studied.",
        )
    )

    result = orchestrator.step(bundle)

    print("=== Quickstart Demo ===")
    print(f"State: {result.state.name}")
    if result.action:
        print(f"Action Type: {result.action.type}")
        print(f"Payload: {result.action.payload}")
        if result.action.rag_refs:
            print(f"RAG References: {result.action.rag_refs}")
    if result.ui_notes:
        print("UI Notes:")
        for key, value in result.ui_notes.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
