"""Orchestrator for the on-device tutoring stack."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import (
    GroundingObservation,
    OrchestratorOutput,
    OrchestratorState,
    SafetySignal,
    ScoringResult,
    SpeechObservation,
    TurnBundle,
    VisionObservation,
)
from .agents.safety import SafetyAgent
from .agents.vision import VisionAgent
from .agents.grounding import GroundingAgent
from .agents.speech import SpeechAgent
from .agents.retrieval import RetrievalAgent
from .agents.scoring import ScoringAgent
from .agents.coaching import CoachingAgent


@dataclass
class OrchestratorConfig:
    """Configuration parameters for orchestrator behavior."""

    min_cv_conf: float = 0.60
    min_vlm_conf: float = 0.60
    min_rag_cov: float = 0.70


class Orchestrator:
    """Coordinates agent execution according to the design spec."""

    def __init__(
        self,
        safety: SafetyAgent,
        vision: VisionAgent,
        grounding: GroundingAgent,
        speech: SpeechAgent,
        retrieval: RetrievalAgent,
        scoring: ScoringAgent,
        coaching: CoachingAgent,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        self.state = OrchestratorState.INIT
        self.safety = safety
        self.vision = vision
        self.grounding = grounding
        self.speech = speech
        self.retrieval = retrieval
        self.scoring = scoring
        self.coaching = coaching
        self.config = config or OrchestratorConfig()

    def step(self, bundle: TurnBundle) -> OrchestratorOutput:
        """Run a single iteration of the event loop."""

        safety_result = self.safety.check(bundle)
        if safety_result.signal is SafetySignal.SAFE_STOP:
            self.state = OrchestratorState.RETRY
            return OrchestratorOutput(
                state=self.state,
                action=None,
                ui_notes={"remediation": safety_result.remediation},
            )

        self.state = OrchestratorState.CAPTURE
        vision_obs: Optional[VisionObservation] = self.vision.observe(bundle)
        grounding_obs: Optional[GroundingObservation] = None

        if vision_obs and vision_obs.cv_conf >= self.config.min_cv_conf:
            grounding_obs = self.grounding.ground(bundle, vision_obs)
            if grounding_obs and grounding_obs.vlm_conf < self.config.min_vlm_conf:
                grounding_obs = None

        self.state = OrchestratorState.RETRIEVE
        speech_obs: SpeechObservation = self.speech.transcribe(bundle)
        retrieval_result = self.retrieval.retrieve(bundle, speech_obs, grounding_obs)

        if retrieval_result and retrieval_result.rag_cov < self.config.min_rag_cov:
            next_action = self.coaching.handle_low_coverage(bundle, retrieval_result)
            self.state = OrchestratorState.COACH
            return OrchestratorOutput(
                state=self.state,
                action=next_action,
                ui_notes={},
            )

        self.state = OrchestratorState.SCORE
        scoring_result: ScoringResult = self.scoring.score(
            bundle, speech_obs, vision_obs, grounding_obs, retrieval_result
        )

        self.state = OrchestratorState.COACH
        action = self.coaching.coach(bundle, scoring_result, retrieval_result)
        ui_notes = dict(scoring_result.ui_notes)
        return OrchestratorOutput(state=self.state, action=action, ui_notes=ui_notes)

