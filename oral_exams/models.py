"""Shared data models for the OralExams on-device tutoring stack."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, List, Optional, Sequence


class OrchestratorState(Enum):
    """Finite state machine for the orchestrator lifecycle."""

    INIT = auto()
    CALIBRATE_CAM = auto()
    CALIBRATE_MIC = auto()
    ITEM_READY = auto()
    CAPTURE = auto()
    RETRIEVE = auto()
    SCORE = auto()
    COACH = auto()
    ITEM_DONE = auto()
    RETRY = auto()
    SESSION_END = auto()


@dataclass
class VisionROI:
    bbox: Sequence[float]
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    pen_like: Optional[bool] = None
    type: Optional[str] = None


@dataclass
class VisionActivity:
    writing: bool
    pointing: bool
    reading: bool


@dataclass
class VisionObservation:
    timestamp: str
    cv_conf: float
    lighting: str
    multiperson: bool
    face: Optional[VisionROI] = None
    board: List[VisionROI] = field(default_factory=list)
    hands: List[VisionROI] = field(default_factory=list)
    activity: Optional[VisionActivity] = None


@dataclass
class GroundingObservation:
    vlm_conf: float
    grounding: List[dict]
    math_fragments: List[str]
    engagement: Optional[str] = None


@dataclass
class SpeechObservation:
    transcript: str
    asr_conf: float
    wpm: float
    fillers_per_min: float


@dataclass
class RetrievalHit:
    id: str
    span: str
    snippet: Optional[str] = None


@dataclass
class RetrievalResult:
    hits: List[RetrievalHit]
    rag_cov: float
    conflicts: List[str] = field(default_factory=list)


@dataclass
class VoiceScore:
    clarity: int
    terminology: int
    coherence: int
    pace: str
    fillers_per_min: float
    coach_notes: List[str] = field(default_factory=list)


@dataclass
class ContentScore:
    content_score: int
    evidence: List[str]
    misconceptions: List[str]
    assumption_flags: List[str]
    unit_check: dict


@dataclass
class NextAction:
    type: str
    payload: str
    rag_refs: List[str] = field(default_factory=list)


@dataclass
class ScoringResult:
    content: ContentScore
    voice: VoiceScore
    next_action: NextAction
    ui_notes: dict


@dataclass
class ItemContext:
    item_id: str
    stem: str


@dataclass
class TurnBundle:
    item: ItemContext
    raw_video_frame: Any | None = None
    raw_video_path: Optional[Path] = None
    raw_audio_path: Optional[Path] = None
    speech: Optional[SpeechObservation] = None
    vision: Optional[VisionObservation] = None
    grounding: Optional[GroundingObservation] = None
    retrieval: Optional[RetrievalResult] = None


@dataclass
class OrchestratorOutput:
    state: OrchestratorState
    action: Optional[NextAction]
    ui_notes: dict


class SafetySignal(Enum):
    SAFE_OK = auto()
    SAFE_STOP = auto()


@dataclass
class SafetyResult:
    signal: SafetySignal
    remediation: List[str] = field(default_factory=list)

