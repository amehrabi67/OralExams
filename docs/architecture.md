# OralExams On-Device Tutor Architecture

This document distills the user specification into implementable software
components. The goal is to keep the tutor fully on-device while orchestrating a
cohort of lightweight agents responsible for safety, sensing, retrieval, scoring
and coaching.

## State Machine

The orchestrator follows the finite state machine below:

```
INIT → CALIBRATE_CAM/MIC → ITEM_READY → CAPTURE {Vision tick ↔ ASR} →
RETRIEVE → SCORE → COACH → ITEM_DONE | RETRY → SESSION_END
```

The `Orchestrator` class in `oral_exams/orchestrator.py` encodes the critical
transitions required to move from sensing to coaching while honouring the
confidence gates defined in the spec:

* visual actions require `cv_conf ≥ 0.60` and `vlm_conf ≥ 0.60`;
* content explanations require `rag_cov ≥ 0.70`.

If the RAG coverage is insufficient the orchestrator routes to
`handle_low_coverage` which surfaces prerequisite guidance instead of the main
coaching step.

## Agents

The software is organised around agents living under `oral_exams/agents/`:

| Agent | Responsibility | Key Notes |
| --- | --- | --- |
| `SafetyAgent` | Enforce offline, single-learner capture, light/occlusion gates. | Returns remediation cues before the loop proceeds. |
| `VisionAgent` | OpenCV pipeline for face/board/hand localisation and activity heuristics. | Passes smoothed ROIs to grounding. |
| `GroundingAgent` | FastVLM-based semantic confirmation of math regions and fragments. | Only trusted if `vlm_conf ≥ 0.60`. |
| `SpeechAgent` | Whisper/Vosk ASR with VAD-driven streaming. | Supplies transcript metrics (WPM, fillers). |
| `RetrievalAgent` | Local FAISS index query builder computing RAG coverage. | Provides doc spans for downstream citing. |
| `ScoringAgent` | Rubric-based grading and misconception checks anchored to RAG results. | Combines rule-based checks and OpenELM reasoning. |
| `CoachingAgent` | Condenses scoring + voice feedback into a single actionable coaching step. | Defers to prerequisite prompt when coverage is low. |

Protocol interfaces isolate the orchestrator from specific model choices, making
it straightforward to substitute real implementations for the included stub
agents during bring-up.

## Data Model

`oral_exams/models.py` contains dataclasses that implement the bus schema from
the specification, including:

* `VisionObservation` with ROI metadata, activities and confidence scores.
* `GroundingObservation` capturing math fragments and engagement signals.
* `SpeechObservation` with ASR confidence and fluency metrics.
* `RetrievalResult` with FAISS hits and coverage ratio.
* `ScoringResult` and `NextAction` bundling the rubric outcome and the single
  coaching move.
* `SafetyResult` to express gating decisions made before the loop proceeds.

The orchestrator step consumes a `TurnBundle` that aggregates all modalities for
the current prompt attempt.

## Extension Points

* Replace `DummyVisionAgent` with the OpenCV pipeline described in the spec,
  feeding smoothed ROIs and writing cues.
* Swap `DummyGroundingAgent` for a FastVLM client that accepts ROI crops and
  returns grounded math fragments.
* Implement a real `SpeechAgent` using Whisper.cpp or Vosk streaming APIs.
* Implement `RetrievalAgent` on top of FAISS or another local vector index.
* Provide a concrete `ScoringAgent` leveraging OpenELM for rubric scoring and
  rule-based checks for slope units, R² interpretation, and assumptions.
* Extend `DefaultCoachingAgent` to trigger confusion proxies and targeted
  example playback when rule-based cues align.

All high-risk modules should respect the 10-second rolling buffers and call the
safety agent to zeroise memory when the session ends.

