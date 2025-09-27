"""Visual grounding agent placeholder."""
from __future__ import annotations

import re
from typing import Protocol

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:  # pragma: no cover - optional dependency
    RapidOCR = None

from ..models import GroundingObservation, TurnBundle, VisionObservation


class GroundingAgent(Protocol):
    def ground(
        self, bundle: TurnBundle, vision: VisionObservation
    ) -> GroundingObservation | None:
        ...


class RapidOCRGroundingAgent:
    """Use RapidOCR to spot math fragments on the captured frame."""

    def __init__(self, device_id: int = 0) -> None:
        if RapidOCR is None:
            raise ImportError(
                "rapidocr-onnxruntime is required for RapidOCRGroundingAgent."
            )
        self._ocr = RapidOCR(device_id=device_id)

    def ground(
        self, bundle: TurnBundle, vision: VisionObservation
    ) -> GroundingObservation | None:
        frame = bundle.raw_video_frame
        if frame is None:
            return None

        ocr_result, _ = self._ocr(frame[:, :, ::-1])
        if not ocr_result:
            return GroundingObservation(vlm_conf=0.4, grounding=[], math_fragments=[])

        fragments: list[str] = []
        grounding_rows: list[dict] = []
        confidences = []

        for points, text, score in ocr_result:
            bbox = [float(coord) for point in points for coord in point]
            grounding_rows.append({"region": "board", "tag": text, "bbox": bbox})
            confidences.append(float(score))
            fragments.extend(_extract_math_fragments(text))

        vlm_conf = min(max(sum(confidences) / (len(confidences) or 1), 0.0), 1.0)
        if fragments:
            vlm_conf = max(vlm_conf, 0.65)

        unique_fragments = sorted(set(fragments))
        return GroundingObservation(
            vlm_conf=vlm_conf,
            grounding=grounding_rows,
            math_fragments=unique_fragments,
        )


_MATH_REGEXES = [
    re.compile(r"R\s*\^?2\s*=\s*[-+]?\d+(?:\.\d+)?"),
    re.compile(r"y\s*=\s*[^\s]+"),
    re.compile(r"β\d"),
]


def _extract_math_fragments(text: str) -> list[str]:
    matches: list[str] = []
    for pattern in _MATH_REGEXES:
        matches.extend(match.group(0) for match in pattern.finditer(text))
    return matches


class NullGroundingAgent:
    """Fallback grounding agent when OCR models are unavailable."""

    def ground(
        self, bundle: TurnBundle, vision: VisionObservation
    ) -> GroundingObservation | None:
        return GroundingObservation(vlm_conf=0.0, grounding=[], math_fragments=[])

