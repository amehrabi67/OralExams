"""Run a live turn through the orchestrator using local agents."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import cv2
import sounddevice as sd
import soundfile as sf

from oral_exams.agents.coaching import DefaultCoachingAgent
from oral_exams.agents.grounding import NullGroundingAgent, RapidOCRGroundingAgent
from oral_exams.agents.retrieval import LocalRetrievalAgent
from oral_exams.agents.safety import LocalSafetyAgent, LocalSafetyConfig
from oral_exams.agents.scoring import RuleBasedScoringAgent
from oral_exams.agents.speech import VoskSpeechAgent
from oral_exams.agents.vision import OpenCVVisionAgent
from oral_exams.models import ItemContext, TurnBundle
from oral_exams.orchestrator import Orchestrator


def capture_frame(camera_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to access the camera. Double-check permissions and index.")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture a frame from the camera.")
    return frame


def record_audio(duration: float, sample_rate: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    sf.write(output_path, recording, sample_rate)


def build_grounding_agent(enable_ocr: bool):
    if enable_ocr:
        try:
            return RapidOCRGroundingAgent()
        except ImportError:
            print("rapidocr-onnxruntime not installed. Falling back to text-only mode.")
    return NullGroundingAgent()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a live OralExams turn.")
    parser.add_argument("stem", help="Prompt to pose during the turn.")
    parser.add_argument("--item-id", default="demo_r2_01", help="Item identifier for bookkeeping.")
    parser.add_argument("--duration", type=float, default=12.0, help="Audio capture duration in seconds.")
    parser.add_argument("--sample-rate", type=int, default=16_000, help="Audio sample rate for recording.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index to capture a frame from.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./live_turn_artifacts"),
        help="Directory for storing captured audio clips.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="Optional Hugging Face token used when downloading ASR/OCR models.",
    )
    parser.add_argument(
        "--enable-ocr",
        action="store_true",
        help="Enable RapidOCR-based math fragment grounding (requires rapidocr-onnxruntime).",
    )
    parser.add_argument(
        "--min-mean-luma",
        type=float,
        default=40.0,
        help="Minimum average frame brightness required by the safety agent before proceeding.",
    )
    parser.add_argument(
        "--dim-threshold",
        type=float,
        default=40.0,
        help="Luma threshold where the vision agent treats the frame as dim for confidence scoring.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Capturing video frame…")
    frame = capture_frame(args.camera_index)

    audio_path = args.output_dir / "turn_audio.wav"
    print(f"Recording {args.duration}s of audio to {audio_path}…")
    record_audio(args.duration, args.sample_rate, audio_path)

    bundle = TurnBundle(
        item=ItemContext(item_id=args.item_id, stem=args.stem),
        raw_video_frame=frame,
        raw_audio_path=audio_path,
    )

    speech_agent = VoskSpeechAgent(hf_token=args.hf_token)
    orchestrator = Orchestrator(
        safety=LocalSafetyAgent(config=LocalSafetyConfig(min_mean_luma=args.min_mean_luma)),
        vision=OpenCVVisionAgent(dim_threshold=args.dim_threshold),
        grounding=build_grounding_agent(args.enable_ocr),
        speech=speech_agent,
        retrieval=LocalRetrievalAgent(),
        scoring=RuleBasedScoringAgent(),
        coaching=DefaultCoachingAgent(),
    )

    print("Running orchestrator…")
    result = orchestrator.step(bundle)

    print("=== Orchestrator Output ===")
    print(f"State: {result.state.name}")
    if result.action:
        print(f"Action Type: {result.action.type}")
        print(f"Payload: {result.action.payload}")
        if result.action.rag_refs:
            print(f"RAG Refs: {result.action.rag_refs}")
    if result.ui_notes:
        print("UI Notes:")
        for key, value in result.ui_notes.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
