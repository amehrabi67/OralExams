"""Enhanced live session with question management support."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure the repository root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import sounddevice as sd
import soundfile as sf

from oral_exams.question_manager import QuestionManager, Question, ExpectedResponse, DifficultyLevel, QuestionType
from oral_exams.agents.coaching import EnhancedCoachingAgent
from oral_exams.agents.grounding import NullGroundingAgent, RapidOCRGroundingAgent
from oral_exams.agents.retrieval import LocalRetrievalAgent
from oral_exams.agents.safety import LocalSafetyAgent
from oral_exams.agents.scoring import EnhancedScoringAgent
from oral_exams.agents.speech import VoskSpeechAgent
from oral_exams.agents.vision import OpenCVVisionAgent
from oral_exams.models import ItemContext, TurnBundle
from oral_exams.orchestrator import Orchestrator


def capture_frame(camera_index: int) -> np.ndarray:
    """Capture a frame from the camera."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to access the camera. Double-check permissions and index.")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture a frame from the camera.")
    return frame


def record_audio(duration: float, sample_rate: int, output_path: Path) -> None:
    """Record audio for the specified duration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    sf.write(output_path, recording, sample_rate)


def build_grounding_agent(enable_ocr: bool):
    """Build the grounding agent."""
    if enable_ocr:
        try:
            return RapidOCRGroundingAgent()
        except ImportError:
            print("rapidocr-onnxruntime not installed. Falling back to text-only mode.")
    return NullGroundingAgent()


def create_sample_questions(question_manager: QuestionManager) -> None:
    """Create sample questions for demonstration."""
    sample_questions = [
        Question(
            question_id="r2_interpretation_01",
            stem="Interpret R² = 0.64 for predicting exam score from hours studied.",
            context="A student has calculated R² = 0.64 for a linear regression model predicting exam scores from study hours.",
            expected_response=ExpectedResponse(
                key_concepts=["coefficient of determination", "variance explained", "model fit"],
                required_terms=["R-squared", "64%", "variance", "explained"],
                optional_terms=["correlation", "linear relationship", "prediction"],
                example_phrases=[
                    "R-squared of 0.64 means 64% of the variance in exam scores is explained by study hours",
                    "The model explains 64% of the variation in exam scores",
                    "64% of the differences in exam scores can be attributed to study hours"
                ],
                common_misconceptions=[
                    "R² is the correlation coefficient",
                    "R² of 0.64 means 64% accuracy",
                    "R² tells us the strength of the relationship"
                ],
                scoring_criteria={
                    "mentions_percentage": 2,
                    "mentions_variance": 2,
                    "mentions_explanation": 2,
                    "avoids_misconceptions": 1
                },
                difficulty=DifficultyLevel.INTERMEDIATE,
                question_type=QuestionType.INTERPRETATION
            )
        ),
        Question(
            question_id="regression_assumptions_01",
            stem="What are the key assumptions of linear regression?",
            context="A student is learning about linear regression and needs to understand the underlying assumptions.",
            expected_response=ExpectedResponse(
                key_concepts=["linearity", "independence", "homoscedasticity", "normality"],
                required_terms=["linear relationship", "independent observations", "constant variance", "normal distribution"],
                optional_terms=["residuals", "outliers", "multicollinearity"],
                example_phrases=[
                    "The relationship between variables should be linear",
                    "Observations should be independent of each other",
                    "The variance of residuals should be constant",
                    "Residuals should be normally distributed"
                ],
                common_misconceptions=[
                    "All variables must be normally distributed",
                    "The dependent variable must be continuous",
                    "More data always means better results"
                ],
                scoring_criteria={
                    "mentions_linearity": 2,
                    "mentions_independence": 2,
                    "mentions_homoscedasticity": 2,
                    "mentions_normality": 2,
                    "completeness": 1
                },
                difficulty=DifficultyLevel.INTERMEDIATE,
                question_type=QuestionType.CONCEPTUAL
            )
        )
    ]
    
    for question in sample_questions:
        question_manager.add_question(question)


def run_live_session(question_id: str, duration: float, camera_index: int, 
                    enable_ocr: bool, hf_token: str, questions_file: Path = None) -> None:
    """Run a live session with the specified question."""
    # Initialize question manager
    question_manager = QuestionManager(questions_file)
    
    # Create sample questions if none exist
    if not question_manager.list_questions():
        print("Creating sample questions...")
        create_sample_questions(question_manager)
    
    # Get the question
    question = question_manager.get_question(question_id)
    if not question:
        print(f"Question '{question_id}' not found!")
        print("Available questions:")
        for q in question_manager.list_questions():
            print(f"  - {q.question_id}: {q.stem}")
        return
    
    print(f"Running live session for question: {question_id}")
    print(f"Question: {question.stem}")
    if question.context:
        print(f"Context: {question.context}")
    print()
    
    # Capture video frame
    print("Capturing video frame...")
    try:
        frame = capture_frame(camera_index)
    except RuntimeError as e:
        print(f"Camera error: {e}")
        print("Falling back to simulation mode...")
        run_simulation_mode(question, duration, hf_token, question_manager)
        return
    
    # Record audio
    audio_path = Path("./live_turn_artifacts/turn_audio.wav")
    print(f"Recording {duration}s of audio to {audio_path}...")
    record_audio(duration, 16000, audio_path)
    
    # Create bundle
    bundle = TurnBundle(
        item=ItemContext(item_id=question.question_id, stem=question.stem),
        raw_video_frame=frame,
        raw_audio_path=audio_path,
    )
    
    # Create orchestrator with enhanced agents
    speech_agent = VoskSpeechAgent(hf_token=hf_token)
    orchestrator = Orchestrator(
        safety=LocalSafetyAgent(),
        vision=OpenCVVisionAgent(),
        grounding=build_grounding_agent(enable_ocr),
        speech=speech_agent,
        retrieval=LocalRetrievalAgent(),
        scoring=EnhancedScoringAgent(question_manager),
        coaching=EnhancedCoachingAgent(question_manager),
    )
    
    # Run orchestrator
    print("Running orchestrator...")
    result = orchestrator.step(bundle)
    
    # Display results
    display_results(result, question)


def run_simulation_mode(question: Question, duration: float, hf_token: str, 
                       question_manager: QuestionManager) -> None:
    """Run in simulation mode when camera is not available."""
    from .enhanced_simulated_turn import StubSpeechAgent, make_blank_frame, make_silence_wav
    
    # Create a simulated response based on the question
    if question.expected_response and question.expected_response.example_phrases:
        simulated_response = question.expected_response.example_phrases[0]
    else:
        simulated_response = "This is a simulated response for the question."
    
    print(f"Simulated student response: {simulated_response}")
    print()
    
    # Create artifacts
    out_dir = Path("./sim_turn_artifacts")
    audio_path = out_dir / "silent.wav"
    make_silence_wav(audio_path)
    
    frame = make_blank_frame()
    bundle = TurnBundle(
        item=ItemContext(item_id=question.question_id, stem=question.stem),
        raw_video_frame=frame,
        raw_audio_path=audio_path
    )
    
    # Create orchestrator with stub speech agent
    orchestrator = Orchestrator(
        safety=LocalSafetyAgent(),
        vision=OpenCVVisionAgent(),
        grounding=NullGroundingAgent(),
        speech=StubSpeechAgent(simulated_response),
        retrieval=LocalRetrievalAgent(),
        scoring=EnhancedScoringAgent(question_manager),
        coaching=EnhancedCoachingAgent(question_manager),
    )
    
    # Run orchestrator
    print("Running orchestrator...")
    result = orchestrator.step(bundle)
    
    # Display results
    display_results(result, question)


def display_results(result, question: Question) -> None:
    """Display the orchestrator results."""
    print("\n=== Tutoring Results ===")
    print(f"State: {result.state.name}")
    
    if result.action:
        print(f"Action Type: {result.action.type}")
        print(f"Feedback: {result.action.payload}")
        if result.action.rag_refs:
            print(f"References: {', '.join(result.action.rag_refs)}")
    
    if result.ui_notes:
        print("\nDetailed Analysis:")
        for key, value in result.ui_notes.items():
            if key == "response_analysis" and isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
    
    # Show question context if available
    if question.expected_response:
        print(f"\nQuestion Context:")
        print(f"  Difficulty: {question.expected_response.difficulty.value}")
        print(f"  Type: {question.expected_response.question_type.value}")
        print(f"  Key Concepts: {', '.join(question.expected_response.key_concepts)}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced Live Session")
    parser.add_argument("question_id", help="Question ID to run")
    parser.add_argument("--duration", type=float, default=12.0, 
                       help="Audio capture duration in seconds")
    parser.add_argument("--camera-index", type=int, default=0, 
                       help="OpenCV camera index")
    parser.add_argument("--enable-ocr", action="store_true", 
                       help="Enable RapidOCR-based math fragment grounding")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), 
                       help="Hugging Face token for model downloads")
    parser.add_argument("--questions-file", type=Path, 
                       help="Path to questions JSON file")
    parser.add_argument("--list-questions", action="store_true",
                       help="List available questions")
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()
    
    if args.list_questions:
        question_manager = QuestionManager(args.questions_file)
        if not question_manager.list_questions():
            create_sample_questions(question_manager)
        
        print("Available questions:")
        for question in question_manager.list_questions():
            print(f"  - {question.question_id}: {question.stem}")
            if question.expected_response:
                print(f"    Difficulty: {question.expected_response.difficulty.value}")
                print(f"    Type: {question.expected_response.question_type.value}")
        return
    
    run_live_session(
        question_id=args.question_id,
        duration=args.duration,
        camera_index=args.camera_index,
        enable_ocr=args.enable_ocr,
        hf_token=args.hf_token,
        questions_file=args.questions_file
    )


if __name__ == "__main__":
    main()
