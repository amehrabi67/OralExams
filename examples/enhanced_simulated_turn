"""Enhanced simulated turn with question management support."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the repository root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import soundfile as sf

from oral_exams.question_manager import QuestionManager, Question, ExpectedResponse, DifficultyLevel, QuestionType
from oral_exams.agents.coaching import EnhancedCoachingAgent
from oral_exams.agents.grounding import NullGroundingAgent
from oral_exams.agents.retrieval import LocalRetrievalAgent
from oral_exams.agents.safety import LocalSafetyAgent
from oral_exams.agents.scoring import EnhancedScoringAgent
from oral_exams.agents.vision import OpenCVVisionAgent
from oral_exams.models import ItemContext, TurnBundle, SpeechObservation
from oral_exams.orchestrator import Orchestrator


class StubSpeechAgent:
    """Stub speech agent for simulation."""
    
    def __init__(self, transcript: str = "R squared equals 0.64 for this model."):
        self._transcript = transcript

    def transcribe(self, bundle: TurnBundle) -> SpeechObservation:
        """Produce a plausible SpeechObservation."""
        return SpeechObservation(
            transcript=self._transcript, 
            asr_conf=0.95, 
            wpm=120.0, 
            fillers_per_min=0.0
        )


def make_blank_frame(width: int = 640, height: int = 480):
    """Create a blank frame with sufficient lighting."""
    # BGR image with sufficient lighting (gray instead of black)
    return np.full((height, width, 3), 128, dtype=np.uint8)


def make_silence_wav(path: Path, duration_s: float = 2.0, sr: int = 16000):
    """Create a silent audio file."""
    samples = int(duration_s * sr)
    data = np.zeros((samples, 1), dtype="float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, sr)


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
        ),
        Question(
            question_id="correlation_vs_regression_01",
            stem="Explain the difference between correlation and regression.",
            context="A student is confused about when to use correlation vs regression analysis.",
            expected_response=ExpectedResponse(
                key_concepts=["correlation measures association", "regression predicts values", "causation vs association"],
                required_terms=["correlation", "regression", "association", "prediction", "causation"],
                optional_terms=["direction", "strength", "causal relationship"],
                example_phrases=[
                    "Correlation measures the strength and direction of association between variables",
                    "Regression is used to predict the value of one variable from another",
                    "Correlation doesn't imply causation, but regression can help establish causal relationships"
                ],
                common_misconceptions=[
                    "Correlation and regression are the same thing",
                    "High correlation means causation",
                    "Regression always implies causation"
                ],
                scoring_criteria={
                    "distinguishes_correlation": 3,
                    "distinguishes_regression": 3,
                    "mentions_causation": 2,
                    "provides_examples": 2
                },
                difficulty=DifficultyLevel.BEGINNER,
                question_type=QuestionType.CONCEPTUAL
            )
        )
    ]
    
    for question in sample_questions:
        question_manager.add_question(question)


def run_simulation(question_id: str, student_response: str, questions_file: Path = None) -> None:
    """Run a simulation with a specific question and student response."""
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
    
    print(f"Running simulation for question: {question_id}")
    print(f"Question: {question.stem}")
    print(f"Student response: {student_response}")
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
    
    # Create orchestrator with enhanced agents
    orchestrator = Orchestrator(
        safety=LocalSafetyAgent(),
        vision=OpenCVVisionAgent(),
        grounding=NullGroundingAgent(),
        speech=StubSpeechAgent(student_response),
        retrieval=LocalRetrievalAgent(),
        scoring=EnhancedScoringAgent(question_manager),
        coaching=EnhancedCoachingAgent(question_manager),
    )
    
    # Run the simulation
    result = orchestrator.step(bundle)
    
    # Display results
    print("=== Simulation Results ===")
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


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enhanced Simulated Turn")
    parser.add_argument("--question-id", default="r2_interpretation_01", 
                       help="Question ID to run")
    parser.add_argument("--response", 
                       default="R squared of 0.64 means 64% of the variance in exam scores is explained by study hours",
                       help="Student response to simulate")
    parser.add_argument("--questions-file", type=Path, 
                       help="Path to questions JSON file")
    parser.add_argument("--list-questions", action="store_true",
                       help="List available questions")
    
    args = parser.parse_args()
    
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
    
    run_simulation(args.question_id, args.response, args.questions_file)


if __name__ == "__main__":
    main()
