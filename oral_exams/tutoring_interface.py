"""User-friendly interface for the tutoring system."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from .question_manager import QuestionManager, Question, ExpectedResponse, DifficultyLevel, QuestionType
from .orchestrator import Orchestrator
from .agents.safety import LocalSafetyAgent
from .agents.vision import OpenCVVisionAgent
from .agents.grounding import NullGroundingAgent, RapidOCRGroundingAgent
from .agents.speech import VoskSpeechAgent
from .agents.retrieval import LocalRetrievalAgent
from .agents.scoring import EnhancedScoringAgent
from .agents.coaching import EnhancedCoachingAgent
from .models import ItemContext, TurnBundle


class TutoringInterface:
    """Main interface for the tutoring system."""
    
    def __init__(self, questions_file: Optional[Path] = None):
        self.question_manager = QuestionManager(questions_file)
        self.orchestrator = None
        self._setup_orchestrator()
    
    def _setup_orchestrator(self) -> None:
        """Set up the orchestrator with enhanced agents."""
        self.orchestrator = Orchestrator(
            safety=LocalSafetyAgent(),
            vision=OpenCVVisionAgent(),
            grounding=NullGroundingAgent(),
            speech=VoskSpeechAgent(),
            retrieval=LocalRetrievalAgent(),
            scoring=EnhancedScoringAgent(self.question_manager),
            coaching=EnhancedCoachingAgent(self.question_manager),
        )
    
    def add_question_interactive(self) -> None:
        """Interactive interface for adding questions."""
        print("\n=== Add New Question ===")
        
        question_id = input("Question ID: ").strip()
        if not question_id:
            print("Question ID is required!")
            return
        
        if self.question_manager.get_question(question_id):
            print(f"Question with ID '{question_id}' already exists!")
            return
        
        stem = input("Question stem: ").strip()
        if not stem:
            print("Question stem is required!")
            return
        
        context = input("Context (optional): ").strip() or None
        
        print("\n=== Expected Response ===")
        key_concepts = input("Key concepts (comma-separated): ").strip().split(",")
        key_concepts = [concept.strip() for concept in key_concepts if concept.strip()]
        
        required_terms = input("Required terms (comma-separated): ").strip().split(",")
        required_terms = [term.strip() for term in required_terms if term.strip()]
        
        optional_terms = input("Optional terms (comma-separated): ").strip().split(",")
        optional_terms = [term.strip() for term in optional_terms if term.strip()]
        
        example_phrases = input("Example phrases (comma-separated): ").strip().split(",")
        example_phrases = [phrase.strip() for phrase in example_phrases if phrase.strip()]
        
        misconceptions = input("Common misconceptions (comma-separated): ").strip().split(",")
        misconceptions = [misconception.strip() for misconception in misconceptions if misconception.strip()]
        
        print("\n=== Scoring Criteria ===")
        scoring_criteria = {}
        while True:
            criterion = input("Criterion name (or 'done' to finish): ").strip()
            if criterion.lower() == 'done':
                break
            try:
                points = int(input(f"Points for '{criterion}': "))
                scoring_criteria[criterion] = points
            except ValueError:
                print("Please enter a valid number!")
                continue
        
        print("\n=== Question Metadata ===")
        print("Difficulty levels: beginner, intermediate, advanced")
        difficulty_input = input("Difficulty: ").strip().lower()
        try:
            difficulty = DifficultyLevel(difficulty_input)
        except ValueError:
            difficulty = DifficultyLevel.BEGINNER
            print(f"Invalid difficulty, defaulting to {difficulty.value}")
        
        print("Question types: conceptual, calculation, interpretation, application")
        question_type_input = input("Question type: ").strip().lower()
        try:
            question_type = QuestionType(question_type_input)
        except ValueError:
            question_type = QuestionType.CONCEPTUAL
            print(f"Invalid question type, defaulting to {question_type.value}")
        
        # Create expected response
        expected_response = ExpectedResponse(
            key_concepts=key_concepts,
            required_terms=required_terms,
            optional_terms=optional_terms,
            example_phrases=example_phrases,
            common_misconceptions=misconceptions,
            scoring_criteria=scoring_criteria,
            difficulty=difficulty,
            question_type=question_type
        )
        
        # Create question
        question = Question(
            question_id=question_id,
            stem=stem,
            context=context,
            expected_response=expected_response
        )
        
        # Add to manager
        self.question_manager.add_question(question)
        print(f"\n✅ Question '{question_id}' added successfully!")
    
    def list_questions(self) -> None:
        """List all available questions."""
        questions = self.question_manager.list_questions()
        if not questions:
            print("No questions available.")
            return
        
        print(f"\n=== Available Questions ({len(questions)}) ===")
        for i, question in enumerate(questions, 1):
            print(f"{i}. {question.question_id}")
            print(f"   Stem: {question.stem}")
            if question.expected_response:
                print(f"   Difficulty: {question.expected_response.difficulty.value}")
                print(f"   Type: {question.expected_response.question_type.value}")
            print()
    
    def search_questions(self) -> None:
        """Search questions by criteria."""
        print("\n=== Search Questions ===")
        topic = input("Topic (optional): ").strip() or None
        difficulty_input = input("Difficulty (beginner/intermediate/advanced, optional): ").strip().lower() or None
        question_type_input = input("Question type (conceptual/calculation/interpretation/application, optional): ").strip().lower() or None
        
        difficulty = None
        if difficulty_input:
            try:
                difficulty = DifficultyLevel(difficulty_input)
            except ValueError:
                print(f"Invalid difficulty: {difficulty_input}")
                return
        
        question_type = None
        if question_type_input:
            try:
                question_type = QuestionType(question_type_input)
            except ValueError:
                print(f"Invalid question type: {question_type_input}")
                return
        
        results = self.question_manager.search_questions(topic, difficulty, question_type)
        
        if not results:
            print("No questions found matching your criteria.")
            return
        
        print(f"\n=== Search Results ({len(results)}) ===")
        for i, question in enumerate(results, 1):
            print(f"{i}. {question.question_id}")
            print(f"   Stem: {question.stem}")
            if question.expected_response:
                print(f"   Difficulty: {question.expected_response.difficulty.value}")
                print(f"   Type: {question.expected_response.question_type.value}")
            print()
    
    def run_question(self, question_id: str, use_simulation: bool = False) -> None:
        """Run a specific question."""
        question = self.question_manager.get_question(question_id)
        if not question:
            print(f"Question '{question_id}' not found!")
            return
        
        print(f"\n=== Running Question: {question_id} ===")
        print(f"Stem: {question.stem}")
        if question.context:
            print(f"Context: {question.context}")
        print()
        
        if use_simulation:
            self._run_simulated_question(question)
        else:
            self._run_live_question(question)
    
    def _run_simulated_question(self, question: Question) -> None:
        """Run a question in simulation mode."""
        from .examples.simulated_turn import make_blank_frame, make_silence_wav, StubSpeechAgent
        
        # Create a simulated response
        if question.expected_response:
            # Use the first example phrase as the simulated response
            simulated_response = question.expected_response.example_phrases[0] if question.expected_response.example_phrases else "This is a simulated response."
        else:
            simulated_response = "This is a simulated response."
        
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
        
        # Create stub speech agent with simulated response
        speech_agent = StubSpeechAgent(simulated_response)
        
        # Temporarily replace speech agent
        original_speech = self.orchestrator.speech
        self.orchestrator.speech = speech_agent
        
        try:
            result = self.orchestrator.step(bundle)
            self._display_result(result)
        finally:
            # Restore original speech agent
            self.orchestrator.speech = original_speech
    
    def _run_live_question(self, question: Question) -> None:
        """Run a question in live mode."""
        import cv2
        import sounddevice as sd
        import soundfile as sf
        import numpy as np
        
        print("Capturing video frame...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Unable to access camera!")
            return
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Failed to capture frame!")
            return
        
        print("Recording audio (5 seconds)...")
        duration = 5.0
        sample_rate = 16000
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        
        audio_path = Path("./live_turn_artifacts/turn_audio.wav")
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(audio_path, recording, sample_rate)
        
        bundle = TurnBundle(
            item=ItemContext(item_id=question.question_id, stem=question.stem),
            raw_video_frame=frame,
            raw_audio_path=audio_path
        )
        
        result = self.orchestrator.step(bundle)
        self._display_result(result)
    
    def _display_result(self, result) -> None:
        """Display the orchestrator result."""
        print("\n=== Tutoring Result ===")
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
    
    def export_questions(self, filename: str) -> None:
        """Export questions to a JSON file."""
        questions = self.question_manager.list_questions()
        data = {
            'questions': [
                {
                    'question_id': q.question_id,
                    'stem': q.stem,
                    'context': q.context,
                    'expected_response': {
                        'key_concepts': q.expected_response.key_concepts,
                        'required_terms': q.expected_response.required_terms,
                        'optional_terms': q.expected_response.optional_terms,
                        'example_phrases': q.expected_response.example_phrases,
                        'common_misconceptions': q.expected_response.common_misconceptions,
                        'scoring_criteria': q.expected_response.scoring_criteria,
                        'difficulty': q.expected_response.difficulty.value,
                        'question_type': q.expected_response.question_type.value
                    } if q.expected_response else None,
                    'metadata': q.metadata
                }
                for q in questions
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Questions exported to {filename}")
    
    def import_questions(self, filename: str) -> None:
        """Import questions from a JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            imported_count = 0
            for q_data in data.get('questions', []):
                question = self.question_manager._dict_to_question(q_data)
                self.question_manager.add_question(question)
                imported_count += 1
            
            print(f"Imported {imported_count} questions from {filename}")
        except Exception as e:
            print(f"Error importing questions: {e}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Tutoring System Interface")
    parser.add_argument("--questions-file", type=Path, help="Path to questions JSON file")
    parser.add_argument("--mode", choices=["interactive", "run", "list", "search", "export", "import"], 
                       default="interactive", help="Operation mode")
    parser.add_argument("--question-id", help="Question ID for run mode")
    parser.add_argument("--simulation", action="store_true", help="Use simulation mode for run")
    parser.add_argument("--export-file", help="Export filename")
    parser.add_argument("--import-file", help="Import filename")
    
    args = parser.parse_args()
    
    interface = TutoringInterface(args.questions_file)
    
    if args.mode == "interactive":
        while True:
            print("\n=== Tutoring System Interface ===")
            print("1. Add question")
            print("2. List questions")
            print("3. Search questions")
            print("4. Run question")
            print("5. Export questions")
            print("6. Import questions")
            print("7. Exit")
            
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == "1":
                interface.add_question_interactive()
            elif choice == "2":
                interface.list_questions()
            elif choice == "3":
                interface.search_questions()
            elif choice == "4":
                question_id = input("Question ID: ").strip()
                if question_id:
                    simulation = input("Use simulation? (y/n): ").strip().lower() == 'y'
                    interface.run_question(question_id, simulation)
            elif choice == "5":
                filename = input("Export filename: ").strip()
                if filename:
                    interface.export_questions(filename)
            elif choice == "6":
                filename = input("Import filename: ").strip()
                if filename:
                    interface.import_questions(filename)
            elif choice == "7":
                print("Goodbye!")
                break
            else:
                print("Invalid option!")
    
    elif args.mode == "run":
        if not args.question_id:
            print("Question ID is required for run mode!")
            return
        interface.run_question(args.question_id, args.simulation)
    
    elif args.mode == "list":
        interface.list_questions()
    
    elif args.mode == "search":
        interface.search_questions()
    
    elif args.mode == "export":
        if not args.export_file:
            print("Export filename is required!")
            return
        interface.export_questions(args.export_file)
    
    elif args.mode == "import":
        if not args.import_file:
            print("Import filename is required!")
            return
        interface.import_questions(args.import_file)


if __name__ == "__main__":
    main()
