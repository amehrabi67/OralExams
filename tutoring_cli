#!/usr/bin/env python3
"""Comprehensive CLI for the tutoring system."""

import argparse
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from oral_exams.tutoring_interface import TutoringInterface
from oral_exams.config import ConfigManager, create_default_config_file
from oral_exams.question_manager import QuestionManager, DifficultyLevel, QuestionType


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="On-Device Tutoring System for Regression Topics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python tutoring_cli.py

  # Run a specific question
  python tutoring_cli.py run r2_interpretation_01

  # Run with simulation
  python tutoring_cli.py run r2_interpretation_01 --simulation

  # List available questions
  python tutoring_cli.py list

  # Add a new question
  python tutoring_cli.py add

  # Search questions
  python tutoring_cli.py search --topic "regression"

  # Export questions
  python tutoring_cli.py export questions.json

  # Import questions
  python tutoring_cli.py import questions.json

  # Create default config
  python tutoring_cli.py config --create-default

  # Show current config
  python tutoring_cli.py config --show
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a tutoring session')
    run_parser.add_argument('question_id', help='Question ID to run')
    run_parser.add_argument('--simulation', action='store_true', 
                           help='Use simulation mode (no camera/microphone)')
    run_parser.add_argument('--duration', type=float, default=12.0,
                           help='Audio recording duration in seconds')
    run_parser.add_argument('--camera-index', type=int, default=0,
                           help='Camera index to use')
    run_parser.add_argument('--enable-ocr', action='store_true',
                           help='Enable OCR for math detection')
    run_parser.add_argument('--hf-token', help='Hugging Face token')
    run_parser.add_argument('--questions-file', type=Path,
                           help='Path to questions JSON file')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available questions')
    list_parser.add_argument('--questions-file', type=Path,
                            help='Path to questions JSON file')
    
    # Add command
    add_parser = subparsers.add_parser('add', help='Add a new question')
    add_parser.add_argument('--questions-file', type=Path,
                           help='Path to questions JSON file')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search questions')
    search_parser.add_argument('--topic', help='Search by topic')
    search_parser.add_argument('--difficulty', choices=['beginner', 'intermediate', 'advanced'],
                              help='Filter by difficulty')
    search_parser.add_argument('--type', choices=['conceptual', 'calculation', 'interpretation', 'application'],
                              help='Filter by question type')
    search_parser.add_argument('--questions-file', type=Path,
                              help='Path to questions JSON file')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export questions')
    export_parser.add_argument('filename', help='Output filename')
    export_parser.add_argument('--questions-file', type=Path,
                              help='Path to questions JSON file')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import questions')
    import_parser.add_argument('filename', help='Input filename')
    import_parser.add_argument('--questions-file', type=Path,
                              help='Path to questions JSON file')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--create-default', action='store_true',
                              help='Create default configuration file')
    config_parser.add_argument('--show', action='store_true',
                              help='Show current configuration')
    config_parser.add_argument('--reset', action='store_true',
                              help='Reset to default configuration')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive mode')
    interactive_parser.add_argument('--questions-file', type=Path,
                                   help='Path to questions JSON file')
    
    args = parser.parse_args()
    
    if not args.command:
        # No command specified, show help
        parser.print_help()
        return
    
    try:
        if args.command == 'run':
            run_tutoring_session(args)
        elif args.command == 'list':
            list_questions(args)
        elif args.command == 'add':
            add_question(args)
        elif args.command == 'search':
            search_questions(args)
        elif args.command == 'export':
            export_questions(args)
        elif args.command == 'import':
            import_questions(args)
        elif args.command == 'config':
            handle_config(args)
        elif args.command == 'interactive':
            run_interactive_mode(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
        if args.debug if hasattr(args, 'debug') else False:
            import traceback
            traceback.print_exc()


def run_tutoring_session(args):
    """Run a tutoring session."""
    interface = TutoringInterface(args.questions_file)
    
    if args.simulation:
        # Use enhanced simulated turn
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "examples"))
        from enhanced_simulated_turn import run_simulation
        run_simulation(args.question_id, "Simulated response", args.questions_file)
    else:
        # Use enhanced live session
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "examples"))
        from enhanced_live_session import run_live_session
        run_live_session(
            question_id=args.question_id,
            duration=args.duration,
            camera_index=args.camera_index,
            enable_ocr=args.enable_ocr,
            hf_token=args.hf_token,
            questions_file=args.questions_file
        )


def list_questions(args):
    """List available questions."""
    interface = TutoringInterface(args.questions_file)
    interface.list_questions()


def add_question(args):
    """Add a new question."""
    interface = TutoringInterface(args.questions_file)
    interface.add_question_interactive()


def search_questions(args):
    """Search questions."""
    interface = TutoringInterface(args.questions_file)
    
    # Convert string arguments to enums
    difficulty = None
    if args.difficulty:
        difficulty = DifficultyLevel(args.difficulty)
    
    question_type = None
    if args.type:
        question_type = QuestionType(args.type)
    
    # Search questions
    question_manager = QuestionManager(args.questions_file)
    results = question_manager.search_questions(
        topic=args.topic,
        difficulty=difficulty,
        question_type=question_type
    )
    
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


def export_questions(args):
    """Export questions."""
    interface = TutoringInterface(args.questions_file)
    interface.export_questions(args.filename)


def import_questions(args):
    """Import questions."""
    interface = TutoringInterface(args.questions_file)
    interface.import_questions(args.filename)


def handle_config(args):
    """Handle configuration commands."""
    if args.create_default:
        create_default_config_file()
        print("Default configuration file created!")
    elif args.show:
        config_manager = ConfigManager()
        config = config_manager.get_config()
        print("Current configuration:")
        print(f"  Safety - Min Luma: {config.safety.min_mean_luma}")
        print(f"  Vision - Min CV Conf: {config.vision.min_cv_conf}")
        print(f"  Grounding - Min VLM Conf: {config.grounding.min_vlm_conf}")
        print(f"  Speech - Model: {config.speech.model_name}")
        print(f"  Retrieval - Min RAG Cov: {config.retrieval.min_rag_cov}")
        print(f"  Debug Mode: {config.debug_mode}")
    elif args.reset:
        config_manager = ConfigManager()
        config_manager.reset_to_default()
        print("Configuration reset to defaults!")
    else:
        print("Use --create-default, --show, or --reset")


def run_interactive_mode(args):
    """Run interactive mode."""
    interface = TutoringInterface(args.questions_file)
    
    while True:
        print("\n=== Tutoring System Interface ===")
        print("1. Add question")
        print("2. List questions")
        print("3. Search questions")
        print("4. Run question (live)")
        print("5. Run question (simulation)")
        print("6. Export questions")
        print("7. Import questions")
        print("8. Configuration")
        print("9. Exit")
        
        choice = input("\nSelect option (1-9): ").strip()
        
        if choice == "1":
            interface.add_question_interactive()
        elif choice == "2":
            interface.list_questions()
        elif choice == "3":
            interface.search_questions()
        elif choice == "4":
            question_id = input("Question ID: ").strip()
            if question_id:
                interface.run_question(question_id, False)
        elif choice == "5":
            question_id = input("Question ID: ").strip()
            if question_id:
                interface.run_question(question_id, True)
        elif choice == "6":
            filename = input("Export filename: ").strip()
            if filename:
                interface.export_questions(filename)
        elif choice == "7":
            filename = input("Import filename: ").strip()
            if filename:
                interface.import_questions(filename)
        elif choice == "8":
            handle_config_interactive()
        elif choice == "9":
            print("Goodbye!")
            break
        else:
            print("Invalid option!")


def handle_config_interactive():
    """Handle configuration in interactive mode."""
    config_manager = ConfigManager()
    
    while True:
        print("\n=== Configuration ===")
        print("1. Show current config")
        print("2. Update safety settings")
        print("3. Update vision settings")
        print("4. Update speech settings")
        print("5. Reset to defaults")
        print("6. Back to main menu")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            config = config_manager.get_config()
            print(f"Safety - Min Luma: {config.safety.min_mean_luma}")
            print(f"Vision - Min CV Conf: {config.vision.min_cv_conf}")
            print(f"Speech - Model: {config.speech.model_name}")
            print(f"Debug Mode: {config.debug_mode}")
        elif choice == "2":
            try:
                min_luma = float(input(f"Min mean luma (current: {config_manager.get_config().safety.min_mean_luma}): "))
                config_manager.update_config(min_mean_luma=min_luma)
                print("Safety settings updated!")
            except ValueError:
                print("Invalid input!")
        elif choice == "3":
            try:
                min_cv_conf = float(input(f"Min CV confidence (current: {config_manager.get_config().vision.min_cv_conf}): "))
                config_manager.update_config(min_cv_conf=min_cv_conf)
                print("Vision settings updated!")
            except ValueError:
                print("Invalid input!")
        elif choice == "4":
            model_name = input(f"Speech model name (current: {config_manager.get_config().speech.model_name}): ")
            if model_name:
                config_manager.update_config(model_name=model_name)
                print("Speech settings updated!")
        elif choice == "5":
            config_manager.reset_to_default()
            print("Configuration reset to defaults!")
        elif choice == "6":
            break
        else:
            print("Invalid option!")


if __name__ == "__main__":
    main()
