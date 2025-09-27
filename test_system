#!/usr/bin/env python3
"""Test script to verify the tutoring system works correctly."""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from oral_exams.question_manager import QuestionManager, Question, ExpectedResponse, DifficultyLevel, QuestionType
        print("✅ Question Manager imported successfully")
    except Exception as e:
        print(f"❌ Question Manager import failed: {e}")
        return False
    
    try:
        from oral_exams.tutoring_interface import TutoringInterface
        print("✅ Tutoring Interface imported successfully")
    except Exception as e:
        print(f"❌ Tutoring Interface import failed: {e}")
        return False
    
    try:
        from oral_exams.config import ConfigManager, TutoringConfig
        print("✅ Configuration system imported successfully")
    except Exception as e:
        print(f"❌ Configuration system import failed: {e}")
        return False
    
    try:
        from oral_exams.agents.scoring import EnhancedScoringAgent
        print("✅ Enhanced Scoring Agent imported successfully")
    except Exception as e:
        print(f"❌ Enhanced Scoring Agent import failed: {e}")
        return False
    
    try:
        from oral_exams.agents.coaching import EnhancedCoachingAgent
        print("✅ Enhanced Coaching Agent imported successfully")
    except Exception as e:
        print(f"❌ Enhanced Coaching Agent import failed: {e}")
        return False
    
    return True

def test_question_manager():
    """Test the question manager functionality."""
    print("\nTesting Question Manager...")
    
    try:
        from oral_exams.question_manager import QuestionManager, Question, ExpectedResponse, DifficultyLevel, QuestionType
        
        # Create a test question manager
        qm = QuestionManager()
        
        # Check if questions exist
        questions = qm.list_questions()
        print(f"✅ Found {len(questions)} questions")
        
        if questions:
            question = questions[0]
            print(f"✅ Sample question: {question.question_id}")
            print(f"   Stem: {question.stem}")
            if question.expected_response:
                print(f"   Difficulty: {question.expected_response.difficulty.value}")
                print(f"   Type: {question.expected_response.question_type.value}")
        
        return True
    except Exception as e:
        print(f"❌ Question Manager test failed: {e}")
        return False

def test_scoring_agent():
    """Test the enhanced scoring agent."""
    print("\nTesting Enhanced Scoring Agent...")
    
    try:
        from oral_exams.question_manager import QuestionManager
        from oral_exams.agents.scoring import EnhancedScoringAgent
        from oral_exams.models import TurnBundle, ItemContext, SpeechObservation
        
        # Create question manager and scoring agent
        qm = QuestionManager()
        scoring_agent = EnhancedScoringAgent(qm)
        
        # Create a test bundle
        bundle = TurnBundle(
            item=ItemContext(item_id="test", stem="Test question"),
            raw_video_frame=None,
            raw_audio_path=None
        )
        
        # Create a test speech observation
        speech_obs = SpeechObservation(
            transcript="R squared means variance explained",
            asr_conf=0.95,
            wpm=120.0,
            fillers_per_min=0.0
        )
        
        # Test scoring (this will use basic scoring since no question context)
        result = scoring_agent.score(bundle, speech_obs, None, None, None)
        
        print(f"✅ Scoring test completed")
        print(f"   Content Score: {result.content.content_score}")
        print(f"   Action Type: {result.next_action.type}")
        print(f"   Feedback: {result.next_action.payload}")
        
        return True
    except Exception as e:
        print(f"❌ Scoring Agent test failed: {e}")
        return False

def test_coaching_agent():
    """Test the enhanced coaching agent."""
    print("\nTesting Enhanced Coaching Agent...")
    
    try:
        from oral_exams.question_manager import QuestionManager
        from oral_exams.agents.coaching import EnhancedCoachingAgent
        from oral_exams.models import TurnBundle, ItemContext, ScoringResult, ContentScore, VoiceScore, NextAction
        
        # Create question manager and coaching agent
        qm = QuestionManager()
        coaching_agent = EnhancedCoachingAgent(qm)
        
        # Create a test bundle
        bundle = TurnBundle(
            item=ItemContext(item_id="test", stem="Test question"),
            raw_video_frame=None,
            raw_audio_path=None
        )
        
        # Create a test scoring result
        scoring_result = ScoringResult(
            content=ContentScore(
                content_score=5,
                evidence=[],
                misconceptions=[],
                assumption_flags=[],
                unit_check={}
            ),
            voice=VoiceScore(
                clarity=3,
                terminology=2,
                coherence=3,
                pace="ok",
                fillers_per_min=0.0,
                coach_notes=[]
            ),
            next_action=NextAction(
                type="test",
                payload="Test feedback"
            ),
            ui_notes={}
        )
        
        # Test coaching
        action = coaching_agent.coach(bundle, scoring_result, None)
        
        print(f"✅ Coaching test completed")
        print(f"   Action Type: {action.type}")
        print(f"   Feedback: {action.payload}")
        
        return True
    except Exception as e:
        print(f"❌ Coaching Agent test failed: {e}")
        return False

def test_configuration():
    """Test the configuration system."""
    print("\nTesting Configuration System...")
    
    try:
        from oral_exams.config import ConfigManager, create_default_config_file
        
        # Create default config
        create_default_config_file(Path("test_config.json"))
        print("✅ Default configuration created")
        
        # Load configuration
        config_manager = ConfigManager(Path("test_config.json"))
        config = config_manager.get_config()
        
        print(f"✅ Configuration loaded")
        print(f"   Safety - Min Luma: {config.safety.min_mean_luma}")
        print(f"   Vision - Min CV Conf: {config.vision.min_cv_conf}")
        print(f"   Speech - Model: {config.speech.model_name}")
        
        # Clean up
        Path("test_config.json").unlink()
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_simulated_turn():
    """Test the simulated turn functionality."""
    print("\nTesting Simulated Turn...")
    
    try:
        import subprocess
        import sys
        
        # Run the enhanced simulated turn
        result = subprocess.run([
            sys.executable, "examples/enhanced_simulated_turn.py", 
            "--question-id", "r2_interpretation_01",
            "--response", "R squared means variance explained"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Simulated turn test completed successfully")
            print("   Output preview:")
            lines = result.stdout.split('\n')
            for line in lines[:10]:  # Show first 10 lines
                if line.strip():
                    print(f"   {line}")
            return True
        else:
            print(f"❌ Simulated turn test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Simulated turn test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Enhanced Tutoring System")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_question_manager,
        test_scoring_agent,
        test_coaching_agent,
        test_configuration,
        test_simulated_turn
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run: python3 tutoring_cli.py")
        print("2. Or run: python3 tutoring_cli.py run r2_interpretation_01 --simulation")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
