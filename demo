#!/usr/bin/env python3
"""Demo script showing the key features of the enhanced tutoring system."""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def demo_question_management():
    """Demonstrate question management features."""
    print("🎯 DEMO: Question Management System")
    print("=" * 50)
    
    from oral_exams.question_manager import QuestionManager
    
    # Create question manager
    qm = QuestionManager()
    
    # List available questions
    questions = qm.list_questions()
    print(f"📚 Found {len(questions)} questions:")
    
    for i, question in enumerate(questions, 1):
        print(f"  {i}. {question.question_id}")
        print(f"     Question: {question.stem}")
        if question.expected_response:
            print(f"     Difficulty: {question.expected_response.difficulty.value}")
            print(f"     Type: {question.expected_response.question_type.value}")
            print(f"     Key Concepts: {', '.join(question.expected_response.key_concepts[:3])}")
        print()
    
    return questions

def demo_scoring_analysis():
    """Demonstrate scoring and analysis features."""
    print("🧠 DEMO: Intelligent Response Analysis")
    print("=" * 50)
    
    from oral_exams.question_manager import QuestionManager
    from oral_exams.agents.scoring import EnhancedScoringAgent
    from oral_exams.models import TurnBundle, ItemContext, SpeechObservation
    
    # Get a question
    qm = QuestionManager()
    question = qm.get_question("r2_interpretation_01")
    
    if not question:
        print("❌ No question found!")
        return
    
    print(f"📝 Question: {question.stem}")
    print()
    
    # Test different student responses
    test_responses = [
        "R squared of 0.64 means 64% of the variance in exam scores is explained by study hours",
        "R squared is 0.64, so it's 64% accurate",
        "I don't know what R squared means",
        "R squared means correlation"
    ]
    
    scoring_agent = EnhancedScoringAgent(qm)
    
    for i, response in enumerate(test_responses, 1):
        print(f"🎤 Student Response {i}: {response}")
        
        # Create test bundle
        bundle = TurnBundle(
            item=ItemContext(item_id=question.question_id, stem=question.stem),
            raw_video_frame=None,
            raw_audio_path=None
        )
        
        # Create speech observation
        speech_obs = SpeechObservation(
            transcript=response,
            asr_conf=0.95,
            wpm=120.0,
            fillers_per_min=0.0
        )
        
        # Score the response
        result = scoring_agent.score(bundle, speech_obs, None, None, None)
        
        print(f"   📊 Content Score: {result.content.content_score}/10")
        print(f"   💬 Feedback: {result.next_action.payload}")
        print(f"   🎯 Action: {result.next_action.type}")
        
        if result.ui_notes.get("response_analysis"):
            analysis = result.ui_notes["response_analysis"]
            if "key_concepts_mentioned" in analysis:
                print(f"   🔍 Key Concepts: {analysis['key_concepts_mentioned']}")
            if "misconceptions_detected" in analysis:
                print(f"   ⚠️  Misconceptions: {analysis['misconceptions_detected']}")
        
        print()

def demo_coaching_strategies():
    """Demonstrate coaching strategies."""
    print("🎓 DEMO: Adaptive Coaching Strategies")
    print("=" * 50)
    
    from oral_exams.question_manager import QuestionManager
    from oral_exams.agents.coaching import EnhancedCoachingAgent
    from oral_exams.models import TurnBundle, ItemContext, ScoringResult, ContentScore, VoiceScore, NextAction
    
    # Get a question
    qm = QuestionManager()
    question = qm.get_question("r2_interpretation_01")
    
    if not question:
        print("❌ No question found!")
        return
    
    coaching_agent = EnhancedCoachingAgent(qm)
    
    # Test different performance levels
    performance_levels = [
        (8, "High performance - student understands well"),
        (5, "Medium performance - student has partial understanding"),
        (2, "Low performance - student needs more help"),
        (1, "Very low performance - student is struggling")
    ]
    
    for score, description in performance_levels:
        print(f"📊 {description} (Score: {score}/10)")
        
        # Create test bundle
        bundle = TurnBundle(
            item=ItemContext(item_id=question.question_id, stem=question.stem),
            raw_video_frame=None,
            raw_audio_path=None
        )
        
        # Create scoring result
        scoring_result = ScoringResult(
            content=ContentScore(
                content_score=score,
                evidence=[],
                misconceptions=[] if score > 3 else ["R² is the correlation coefficient"],
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
        
        # Get coaching
        action = coaching_agent.coach(bundle, scoring_result, None)
        
        print(f"   🎯 Strategy: {action.type}")
        print(f"   💬 Feedback: {action.payload}")
        print()

def demo_simulation():
    """Demonstrate simulation mode."""
    print("🖥️  DEMO: Simulation Mode")
    print("=" * 50)
    
    import subprocess
    import sys
    
    print("Running simulation with different student responses...")
    print()
    
    # Test responses
    responses = [
        "R squared of 0.64 means 64% of the variance is explained",
        "R squared is the correlation coefficient",
        "I think R squared means accuracy"
    ]
    
    for i, response in enumerate(responses, 1):
        print(f"🎤 Simulation {i}: {response}")
        
        try:
            result = subprocess.run([
                sys.executable, "examples/enhanced_simulated_turn.py",
                "--question-id", "r2_interpretation_01",
                "--response", response
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Extract key information from output
                lines = result.stdout.split('\n')
                for line in lines:
                    if "Action Type:" in line:
                        print(f"   🎯 {line.strip()}")
                    elif "Feedback:" in line:
                        print(f"   💬 {line.strip()}")
            else:
                print(f"   ❌ Error: {result.stderr}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()

def main():
    """Run the complete demo."""
    print("🎉 Enhanced Tutoring System Demo")
    print("=" * 60)
    print()
    
    try:
        # Demo 1: Question Management
        questions = demo_question_management()
        
        if not questions:
            print("❌ No questions available for demo!")
            return
        
        # Demo 2: Scoring Analysis
        demo_scoring_analysis()
        
        # Demo 3: Coaching Strategies
        demo_coaching_strategies()
        
        # Demo 4: Simulation Mode
        demo_simulation()
        
        print("🎉 Demo completed successfully!")
        print()
        print("🚀 Next Steps:")
        print("1. Run: python3 tutoring_cli.py")
        print("2. Try: python3 tutoring_cli.py run r2_interpretation_01 --simulation")
        print("3. Add your own questions: python3 tutoring_cli.py add")
        print("4. Explore the code in VS Code!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
