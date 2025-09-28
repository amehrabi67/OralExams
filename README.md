# Enhanced On-Device Tutoring System for Regression Topics

A comprehensive, user-friendly tutoring system that allows educators to create custom questions with expected responses and provides intelligent feedback to students through multimodal analysis.
🎯 Overall Goal of OralExams

The project is a prototype of an on-device tutoring system for statistics and regression that fuses two worlds:

Body Gesture & Environment Sensing (behavioral layer)

Using OpenCV to capture facial orientation, gaze direction, hand/pen movement, and whether the learner is writing or pointing at the board/paper.

Optionally using OCR (RapidOCR) or a visual grounding model to extract equations or fragments written on the paper/whiteboard.

Purpose: provide context signals — e.g., “student is re-reading but not writing,” “board shows R² = 0.64,” “pen is active.”

Knowledge Analysis (cognitive/content layer)

Using Vosk for speech recognition to transcribe the learner’s spoken answer.

Running retrieval-augmented grading: searching a curated knowledge base for the correct interpretation of concepts (like regression assumptions or R² meaning).

Applying a scoring agent that checks student responses against a rubric, identifies misconceptions (e.g., “R² implies causation”), and assigns a content score.

Passing results to the coaching agent, which delivers one actionable piece of feedback (a guided question, a short explanation, or a worked example).

🧩 Why combine gestures + knowledge?

Gestures and body activity give a signal of engagement and process:

If the student is writing → they’re trying to compute.

If they keep pointing back to the same equation without answering → possible confusion.

If they’re re-reading and not progressing → pacing adjustment needed.

Knowledge analysis ensures the tutor doesn’t just guess from gestures but evaluates what the student actually said. The RAG + rubric system keeps feedback grounded, prevents hallucination, and ensures the tutor’s response stays aligned with the curriculum.

Together, these signals let the orchestrator adapt how to respond:

Confusion in gestures + weak transcript → slow down, prerequisite reminder.

Clear gestures + partial answer → targeted micro-hint.

Confident gestures + good answer → confirm correctness, maybe advance difficulty.

📌 Core Mission

To create an AI tutor that listens, watches, and grades in real time — but:

Runs fully on-device (privacy, no cloud).

Uses lightweight free models (OpenCV, Vosk, RapidOCR, TF-IDF retrieval).

Provides trustworthy, rubric-based coaching in statistics/regression.

Ultimately: make oral exams and live tutoring more scalable, by letting a machine evaluate both what students say and how they engage with the task physically.

## 🚀 Key Features

### ✨ **Question Management System**
- **Custom Questions**: Create questions with detailed expected responses
- **Scoring Criteria**: Define specific scoring rubrics for each question
- **Difficulty Levels**: Beginner, Intermediate, Advanced
- **Question Types**: Conceptual, Calculation, Interpretation, Application
- **Import/Export**: JSON-based question database with easy sharing

### 🧠 **Intelligent Response Analysis**
- **Fuzzy Matching**: Advanced text analysis to understand student responses
- **Concept Detection**: Identifies key concepts, required terms, and misconceptions
- **Adaptive Scoring**: Dynamic scoring based on question-specific criteria
- **Personalized Feedback**: Context-aware coaching strategies

### 🎯 **Multimodal Tutoring Pipeline**
- **Safety Agent**: Lighting and face detection for optimal learning conditions
- **Vision Agent**: Computer vision for visual content analysis
- **Speech Agent**: Real-time speech transcription with confidence scoring
- **Retrieval Agent**: Knowledge base search with coverage analysis
- **Scoring Agent**: Comprehensive response evaluation
- **Coaching Agent**: Personalized feedback and next steps

### 🛠️ **Easy-to-Use Interface**
- **Interactive CLI**: User-friendly command-line interface
- **Simulation Mode**: Test without camera/microphone
- **Live Mode**: Real-time tutoring with webcam and microphone
- **Configuration**: Customizable thresholds and settings

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/amehrabi67/OralExams.git
cd OralExams

# Install dependencies
pip install -e .

# For live mode with OCR support
pip install -e .[live]

# Set up Hugging Face token (optional)
export HF_TOKEN="your_hugging_face_token_here"
```

## 🚀 Quick Start

### 1. Interactive Mode
Install Missing Dependencies
subStep 1: Install OpenCV and Other Dependencies
source .venv/bin/activate
pip install opencv-python numpy scikit-learn sounddevice soundfile requests
subStep 2: Install Vosk (Optional)
pip install vosk==0.3.44
subStep 3: Test the System
python examples/enhanced_simulated_turn.py --list-questions
Run these commands in order:
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install all dependencies
pip install opencv-python numpy scikit-learn sounddevice soundfile requests vosk==0.3.44

# 3. Install the package in editable mode
pip install -e . --no-deps

# 4. Test the system
python examples/enhanced_simulated_turn.py --list-questions

```bash
python tutoring_cli.py
```

### 2. Run a Question
```bash
# Live mode (with camera and microphone)
python tutoring_cli.py run r2_interpretation_01

# Simulation mode (no hardware required)
python tutoring_cli.py run r2_interpretation_01 --simulation
```

### 3. List Available Questions
```bash
python tutoring_cli.py list
```

### 4. Add a New Question
```bash
python tutoring_cli.py add
```

## 📚 Usage Examples

### Creating a Custom Question

```bash
python tutoring_cli.py add
```

Follow the interactive prompts to create a question with:
- Question stem and context
- Key concepts and required terms
- Example phrases and common misconceptions
- Scoring criteria and difficulty level

### Running a Tutoring Session

```bash
# Live session with specific question
python tutoring_cli.py run regression_assumptions_01 --duration 15

# Simulation with custom response
python tutoring_cli.py run r2_interpretation_01 --simulation
```

### Searching Questions

```bash
# Search by topic
python tutoring_cli.py search --topic "regression"

# Search by difficulty
python tutoring_cli.py search --difficulty intermediate

# Search by type
python tutoring_cli.py search --type conceptual
```

### Configuration Management

```bash
# Create default configuration
python tutoring_cli.py config --create-default

# Show current settings
python tutoring_cli.py config --show

# Reset to defaults
python tutoring_cli.py config --reset
```

## 🏗️ Architecture

### Core Components

1. **Question Manager** (`oral_exams/question_manager.py`)
   - Manages questions and expected responses
   - Supports import/export functionality
   - Provides search and filtering capabilities

2. **Enhanced Scoring Agent** (`oral_exams/agents/scoring.py`)
   - Compares student responses with expected answers
   - Uses fuzzy matching for concept detection
   - Provides detailed scoring breakdown

3. **Enhanced Coaching Agent** (`oral_exams/agents/coaching.py`)
   - Generates personalized feedback
   - Uses adaptive coaching strategies
   - Provides context-aware guidance

4. **Tutoring Interface** (`oral_exams/tutoring_interface.py`)
   - Main user interface
   - Handles question management
   - Orchestrates tutoring sessions

5. **Configuration System** (`oral_exams/config.py`)
   - Manages system settings
   - Supports customization
   - Provides default configurations

### Agent Pipeline

```
Student Response → Safety Check → Vision Analysis → Speech Transcription
                                                           ↓
Coaching Feedback ← Scoring Analysis ← Retrieval Search ← Grounding
```

## 📝 Question Format

### Basic Question Structure
```json
{
  "question_id": "r2_interpretation_01",
  "stem": "Interpret R² = 0.64 for predicting exam score from hours studied.",
  "context": "A student has calculated R² = 0.64 for a linear regression model...",
  "expected_response": {
    "key_concepts": ["coefficient of determination", "variance explained"],
    "required_terms": ["R-squared", "64%", "variance", "explained"],
    "optional_terms": ["correlation", "linear relationship"],
    "example_phrases": [
      "R-squared of 0.64 means 64% of the variance is explained"
    ],
    "common_misconceptions": [
      "R² is the correlation coefficient"
    ],
    "scoring_criteria": {
      "mentions_percentage": 2,
      "mentions_variance": 2,
      "mentions_explanation": 2
    },
    "difficulty": "intermediate",
    "question_type": "interpretation"
  }
}
```

## ⚙️ Configuration

### Safety Settings
- `min_mean_luma`: Minimum lighting threshold (default: 40.0)
- `max_faces`: Maximum number of faces allowed (default: 1)

### Vision Settings
- `min_cv_conf`: Minimum computer vision confidence (default: 0.60)
- `roi_detection_enabled`: Enable region of interest detection

### Speech Settings
- `model_name`: Vosk model to use (default: "vosk-model-small-en-us-0.15")
- `duration`: Audio recording duration in seconds (default: 12.0)

### Retrieval Settings
- `min_rag_cov`: Minimum retrieval coverage threshold (default: 0.70)
- `max_hits`: Maximum number of retrieval hits

## 🔧 Advanced Usage

### Custom Scoring Criteria

Define custom scoring criteria for each question:

```python
scoring_criteria = {
    "mentions_percentage": 2,      # 2 points for mentioning percentage
    "mentions_variance": 2,        # 2 points for mentioning variance
    "mentions_explanation": 2,     # 2 points for explaining the concept
    "avoids_misconceptions": 1     # 1 point for avoiding common misconceptions
}
```

### Custom Coaching Strategies

The system supports multiple coaching strategies:
- **Affirm**: Positive reinforcement for good responses
- **Hint**: Gentle guidance for partial understanding
- **Clarify**: Explanation of unclear concepts
- **Guide**: Step-by-step guidance for complex topics
- **Correct**: Addressing misconceptions
- **Extend**: Building on good responses

### Batch Operations

```bash
# Export all questions
python tutoring_cli.py export all_questions.json

# Import questions from file
python tutoring_cli.py import imported_questions.json

# Search and export specific questions
python tutoring_cli.py search --difficulty advanced | python tutoring_cli.py export advanced_questions.json
```

## 🧪 Testing and Development

### Simulation Mode
Perfect for development and testing without hardware:

```bash
python tutoring_cli.py run r2_interpretation_01 --simulation
```

### Enhanced Simulated Turn
```bash
python examples/enhanced_simulated_turn.py --question-id r2_interpretation_01 --response "R squared means variance explained"
```

### Live Session
```bash
python examples/enhanced_live_session.py r2_interpretation_01 --duration 10 --enable-ocr
```

## 📊 Output Analysis

The system provides detailed analysis including:

- **Response Analysis**: Key concepts mentioned, required terms found
- **Scoring Breakdown**: Points earned for each criterion
- **Misconception Detection**: Common errors identified
- **Coaching Recommendations**: Next steps for improvement

### Example Output
```
=== Tutoring Results ===
State: COACH
Action Type: hint
Feedback: Good start! Consider also mentioning: variance, explanation

Detailed Analysis:
  response_analysis:
    key_concepts_mentioned: ['coefficient of determination']
    required_terms_mentioned: ['R-squared', '64%']
    misconceptions_detected: []
    score_breakdown: {'mentions_percentage': 2, 'mentions_variance': 0, 'mentions_explanation': 0}
    total_score: 2/7
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your questions and improvements
4. Test with simulation mode
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on the original OralExams framework
- Uses Vosk for speech recognition
- Integrates OpenCV for computer vision
- Leverages scikit-learn for machine learning

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Use the interactive CLI help: `python tutoring_cli.py --help`

---

**Happy Tutoring! 🎓**
