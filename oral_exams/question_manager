"""Question and expected response management system."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum


class DifficultyLevel(Enum):
    """Difficulty levels for questions."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class QuestionType(Enum):
    """Types of questions."""
    CONCEPTUAL = "conceptual"
    CALCULATION = "calculation"
    INTERPRETATION = "interpretation"
    APPLICATION = "application"


@dataclass
class ExpectedResponse:
    """Expected response structure with key concepts and scoring criteria."""
    key_concepts: List[str]
    required_terms: List[str]
    optional_terms: List[str]
    example_phrases: List[str]
    common_misconceptions: List[str]
    scoring_criteria: Dict[str, Any]
    difficulty: DifficultyLevel
    question_type: QuestionType


@dataclass
class Question:
    """Question structure with metadata and expected responses."""
    question_id: str
    stem: str
    context: Optional[str] = None
    expected_response: ExpectedResponse = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class QuestionManager:
    """Manages questions and expected responses for the tutoring system."""
    
    def __init__(self, questions_file: Optional[Path] = None):
        self.questions_file = questions_file or Path("oral_exams/resources/questions.json")
        self.questions: Dict[str, Question] = {}
        self.load_questions()
    
    def load_questions(self) -> None:
        """Load questions from the JSON file."""
        if not self.questions_file.exists():
            self.questions_file.parent.mkdir(parents=True, exist_ok=True)
            self.save_questions()
            return
        
        try:
            with open(self.questions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for q_data in data.get('questions', []):
                question = self._dict_to_question(q_data)
                self.questions[question.question_id] = question
        except Exception as e:
            print(f"Error loading questions: {e}")
            self._create_default_questions()
    
    def save_questions(self) -> None:
        """Save questions to the JSON file."""
        data = {
            'questions': [self._question_to_dict(q) for q in self.questions.values()]
        }
        
        with open(self.questions_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def add_question(self, question: Question) -> None:
        """Add a new question."""
        self.questions[question.question_id] = question
        self.save_questions()
    
    def get_question(self, question_id: str) -> Optional[Question]:
        """Get a question by ID."""
        return self.questions.get(question_id)
    
    def list_questions(self) -> List[Question]:
        """List all questions."""
        return list(self.questions.values())
    
    def search_questions(self, topic: str = None, difficulty: DifficultyLevel = None, 
                        question_type: QuestionType = None) -> List[Question]:
        """Search questions by criteria."""
        results = []
        for question in self.questions.values():
            if topic and topic.lower() not in question.stem.lower():
                continue
            if difficulty and question.expected_response.difficulty != difficulty:
                continue
            if question_type and question.expected_response.question_type != question_type:
                continue
            results.append(question)
        return results
    
    def _question_to_dict(self, question: Question) -> Dict[str, Any]:
        """Convert question to dictionary for JSON serialization."""
        data = asdict(question)
        if question.expected_response:
            data['expected_response']['difficulty'] = question.expected_response.difficulty.value
            data['expected_response']['question_type'] = question.expected_response.question_type.value
        return data
    
    def _dict_to_question(self, data: Dict[str, Any]) -> Question:
        """Convert dictionary to question object."""
        expected_response_data = data.get('expected_response', {})
        if expected_response_data:
            expected_response = ExpectedResponse(
                key_concepts=expected_response_data.get('key_concepts', []),
                required_terms=expected_response_data.get('required_terms', []),
                optional_terms=expected_response_data.get('optional_terms', []),
                example_phrases=expected_response_data.get('example_phrases', []),
                common_misconceptions=expected_response_data.get('common_misconceptions', []),
                scoring_criteria=expected_response_data.get('scoring_criteria', {}),
                difficulty=DifficultyLevel(expected_response_data.get('difficulty', 'beginner')),
                question_type=QuestionType(expected_response_data.get('question_type', 'conceptual'))
            )
        else:
            expected_response = None
        
        return Question(
            question_id=data['question_id'],
            stem=data['stem'],
            context=data.get('context'),
            expected_response=expected_response,
            metadata=data.get('metadata', {})
        )
    
    def _create_default_questions(self) -> None:
        """Create default questions for regression topics."""
        default_questions = [
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
        
        for question in default_questions:
            self.questions[question.question_id] = question
        
        self.save_questions()
