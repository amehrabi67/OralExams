"""Enhanced coaching decision logic with question management."""
from __future__ import annotations

from typing import Protocol, List, Dict, Any, Optional
from dataclasses import dataclass

from ..models import NextAction, RetrievalResult, ScoringResult, TurnBundle
from ..question_manager import QuestionManager, Question, ExpectedResponse


class CoachingAgent(Protocol):
    def handle_low_coverage(
        self, bundle: TurnBundle, retrieval: RetrievalResult | None
    ) -> NextAction:
        ...

    def coach(
        self,
        bundle: TurnBundle,
        scoring: ScoringResult,
        retrieval: RetrievalResult | None,
    ) -> NextAction:
        ...


@dataclass
class CoachingStrategy:
    """Strategy for providing coaching based on student performance."""
    action_type: str
    payload: str
    priority: int
    conditions: List[str]


class EnhancedCoachingAgent:
    """Enhanced coaching agent that provides personalized feedback based on question context."""

    def __init__(self, question_manager: Optional[QuestionManager] = None):
        self.question_manager = question_manager or QuestionManager()
        self._coaching_strategies = self._initialize_strategies()

    def handle_low_coverage(
        self, bundle: TurnBundle, retrieval: RetrievalResult | None
    ) -> NextAction:
        """Handle cases where retrieval coverage is low."""
        question = self._get_question_from_bundle(bundle)
        
        if question and question.expected_response:
            # Provide specific guidance based on question type
            if question.expected_response.question_type.value == "conceptual":
                payload = "Let's review the key concepts. Can you explain what this question is asking about?"
            elif question.expected_response.question_type.value == "calculation":
                payload = "Let's work through this step by step. What formula or method should we use?"
            elif question.expected_response.question_type.value == "interpretation":
                payload = "Let's focus on what the numbers mean. How would you explain this result?"
            else:
                payload = "Let's break this down into smaller parts. What do you think this question is asking?"
        else:
            payload = "Not in notes. Nearest prerequisite: variance vs variance explained."
        
        refs = retrieval.hits[0:1] if retrieval else []
        return NextAction(
            type="inform_gap",
            payload=payload,
            rag_refs=[hit.id for hit in refs],
        )

    def coach(
        self,
        bundle: TurnBundle,
        scoring: ScoringResult,
        retrieval: RetrievalResult | None,
    ) -> NextAction:
        """Provide coaching based on scoring results and question context."""
        question = self._get_question_from_bundle(bundle)
        
        if not question or not question.expected_response:
            return scoring.next_action
        
        # Analyze the scoring results
        content_score = scoring.content.content_score
        misconceptions = scoring.content.misconceptions
        evidence = scoring.content.evidence
        
        # Get response analysis from UI notes if available
        response_analysis = scoring.ui_notes.get("response_analysis", {})
        
        # Determine coaching strategy
        strategy = self._select_coaching_strategy(
            content_score, misconceptions, response_analysis, question
        )
        
        # Generate personalized feedback
        feedback = self._generate_personalized_feedback(
            strategy, question, response_analysis, misconceptions
        )
        
        refs = [hit.id for hit in retrieval.hits[:1]] if retrieval else []
        
        return NextAction(
            type=strategy.action_type,
            payload=feedback,
            rag_refs=refs,
        )

    def _get_question_from_bundle(self, bundle: TurnBundle) -> Optional[Question]:
        """Extract question from bundle context."""
        if hasattr(bundle.item, 'question_id'):
            return self.question_manager.get_question(bundle.item.question_id)
        return None

    def _initialize_strategies(self) -> List[CoachingStrategy]:
        """Initialize coaching strategies based on performance levels."""
        return [
            # High performance strategies
            CoachingStrategy(
                action_type="affirm",
                payload="Excellent work! You've demonstrated a strong understanding.",
                priority=1,
                conditions=["high_score", "no_misconceptions"]
            ),
            CoachingStrategy(
                action_type="extend",
                payload="Great explanation! Let's explore this concept further.",
                priority=2,
                conditions=["high_score", "some_missing_concepts"]
            ),
            
            # Medium performance strategies
            CoachingStrategy(
                action_type="hint",
                payload="You're on the right track. Consider mentioning: {missing_concepts}",
                priority=3,
                conditions=["medium_score", "missing_key_concepts"]
            ),
            CoachingStrategy(
                action_type="clarify",
                payload="Let's clarify this concept. {concept_explanation}",
                priority=4,
                conditions=["medium_score", "unclear_explanation"]
            ),
            
            # Low performance strategies
            CoachingStrategy(
                action_type="guide",
                payload="Let's work through this step by step. {step_by_step_guidance}",
                priority=5,
                conditions=["low_score", "needs_guidance"]
            ),
            CoachingStrategy(
                action_type="explain",
                payload="Let me explain this concept: {concept_explanation}",
                priority=6,
                conditions=["low_score", "needs_explanation"]
            ),
            
            # Misconception strategies
            CoachingStrategy(
                action_type="correct",
                payload="Let me clarify a common misconception: {misconception_correction}",
                priority=7,
                conditions=["has_misconceptions"]
            ),
            CoachingStrategy(
                action_type="redirect",
                payload="Let's refocus on the main concept: {main_concept}",
                priority=8,
                conditions=["off_topic", "tangential"]
            )
        ]

    def _select_coaching_strategy(
        self, content_score: int, misconceptions: List[str], 
        response_analysis: Dict[str, Any], question: Question
    ) -> CoachingStrategy:
        """Select the most appropriate coaching strategy."""
        
        # Check for misconceptions first (highest priority)
        if misconceptions:
            return self._coaching_strategies[6]  # correct strategy
        
        # Determine performance level
        if content_score >= 8:
            performance_level = "high_score"
        elif content_score >= 5:
            performance_level = "medium_score"
        else:
            performance_level = "low_score"
        
        # Check for specific conditions
        conditions = [performance_level]
        
        if misconceptions:
            conditions.append("has_misconceptions")
        
        # Check for missing concepts
        key_concepts_mentioned = response_analysis.get("key_concepts_mentioned", [])
        required_concepts = question.expected_response.key_concepts
        missing_concepts = set(required_concepts) - set(key_concepts_mentioned)
        
        if missing_concepts:
            conditions.append("missing_key_concepts")
        
        # Find matching strategy
        for strategy in sorted(self._coaching_strategies, key=lambda x: x.priority):
            if all(condition in conditions for condition in strategy.conditions):
                return strategy
        
        # Default strategy
        return self._coaching_strategies[4]  # guide strategy

    def _generate_personalized_feedback(
        self, strategy: CoachingStrategy, question: Question, 
        response_analysis: Dict[str, Any], misconceptions: List[str]
    ) -> str:
        """Generate personalized feedback based on strategy and context."""
        payload = strategy.payload
        
        # Replace placeholders with specific content
        if "{missing_concepts}" in payload:
            key_concepts_mentioned = response_analysis.get("key_concepts_mentioned", [])
            required_concepts = question.expected_response.key_concepts
            missing_concepts = set(required_concepts) - set(key_concepts_mentioned)
            if missing_concepts:
                payload = payload.replace("{missing_concepts}", ", ".join(list(missing_concepts)[:2]))
            else:
                payload = "You're on the right track!"
        
        if "{concept_explanation}" in payload:
            # Provide a brief explanation of the main concept
            main_concept = question.expected_response.key_concepts[0] if question.expected_response.key_concepts else "the concept"
            payload = payload.replace("{concept_explanation}", f"{main_concept} is about...")
        
        if "{misconception_correction}" in payload:
            if misconceptions:
                misconception = misconceptions[0]
                payload = payload.replace("{misconception_correction}", f"{misconception}")
            else:
                payload = "Let me clarify this concept."
        
        if "{step_by_step_guidance}" in payload:
            if question.expected_response.question_type.value == "calculation":
                payload = payload.replace("{step_by_step_guidance}", 
                    "First, identify what you're calculating. Then, choose the right formula. Finally, plug in the values.")
            elif question.expected_response.question_type.value == "interpretation":
                payload = payload.replace("{step_by_step_guidance}", 
                    "First, identify what the number represents. Then, explain what it means in context. Finally, discuss the implications.")
            else:
                payload = payload.replace("{step_by_step_guidance}", 
                    "Let's start with the basic concept and build from there.")
        
        if "{main_concept}" in payload:
            main_concept = question.expected_response.key_concepts[0] if question.expected_response.key_concepts else "the main concept"
            payload = payload.replace("{main_concept}", main_concept)
        
        return payload

    def add_custom_strategy(self, strategy: CoachingStrategy) -> None:
        """Add a custom coaching strategy."""
        self._coaching_strategies.append(strategy)
        self._coaching_strategies.sort(key=lambda x: x.priority)

    def get_available_strategies(self) -> List[CoachingStrategy]:
        """Get all available coaching strategies."""
        return self._coaching_strategies.copy()


# Keep the original DefaultCoachingAgent for backward compatibility
class DefaultCoachingAgent(EnhancedCoachingAgent):
    """Backward compatible default coaching agent."""
    
    def __init__(self):
        super().__init__(None)  # No question manager for basic coaching
