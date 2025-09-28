"""Enhanced scoring agent for rubric-based evaluation with question management."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, List, Dict, Any, Optional

from ..models import (
    ContentScore,
    GroundingObservation,
    NextAction,
    RetrievalResult,
    ScoringResult,
    SpeechObservation,
    TurnBundle,
    VisionObservation,
    VoiceScore,
)
from ..question_manager import QuestionManager, Question, ExpectedResponse


class ScoringAgent(Protocol):
    def score(
        self,
        bundle: TurnBundle,
        speech: SpeechObservation,
        vision: VisionObservation | None,
        grounding: GroundingObservation | None,
        retrieval: RetrievalResult | None,
    ) -> ScoringResult:
        ...


@dataclass
class ResponseAnalysis:
    """Analysis of student response against expected answer."""
    key_concepts_mentioned: List[str]
    required_terms_mentioned: List[str]
    optional_terms_mentioned: List[str]
    misconceptions_detected: List[str]
    example_phrases_found: List[str]
    score_breakdown: Dict[str, int]
    total_score: int
    max_possible_score: int
    feedback_points: List[str]


class EnhancedScoringAgent:
    """Enhanced scorer that compares student responses with expected answers."""

    def __init__(self, question_manager: Optional[QuestionManager] = None):
        self.question_manager = question_manager or QuestionManager()
        self._response_cache: Dict[str, ResponseAnalysis] = {}

    def score(
        self,
        bundle: TurnBundle,
        speech: SpeechObservation,
        vision: VisionObservation | None,
        grounding: GroundingObservation | None,
        retrieval: RetrievalResult | None,
    ) -> ScoringResult:
        """Score student response against expected answer."""
        transcript = speech.transcript.lower()
        
        # Get the question and expected response
        question = self._get_question_from_bundle(bundle)
        if not question or not question.expected_response:
            # Fallback to basic scoring if no question context
            return self._basic_score(bundle, speech, vision, grounding, retrieval)
        
        # Analyze response against expected answer
        analysis = self._analyze_response(transcript, question.expected_response)
        
        # Calculate content score based on analysis
        content_score = self._calculate_content_score(analysis, question.expected_response)
        
        # Detect misconceptions
        misconceptions = self._detect_misconceptions(transcript, question.expected_response)
        
        # Get evidence from retrieval
        evidence = [f"{hit.id}:{hit.span}" for hit in retrieval.hits] if retrieval else []
        
        # Score voice quality
        voice_score = self._score_voice(speech)
        
        # Unit and assumption checks
        unit_check = self._check_units_and_assumptions(transcript, question)
        
        # Create content score
        content = ContentScore(
            content_score=content_score,
            evidence=evidence,
            misconceptions=misconceptions,
            assumption_flags=self._check_assumptions(transcript, question),
            unit_check=unit_check,
        )
        
        # Generate next action based on analysis
        next_action = self._generate_next_action(analysis, content, question, retrieval)
        
        # UI notes
        ui_notes = {
            "need_camera_reframe": bool(vision and vision.lighting == "dim"),
            "speech_confidence": speech.asr_conf,
            "response_analysis": {
                "key_concepts_mentioned": analysis.key_concepts_mentioned,
                "required_terms_mentioned": analysis.required_terms_mentioned,
                "misconceptions_detected": analysis.misconceptions_detected,
                "score_breakdown": analysis.score_breakdown,
                "total_score": f"{analysis.total_score}/{analysis.max_possible_score}"
            }
        }
        
        return ScoringResult(
            content=content,
            voice=voice_score,
            next_action=next_action,
            ui_notes=ui_notes,
        )

    def _get_question_from_bundle(self, bundle: TurnBundle) -> Optional[Question]:
        """Extract question from bundle context."""
        if hasattr(bundle.item, 'question_id'):
            return self.question_manager.get_question(bundle.item.question_id)
        return None

    def _analyze_response(self, transcript: str, expected: ExpectedResponse) -> ResponseAnalysis:
        """Analyze student response against expected answer."""
        key_concepts_mentioned = []
        required_terms_mentioned = []
        optional_terms_mentioned = []
        misconceptions_detected = []
        example_phrases_found = []
        
        # Check for key concepts
        for concept in expected.key_concepts:
            if self._fuzzy_match(concept, transcript):
                key_concepts_mentioned.append(concept)
        
        # Check for required terms
        for term in expected.required_terms:
            if self._fuzzy_match(term, transcript):
                required_terms_mentioned.append(term)
        
        # Check for optional terms
        for term in expected.optional_terms:
            if self._fuzzy_match(term, transcript):
                optional_terms_mentioned.append(term)
        
        # Check for misconceptions
        for misconception in expected.common_misconceptions:
            if self._fuzzy_match(misconception, transcript):
                misconceptions_detected.append(misconception)
        
        # Check for example phrases
        for phrase in expected.example_phrases:
            if self._fuzzy_match(phrase, transcript):
                example_phrases_found.append(phrase)
        
        # Calculate score breakdown
        score_breakdown = {}
        total_score = 0
        max_possible_score = 0
        
        for criterion, points in expected.scoring_criteria.items():
            max_possible_score += points
            if self._check_criterion(criterion, transcript, expected):
                score_breakdown[criterion] = points
                total_score += points
            else:
                score_breakdown[criterion] = 0
        
        return ResponseAnalysis(
            key_concepts_mentioned=key_concepts_mentioned,
            required_terms_mentioned=required_terms_mentioned,
            optional_terms_mentioned=optional_terms_mentioned,
            misconceptions_detected=misconceptions_detected,
            example_phrases_found=example_phrases_found,
            score_breakdown=score_breakdown,
            total_score=total_score,
            max_possible_score=max_possible_score,
            feedback_points=[]
        )

    def _fuzzy_match(self, pattern: str, text: str) -> bool:
        """Check if pattern matches text with fuzzy matching."""
        pattern_lower = pattern.lower()
        text_lower = text.lower()
        
        # Direct substring match
        if pattern_lower in text_lower:
            return True
        
        # Word-based matching
        pattern_words = set(re.findall(r'\w+', pattern_lower))
        text_words = set(re.findall(r'\w+', text_lower))
        
        # Check if most words from pattern are in text
        if len(pattern_words) > 0:
            match_ratio = len(pattern_words.intersection(text_words)) / len(pattern_words)
            return match_ratio >= 0.6
        
        return False

    def _check_criterion(self, criterion: str, transcript: str, expected: ExpectedResponse) -> bool:
        """Check if a specific scoring criterion is met."""
        criterion_lower = criterion.lower()
        
        if "mentions_percentage" in criterion_lower:
            return bool(re.search(r'\d+%|\d+\s*percent', transcript))
        elif "mentions_variance" in criterion_lower:
            return "variance" in transcript or "variation" in transcript
        elif "mentions_explanation" in criterion_lower:
            return any(word in transcript for word in ["explain", "means", "indicates", "shows"])
        elif "avoids_misconceptions" in criterion_lower:
            return len(self._detect_misconceptions(transcript, expected)) == 0
        elif "distinguishes_correlation" in criterion_lower:
            return "correlation" in transcript and "association" in transcript
        elif "distinguishes_regression" in criterion_lower:
            return "regression" in transcript and "predict" in transcript
        elif "mentions_causation" in criterion_lower:
            return "causation" in transcript or "causal" in transcript
        elif "provides_examples" in criterion_lower:
            return any(word in transcript for word in ["example", "for instance", "such as"])
        elif "completeness" in criterion_lower:
            return len(expected.required_terms) <= len(self._analyze_response(transcript, expected).required_terms_mentioned)
        
        return False

    def _calculate_content_score(self, analysis: ResponseAnalysis, expected: ExpectedResponse) -> int:
        """Calculate content score based on analysis."""
        # Base score from analysis
        base_score = analysis.total_score
        
        # Bonus for mentioning key concepts
        concept_bonus = min(2, len(analysis.key_concepts_mentioned))
        
        # Penalty for misconceptions
        misconception_penalty = len(analysis.misconceptions_detected) * 2
        
        # Calculate final score (0-10 scale)
        final_score = max(0, min(10, base_score + concept_bonus - misconception_penalty))
        
        return final_score

    def _detect_misconceptions(self, transcript: str, expected: ExpectedResponse) -> List[str]:
        """Detect misconceptions in student response."""
        misconceptions = []
        for misconception in expected.common_misconceptions:
            if self._fuzzy_match(misconception, transcript):
                misconceptions.append(misconception)
        return misconceptions

    def _check_units_and_assumptions(self, transcript: str, question: Question) -> Dict[str, bool]:
        """Check for proper units and assumptions."""
        unit_check = {}
        
        # Check for units if it's a calculation question
        if question.expected_response and question.expected_response.question_type.value == "calculation":
            unit_check["units_mentioned"] = bool(re.search(r'\b(per|each|unit|hour|day|week)\b', transcript))
        
        # Check for statistical assumptions
        unit_check["assumptions_mentioned"] = any(term in transcript for term in [
            "assumption", "assume", "condition", "requirement"
        ])
        
        return unit_check

    def _check_assumptions(self, transcript: str, question: Question) -> List[str]:
        """Check for statistical assumptions."""
        assumptions = []
        
        if "linear" in transcript and "relationship" in transcript:
            assumptions.append("linearity")
        if "independent" in transcript:
            assumptions.append("independence")
        if "variance" in transcript and "constant" in transcript:
            assumptions.append("homoscedasticity")
        if "normal" in transcript and "distribution" in transcript:
            assumptions.append("normality")
        
        return assumptions

    def _score_voice(self, speech: SpeechObservation) -> VoiceScore:
        """Score voice quality."""
        wpm = speech.wpm
        pace = "ok"
        if wpm and (wpm < 90 or wpm > 170):
            pace = "slow" if wpm < 90 else "fast"

        clarity = 1 if speech.transcript.strip() == "" else 3
        terminology = 3 if any(term in speech.transcript.lower() for term in ["variance", "regression", "correlation"]) else 2
        coherence = 3 if len(speech.transcript.split()) > 6 else 2

        return VoiceScore(
            clarity=clarity,
            terminology=terminology,
            coherence=coherence,
            pace=pace,
            fillers_per_min=speech.fillers_per_min,
            coach_notes=[],
        )

    def _generate_next_action(self, analysis: ResponseAnalysis, content: ContentScore, 
                            question: Question, retrieval: RetrievalResult | None) -> NextAction:
        """Generate next action based on analysis."""
        refs = [hit.id for hit in retrieval.hits[:1]] if retrieval else []
        
        # If there are misconceptions, address them first
        if content.misconceptions:
            misconception = content.misconceptions[0]
            return NextAction(
                type="explain_short",
                payload=f"Let me clarify: {misconception}",
                rag_refs=refs,
            )
        
        # If score is low, provide guided question
        if content.content_score <= 3:
            return NextAction(
                type="ask_guided_q",
                payload=f"Let's break this down: {question.stem}",
                rag_refs=refs,
            )
        
        # If score is medium, provide specific feedback
        elif content.content_score <= 6:
            missing_concepts = set(question.expected_response.key_concepts) - set(analysis.key_concepts_mentioned)
            if missing_concepts:
                return NextAction(
                    type="hint",
                    payload=f"Good start! Consider also mentioning: {', '.join(list(missing_concepts)[:2])}",
                    rag_refs=refs,
                )
        
        # If score is high, affirm and extend
        return NextAction(
            type="affirm",
            payload="Excellent explanation! You've covered the key concepts well.",
            rag_refs=refs,
        )

    def _basic_score(self, bundle: TurnBundle, speech: SpeechObservation, 
                    vision: VisionObservation | None, grounding: GroundingObservation | None, 
                    retrieval: RetrievalResult | None) -> ScoringResult:
        """Fallback basic scoring when no question context is available."""
        transcript = speech.transcript.lower()
        
        # Basic keyword detection
        mentions_variance = "variance" in transcript or "percent" in transcript
        mentions_r2 = "r^2" in transcript or "r-squared" in transcript or "sixty four" in transcript or "64" in transcript
        mentions_hours = "hour" in transcript
        explains_non_causal = "because" not in transcript and "cause" not in transcript
        
        points = sum([mentions_variance, mentions_r2, mentions_hours, explains_non_causal])
        content_score = max(0, min(4, points))
        
        misconceptions: list[str] = []
        if "cause" in transcript or "causes" in transcript:
            misconceptions.append("R² should not be interpreted as causal.")
        
        evidence = [f"{hit.id}:{hit.span}" for hit in retrieval.hits] if retrieval else []
        
        voice_score = self._score_voice(speech)
        unit_check = {
            "slope_units_present": "per" in transcript or "each" in transcript,
            "mentions_hours": mentions_hours,
        }
        
        content = ContentScore(
            content_score=content_score,
            evidence=evidence,
            misconceptions=misconceptions,
            assumption_flags=[],
            unit_check=unit_check,
        )
        
        refs = [hit.id for hit in retrieval.hits[:1]] if retrieval else []
        next_action = NextAction(
            type="ask_guided_q",
            payload="Please explain what R²=0.64 means in your own words.",
            rag_refs=refs,
        )
        
        ui_notes = {
            "need_camera_reframe": bool(vision and vision.lighting == "dim"),
            "speech_confidence": speech.asr_conf,
        }
        
        return ScoringResult(
            content=content,
            voice=voice_score,
            next_action=next_action,
            ui_notes=ui_notes,
        )


# Keep the original RuleBasedScoringAgent for backward compatibility
class RuleBasedScoringAgent(EnhancedScoringAgent):
    """Backward compatible rule-based scorer."""
    
    def __init__(self):
        super().__init__(None)  # No question manager for basic scoring
