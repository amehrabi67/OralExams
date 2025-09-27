"""Retrieval and RAG coverage agent."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..models import (
    GroundingObservation,
    RetrievalHit,
    RetrievalResult,
    SpeechObservation,
    TurnBundle,
)


class RetrievalAgent(Protocol):
    def retrieve(
        self,
        bundle: TurnBundle,
        speech: SpeechObservation,
        grounding: GroundingObservation | None,
    ) -> RetrievalResult | None:
        ...


@dataclass
class KnowledgeBaseEntry:
    id: str
    title: str
    content: str
    checkpoints: List[str]


class LocalRetrievalAgent:
    """In-memory TF-IDF retrieval over a small curated knowledge base."""

    def __init__(
        self,
        kb_path: str | Path | None = None,
        top_k: int = 3,
    ) -> None:
        kb_path = Path(kb_path or Path(__file__).resolve().parent.parent / "resources" / "knowledge_base.json")
        with open(kb_path, "r", encoding="utf-8") as handle:
            raw_entries = json.load(handle)
        self.entries = [KnowledgeBaseEntry(**entry) for entry in raw_entries]
        self.top_k = top_k
        self._vectorizer = TfidfVectorizer(stop_words="english")
        corpus = [entry.content for entry in self.entries]
        self._matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(
        self,
        bundle: TurnBundle,
        speech: SpeechObservation,
        grounding: GroundingObservation | None,
    ) -> RetrievalResult | None:
        query_terms = [bundle.item.stem, speech.transcript]
        if grounding:
            query_terms.extend(grounding.math_fragments)
        query = "\n".join(term for term in query_terms if term)
        if not query.strip():
            return None

        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix)[0]
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[: self.top_k]

        hits: list[RetrievalHit] = []
        covered_checkpoints: set[str] = set()
        for index, score in ranked:
            entry = self.entries[index]
            snippet = entry.content[:220] + ("…" if len(entry.content) > 220 else "")
            hits.append(
                RetrievalHit(
                    id=entry.id,
                    span=entry.title,
                    snippet=snippet,
                )
            )
            covered_checkpoints.update(entry.checkpoints)

        rag_cov = _estimate_coverage(bundle.item.stem, covered_checkpoints)
        return RetrievalResult(hits=hits, rag_cov=rag_cov, conflicts=[])


def _estimate_coverage(stem: str, covered: set[str]) -> float:
    checkpoints: set[str] = set()
    stem_lower = stem.lower()
    if "r^2" in stem_lower or "r-squared" in stem_lower:
        checkpoints.update({"variance_explained", "no_causality"})
    if "slope" in stem_lower:
        checkpoints.add("slope_units")
    if "intercept" in stem_lower:
        checkpoints.add("intercept_context")
    if "assumption" in stem_lower:
        checkpoints.add("assumptions")

    if not checkpoints:
        return 0.5 if covered else 0.0

    matched = checkpoints.intersection(covered)
    return len(matched) / len(checkpoints)

