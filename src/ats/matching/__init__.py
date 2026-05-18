"""Matching engine — Strategy Pattern.

Public surface:
    from ats.matching import MATCHERS, CandidateMatch, MatchingStrategy

`MATCHERS` is the registry consumed by FastAPI and the CLI:
    MATCHERS[strategy_name].match_by_vacancy(session, job_id)
"""
from ats.matching.base import (
    CandidateMatch,
    EmbeddingMissing,
    MatchingStrategy,
    VacancyNotFound,
)
from ats.matching.llm import LlmMatcher
from ats.matching.rrf import RrfMatcher
from ats.matching.semantic import SemanticMatcher
from ats.matching.tfidf import TfidfMatcher

MATCHERS: dict[str, MatchingStrategy] = {
    SemanticMatcher.name: SemanticMatcher(),
    TfidfMatcher.name: TfidfMatcher(),
    RrfMatcher.name: RrfMatcher(),
    LlmMatcher.name: LlmMatcher(),
}

__all__ = [
    "MATCHERS",
    "CandidateMatch",
    "EmbeddingMissing",
    "LlmMatcher",
    "MatchingStrategy",
    "RrfMatcher",
    "SemanticMatcher",
    "TfidfMatcher",
    "VacancyNotFound",
]
