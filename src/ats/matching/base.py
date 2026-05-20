"""Strategy interface for the matching engine.

`MatchingStrategy` is the ABC that semantic / TF-IDF / RRF / LLM matchers
implement. `CandidateMatch` is the shared output schema — FastAPI returns
it directly, Streamlit consumes it for rendering.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

# Source-file prefix for the Resume.csv eval corpus. Production matchers
# (Semantic, TF-IDF) exclude rows whose `source_file` starts with this
# prefix so the UI never surfaces eval-only mock candidates. Eval scripts
# bypass the matchers and run their own pgvector queries scoped to this
# prefix explicitly.
RESUME_CSV_PREFIX = "resume_csv_"


class CandidateMatch(BaseModel):
    candidate_id: int
    name: str | None
    email: str | None
    source_file: str
    score: float                       # ∈ [0, 1] — higher = better
    strategy: str                      # "semantic" / "tfidf" / / "rrf" / "llm"
    explanation: str | None = None     # populated by the LLM strategy
    parsed_json: dict | None = None    # full ParsedResume minus raw_text / embedding

    model_config = ConfigDict(frozen=True)


class VacancyNotFound(LookupError):
    """Raised when the requested vacancy id is not in the database."""


class EmbeddingMissing(RuntimeError):
    """Raised when a row needed for matching has no embedding yet."""


class MatchingStrategy(ABC):
    """Strategy pattern for matching candidates against a vacancy."""

    name: str = "abstract"

    @abstractmethod
    async def match_by_vacancy(
        self,
        session: AsyncSession,
        job_id: int,
        top_k: int | None = None,
    ) -> list[CandidateMatch]:
        """Rank candidates against the vacancy row with the given id."""

    @abstractmethod
    async def match_by_text(
        self,
        session: AsyncSession,
        vacancy_text: str,
        top_k: int | None = None,
        vacancy_title: str | None = None,
    ) -> list[CandidateMatch]:
        """Rank candidates against ad-hoc vacancy text (no DB row required)."""
