"""Response models for the API layer.

Most of the matching engine's output (`CandidateMatch`) is already
pydantic-frozen, so we re-export it here without copying. The thin
envelopes below wrap lists with metadata that's useful for paginated /
strategy-tagged responses but not part of the matching contract.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ats.matching.base import CandidateMatch

__all__ = [
    "CandidateMatch",
    "RecommendationsResponse",
    "VacancySummary",
    "VacancyDetail",
    "VacancyCreate",
    "VacancyUpdate",
    "CandidateSummary",
    "CandidateDetail",
    "CandidateUpdate",
]


class RecommendationsResponse(BaseModel):
    strategy: str = Field(description="matcher name: semantic | tfidf | llm")
    top_k: int
    count: int
    results: list[CandidateMatch]

    model_config = ConfigDict(frozen=True)


class VacancySummary(BaseModel):
    id: int
    title: str
    source_filename: str
    experience: str | None = None

    model_config = ConfigDict(frozen=True)


class VacancyDetail(BaseModel):
    id: int
    title: str
    description: str
    experience: str | None
    source_filename: str
    created_at: datetime

    model_config = ConfigDict(frozen=True)


class VacancyCreate(BaseModel):
    title: str = Field(min_length=1, max_length=512)
    description: str = Field(min_length=1)
    experience: str | None = Field(default=None, max_length=128)
    source_filename: str | None = Field(
        default=None,
        max_length=512,
        description="Optional. Auto-generated as `ui_<uuid8>.json` when omitted.",
    )

    model_config = ConfigDict(frozen=True)


class VacancyUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=512)
    description: str | None = Field(default=None, min_length=1)
    experience: str | None = Field(default=None, max_length=128)

    model_config = ConfigDict(frozen=True)


class CandidateSummary(BaseModel):
    id: int
    name: str | None
    email: str | None
    source_file: str

    model_config = ConfigDict(frozen=True)


class CandidateDetail(BaseModel):
    id: int
    name: str | None
    email: str | None
    source_file: str
    parsed_json: dict | None
    raw_text_excerpt: str | None = Field(
        default=None,
        description="First 3000 chars of raw_text — for UI display only",
    )
    created_at: datetime

    model_config = ConfigDict(frozen=True)


class CandidateUpdate(BaseModel):
    """Manual HR-side edit of name/email. Other fields stay parser-managed.

    Empty-string values are accepted and stored as NULL (so the UI can clear
    a wrong name). Use `None` (omit the field) to leave it unchanged.
    """

    name: str | None = Field(default=None, max_length=256)
    email: str | None = Field(default=None, max_length=256)

    model_config = ConfigDict(frozen=True)
