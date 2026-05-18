"""The `/recommendations` endpoint — the assignment's THE endpoint.

Accepts either a stored vacancy id OR ad-hoc text (mutually exclusive),
runs the chosen matcher, and returns top-K candidates with scores and
(when strategy=llm) natural-language explanations.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ats.api.dependencies import get_session, resolve_matcher
from ats.api.schemas import RecommendationsResponse
from ats.core.config import settings
from ats.core.logger import bind_context, clear_context, get_logger
from ats.matching import MATCHERS
from ats.matching.base import MatchingStrategy
from ats.matching.llm import LlmMatcher

log = get_logger(__name__)
router = APIRouter(tags=["recommendations"])


@router.get(
    "/recommendations",
    response_model=RecommendationsResponse,
    summary="Top-K candidates for a vacancy (by id) or ad-hoc text",
    responses={
        400: {"description": "Both/neither job_id and text, or unknown strategy"},
        404: {"description": "Vacancy id not found"},
        412: {"description": "Vacancy or candidate missing required embedding"},
    },
)
async def get_recommendations(
    job_id: int | None = Query(
        None,
        ge=1,
        description="DB-stored vacancy id. Mutually exclusive with `text`.",
    ),
    text: str | None = Query(
        None,
        min_length=1,
        max_length=20_000,
        description="Ad-hoc vacancy text. Mutually exclusive with `job_id`.",
    ),
    title: str | None = Query(
        None,
        max_length=512,
        description="Optional title to prepend to ad-hoc text.",
    ),
    top_k: int = Query(
        default=settings.matching.top_k,
        ge=1,
        le=50,
        description="Number of candidates to return.",
    ),
    retriever: str | None = Query(
        default=None,
        description=(
            "Override the LLM matcher's shortlist source. "
            "Honored only when `strategy=llm`. One of {semantic, tfidf, rrf}."
        ),
    ),
    matcher: MatchingStrategy = Depends(resolve_matcher),
    session: AsyncSession = Depends(get_session),
) -> RecommendationsResponse:
    if (job_id is None) == (text is None):
        raise HTTPException(
            status_code=400,
            detail="exactly one of {job_id, text} must be provided",
        )

    if retriever is not None:
        if matcher.name != "llm":
            raise HTTPException(
                400,
                f"retriever override only valid when strategy=llm "
                f"(got strategy={matcher.name!r})",
            )
        retriever_obj = MATCHERS.get(retriever)
        if retriever_obj is None or retriever_obj.name == "llm":
            raise HTTPException(
                400,
                f"invalid retriever {retriever!r}; "
                f"choose one of: semantic, tfidf, rrf",
            )
        # Build a fresh LlmMatcher wired to the chosen retriever for THIS
        # request only — the registry instance keeps its default.
        matcher = LlmMatcher(retriever=retriever_obj)

    bind_context(
        strategy=matcher.name, top_k=top_k,
        job_id=job_id, ad_hoc=text is not None,
        retriever=retriever,
    )
    try:
        if job_id is not None:
            results = await matcher.match_by_vacancy(
                session, job_id=job_id, top_k=top_k,
            )
        else:
            assert text is not None  # narrowing for type-checker
            results = await matcher.match_by_text(
                session, vacancy_text=text, top_k=top_k, vacancy_title=title,
            )
        log.info("api_recommendations_ok", count=len(results))
        return RecommendationsResponse(
            strategy=matcher.name,
            top_k=top_k,
            count=len(results),
            results=results,
        )
    finally:
        clear_context()
