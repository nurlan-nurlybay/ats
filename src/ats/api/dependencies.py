"""FastAPI dependency shims.

`get_session` is re-exported unchanged from `ats.db.base`. The strategy
resolver lives here so handlers can `Depends(resolve_matcher)` instead
of re-implementing the registry lookup + 400 in every endpoint.
"""
from __future__ import annotations

from fastapi import HTTPException, Query

from ats.core.config import settings
from ats.db.base import get_session
from ats.matching import MATCHERS
from ats.matching.base import MatchingStrategy

__all__ = ["get_session", "resolve_matcher"]


def resolve_matcher(
    strategy: str = Query(
        default=settings.matching.default_strategy,
        description="Matcher name: semantic | tfidf | llm",
    ),
) -> MatchingStrategy:
    if strategy not in MATCHERS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown strategy {strategy!r}; available: {sorted(MATCHERS)}",
        )
    return MATCHERS[strategy]
