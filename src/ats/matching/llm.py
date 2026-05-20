"""LLM rerank matcher — Qwen via DashScope (OpenAI-compatible).

Two-stage retrieval: a cheap matcher (default: SemanticMatcher) returns a
short-list, then the LLM rescores each candidate and produces a natural-
language explanation. This is the strategy that satisfies the assignment's
explainability rubric.

Production concerns:
  - bounded concurrency via asyncio.Semaphore (DashScope rate-limit guard)
  - retry + timeout via AsyncOpenAI's built-in max_retries / timeout
    (exp. backoff on 429 / 5xx — no tenacity / backoff needed)
  - three-layer JSON parsing: response_format=json_object → strict pydantic
    → regex-extract-first-object → fail
  - graceful degradation: a failed LLM call keeps the retriever's score and
    drops the explanation; the caller is never blocked
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Final, NamedTuple

from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ats.core.config import settings
from ats.core.logger import get_logger
from ats.core.prompts import load_prompts
from ats.db.models import Candidate, Vacancy
from ats.matching.base import (
    CandidateMatch,
    MatchingStrategy,
    VacancyNotFound,
)
from ats.matching.rrf import RrfMatcher

log = get_logger(__name__)

_CYRILLIC_RE: Final = re.compile(r"[Ѐ-ӿ]")
_JSON_OBJECT_RE: Final = re.compile(r"\{[\s\S]*\}")

_MAX_DESCRIPTION_CHARS = 3000
_MAX_RAW_TEXT_CHARS = 4000
_MAX_SKILLS = 30
_MAX_EXPERIENCE = 5
_MAX_EDUCATION = 3


class LlmParseError(RuntimeError):
    """Raised after all JSON-parsing layers fail."""


class LlmScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1, max_length=600)

    @field_validator("score", mode="before")
    @classmethod
    def _clamp(cls, v: Any) -> float:
        return max(0.0, min(1.0, float(v)))


class _VacInfo(NamedTuple):
    """Lightweight vacancy carrier — avoids coupling to the ORM row."""
    title: str
    description: str


# ─── module-level AsyncOpenAI singleton ─────────────────────────────────────

_client: AsyncOpenAI | None = None
_client_lock = asyncio.Lock()


async def _get_client() -> AsyncOpenAI:
    global _client
    async with _client_lock:
        if _client is None:
            _client = AsyncOpenAI(
                api_key=settings.llm_secrets.dashscope_api_key.get_secret_value(),
                base_url=settings.models.llm.base_url,
                timeout=settings.matching.llm.timeout_seconds,
                max_retries=settings.matching.llm.retry_attempts,
            )
            log.info(
                "llm_client_init",
                base_url=settings.models.llm.base_url,
                model=settings.models.llm.model,
                timeout=settings.matching.llm.timeout_seconds,
                retries=settings.matching.llm.retry_attempts,
            )
        return _client


# ─── prompt assembly ────────────────────────────────────────────────────────

def _is_russian(text: str) -> bool:
    return bool(_CYRILLIC_RE.search(text))


def _format_experience(items: list[dict]) -> str:
    if not items:
        return "(no experience parsed)"
    chunks = []
    for it in items[:_MAX_EXPERIENCE]:
        org = it.get("organization") or "?"
        role = it.get("role") or "?"
        start = it.get("start") or "?"
        end = it.get("end") or "?"
        chunks.append(f"  - {role} @ {org} ({start}–{end})")
    return "\n" + "\n".join(chunks)


def _format_education(items: list[dict]) -> str:
    if not items:
        return "(no education parsed)"
    chunks = []
    for it in items[:_MAX_EDUCATION]:
        inst = it.get("institution") or "?"
        deg = it.get("degree") or "?"
        end = it.get("end") or "?"
        chunks.append(f"  - {deg} @ {inst} ({end})")
    return "\n" + "\n".join(chunks)


def _user_prompt(
    vac: _VacInfo,
    candidate: CandidateMatch,
    raw_text: str,
) -> str:
    pj = candidate.parsed_json or {}
    description = vac.description[:_MAX_DESCRIPTION_CHARS]
    if len(vac.description) > _MAX_DESCRIPTION_CHARS:
        description += "… [truncated]"
    excerpt = raw_text[:_MAX_RAW_TEXT_CHARS]
    if len(raw_text) > _MAX_RAW_TEXT_CHARS:
        excerpt += "… [truncated]"
    return load_prompts().llm_rerank.user.format(
        title=vac.title or "(no title)",
        description=description or "(no description)",
        name=candidate.name or "Unknown",
        languages=", ".join(pj.get("languages_detected", [])) or "?",
        skills="; ".join((pj.get("skills") or [])[:_MAX_SKILLS]) or "(none)",
        experience=_format_experience(pj.get("experience", []) or []),
        education=_format_education(pj.get("education", []) or []),
        raw_text=excerpt or "(no resume text available)",
    )


def _system_prompt(is_ru: bool) -> str:
    p = load_prompts().llm_rerank
    return p.system_ru if is_ru else p.system_en


# ─── JSON parsing layers ────────────────────────────────────────────────────

def _parse_strict(raw: str) -> tuple[float, str]:
    parsed = LlmScore.model_validate_json(raw)
    return parsed.score, parsed.explanation


def _parse_lenient(raw: str) -> tuple[float, str]:
    m = _JSON_OBJECT_RE.search(raw)
    if not m:
        raise json.JSONDecodeError("no JSON object found", raw, 0)
    parsed = LlmScore.model_validate_json(m.group(0))
    return parsed.score, parsed.explanation


# ─── matcher ────────────────────────────────────────────────────────────────

class LlmMatcher(MatchingStrategy):
    name = "llm"

    def __init__(
        self,
        retriever: MatchingStrategy | None = None,
        shortlist_size: int | None = None,
    ) -> None:
        self.retriever = retriever or RrfMatcher()
        self.shortlist_size = shortlist_size or settings.matching.llm.shortlist_size
        self._sema = asyncio.Semaphore(settings.matching.llm.max_concurrent)

    async def match_by_vacancy(
        self,
        session: AsyncSession,
        job_id: int,
        top_k: int | None = None,
    ) -> list[CandidateMatch]:
        top_k = top_k or settings.matching.top_k
        vacancy = await session.get(Vacancy, job_id)
        if vacancy is None:
            raise VacancyNotFound(f"vacancy id={job_id} not found")
        shortlist = await self.retriever.match_by_vacancy(
            session, job_id, top_k=self.shortlist_size
        )
        if not shortlist:
            return []
        log.info(
            "llm_rerank_start",
            job_id=job_id,
            title=vacancy.title,
            shortlist=len(shortlist),
            top_k=top_k,
            retriever=self.retriever.name,
        )
        vac = _VacInfo(title=vacancy.title or "", description=vacancy.description or "")
        raw_texts = await self._fetch_raw_texts(session, shortlist)
        return await self._rerank_and_topk(vac, shortlist, raw_texts, top_k)

    async def match_by_text(
        self,
        session: AsyncSession,
        vacancy_text: str,
        top_k: int | None = None,
        vacancy_title: str | None = None,
    ) -> list[CandidateMatch]:
        top_k = top_k or settings.matching.top_k
        shortlist = await self.retriever.match_by_text(
            session,
            vacancy_text,
            top_k=self.shortlist_size,
            vacancy_title=vacancy_title,
        )
        if not shortlist:
            return []
        log.info(
            "llm_rerank_start_text",
            chars=len(vacancy_text),
            shortlist=len(shortlist),
            top_k=top_k,
            retriever=self.retriever.name,
        )
        vac = _VacInfo(title=vacancy_title or "", description=vacancy_text)
        raw_texts = await self._fetch_raw_texts(session, shortlist)
        return await self._rerank_and_topk(vac, shortlist, raw_texts, top_k)

    async def rerank(
        self,
        session: AsyncSession,
        vacancy_title: str,
        vacancy_description: str,
        shortlist: list[CandidateMatch],
        top_k: int | None = None,
    ) -> list[CandidateMatch]:
        """Public rerank API for custom retrievers (used by eval_llm).

        Takes a pre-built shortlist (any source) plus the vacancy's title and
        description, and returns the top-K candidates with LLM scores and
        explanations attached.
        """
        top_k = top_k or settings.matching.top_k
        if not shortlist:
            return []
        vac = _VacInfo(title=vacancy_title, description=vacancy_description)
        raw_texts = await self._fetch_raw_texts(session, shortlist)
        return await self._rerank_and_topk(vac, shortlist, raw_texts, top_k)

    # ── core rerank loop ────────────────────────────────────────────────────

    async def _fetch_raw_texts(
        self,
        session: AsyncSession,
        shortlist: list[CandidateMatch],
    ) -> dict[int, str]:
        """One round-trip to load raw_text for every candidate in the shortlist.

        Resume.csv candidates have minimal parsed_json (just `{'category': ...}`)
        but populated raw_text — the prompt falls back to raw_text when
        structured fields are missing.
        """
        ids = [c.candidate_id for c in shortlist]
        rows = (await session.execute(
            select(Candidate.id, Candidate.raw_text).where(Candidate.id.in_(ids))
        )).all()
        return {r.id: (r.raw_text or "") for r in rows}

    async def _rerank_and_topk(
        self,
        vac: _VacInfo,
        shortlist: list[CandidateMatch],
        raw_texts: dict[int, str],
        top_k: int,
    ) -> list[CandidateMatch]:
        reranked = await asyncio.gather(
            *(self._score_one(vac, c, raw_texts.get(c.candidate_id, "")) for c in shortlist)
        )
        reranked.sort(key=lambda m: -m.score)
        log.info(
            "llm_rerank_done",
            scored=len(reranked),
            with_explanation=sum(1 for m in reranked if m.explanation),
            top_score=reranked[0].score if reranked else None,
        )
        return reranked[:top_k]

    async def _score_one(
        self,
        vac: _VacInfo,
        c: CandidateMatch,
        raw_text: str,
    ) -> CandidateMatch:
        async with self._sema:
            try:
                score, explanation = await self._call_llm(vac, c, raw_text)
            except (APIError, LlmParseError, ValidationError, asyncio.TimeoutError) as e:
                log.warning(
                    "llm_score_degraded",
                    candidate_id=c.candidate_id,
                    err=type(e).__name__,
                    msg=str(e)[:200],
                )
                return c.model_copy(
                    update={"strategy": self.name, "explanation": None}
                )
        return c.model_copy(
            update={
                "score": score,
                "explanation": explanation,
                "strategy": self.name,
            }
        )

    async def _call_llm(
        self,
        vac: _VacInfo,
        candidate: CandidateMatch,
        raw_text: str,
    ) -> tuple[float, str]:
        client = await _get_client()
        full_text = f"{vac.title} {vac.description}"
        sys_prompt = _system_prompt(is_ru=_is_russian(full_text))
        user_prompt = _user_prompt(vac, candidate, raw_text)

        resp = await client.chat.completions.create(
            model=settings.models.llm.model,
            temperature=settings.models.llm.temperature,
            max_tokens=settings.models.llm.max_tokens,
            response_format={"type": "json_object"},
            # qwen3.x is a reasoning model — internal thinking tokens add 5-15 s
            # of latency per call. The scoring task doesn't need a long
            # deliberation loop; the anchored rubric in the system prompt is
            # enough. Cuts wall-clock by ~3.5×.
            extra_body={"enable_thinking": False},
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()

        for parser in (_parse_strict, _parse_lenient):
            try:
                return parser(raw)
            except (json.JSONDecodeError, ValidationError):
                continue
        raise LlmParseError(f"could not parse LLM output: {raw[:200]!r}")
