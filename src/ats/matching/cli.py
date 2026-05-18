"""CLI driver for the matching engine.

Usage:
    python -m ats.matching                              # rank all 8 vacancies
    python -m ats.matching --job 7                      # single vacancy
    python -m ats.matching --text "..." [--title "..."] # ad-hoc text
    python -m ats.matching --strategy semantic --top-k 5
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ats.core.config import settings
from ats.core.logger import configure_logging, get_logger
from ats.db.base import SessionLocal
from ats.db.models import Vacancy
from ats.matching import MATCHERS, CandidateMatch, MatchingStrategy

log = get_logger(__name__)


def _format_match(rank: int, m: CandidateMatch) -> str:
    short = Path(m.source_file).name
    email = f"<{m.email}>" if m.email else ""
    head = f"  {rank}. ({m.score:.3f}) {m.name or '?'} {email}  {short}"
    if m.explanation:
        return head + f"\n       → {m.explanation}"
    return head


async def _rank_one_vacancy(
    session: AsyncSession,
    matcher: MatchingStrategy,
    job_id: int,
    title: str,
    top_k: int,
) -> None:
    results = await matcher.match_by_vacancy(session, job_id, top_k=top_k)
    print(f"\nVacancy {job_id}: {title}")
    if not results:
        print("  (no candidates with embeddings)")
        return
    for i, m in enumerate(results, 1):
        print(_format_match(i, m))


async def _rank_text(
    session: AsyncSession,
    matcher: MatchingStrategy,
    text: str,
    title: str | None,
    top_k: int,
) -> None:
    results = await matcher.match_by_text(session, text, top_k=top_k, vacancy_title=title)
    print(f"\nAd-hoc text ({'' if title is None else title + ' / '}{len(text)} chars)")
    if not results:
        print("  (no candidates with embeddings)")
        return
    for i, m in enumerate(results, 1):
        print(_format_match(i, m))


async def _run(args: argparse.Namespace) -> None:
    if args.strategy not in MATCHERS:
        raise SystemExit(
            f"unknown strategy {args.strategy!r}; available: {sorted(MATCHERS)}"
        )
    matcher = MATCHERS[args.strategy]
    top_k = args.top_k or settings.matching.top_k

    async with SessionLocal() as session:
        if args.text:
            await _rank_text(session, matcher, args.text, args.title, top_k)
            return

        if args.job is not None:
            vac = await session.get(Vacancy, args.job)
            title = vac.title if vac else f"id={args.job}"
            await _rank_one_vacancy(session, matcher, args.job, title, top_k)
            return

        vacancies = (
            await session.execute(select(Vacancy.id, Vacancy.title).order_by(Vacancy.id))
        ).all()
        for v in vacancies:
            await _rank_one_vacancy(session, matcher, v.id, v.title, top_k)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rank candidates against a vacancy (or all vacancies)."
    )
    ap.add_argument("--job", type=int, default=None, help="Single vacancy id.")
    ap.add_argument("--text", type=str, default=None, help="Ad-hoc vacancy text.")
    ap.add_argument("--title", type=str, default=None, help="Optional title for --text.")
    ap.add_argument(
        "--strategy",
        type=str,
        default=settings.matching.default_strategy,
        help=f"Matcher name (default: {settings.matching.default_strategy}).",
    )
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()
    configure_logging(level="INFO", json_output=False)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
