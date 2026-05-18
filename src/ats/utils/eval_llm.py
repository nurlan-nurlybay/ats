"""Evaluate the LLM rerank matcher on the 24-category Resume.csv benchmark.

Pipeline (per eval vacancy):
  1. pgvector cosine query, scoped to source_file LIKE 'resume_csv_%' →
     top-N shortlist (fair vs eval_semantic, isolates the rerank lift).
  2. Call LlmMatcher.rerank(...) → top-K with LLM scores + explanations.
  3. Score against the Resume.csv category label.

Cost: 15 × 24 ≈ 360 LLM calls at ~3.3 s each, batched 5-wide via the matcher's
semaphore → ~4 minutes wall-clock per full run. Use --subset for cheap smoke
tests, --qualitative to dump the explanations alongside the metrics.

Run:
    PYTHONPATH=src python -m ats.utils.eval_llm                    # full quantitative
    PYTHONPATH=src python -m ats.utils.eval_llm --skip-import      # skip vacancy reimport
    PYTHONPATH=src python -m ats.utils.eval_llm --subset 5         # 5 vacancies, ~$0.05
    PYTHONPATH=src python -m ats.utils.eval_llm --qualitative      # dump explanations
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from pathlib import Path

from sqlalchemy import select

from ats.core.config import settings
from ats.core.logger import configure_logging, get_logger
from ats.db.base import SessionLocal
from ats.db.models import Candidate, Vacancy
from ats.matching import LlmMatcher
from ats.matching.base import CandidateMatch
from ats.utils.embed_vacancies import embed_vacancies
from ats.utils.eval_common import (
    mrr,
    ndcg_at_k,
    p_at_k,
    print_metric_table,
    shortlist_rrf,
    shortlist_semantic,
    shortlist_tfidf,
    source_to_category,
)
from ats.utils.import_vacancies import import_vacancies

log = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_VACANCIES_DIR = PROJECT_ROOT / "data" / "eval" / "vacancies"
RESUME_CSV_PREFIX = "resume_csv_"


async def _shortlist(
    session,
    *,
    retriever: str,
    vac_text: str,
    vac_embedding: list[float],
    n: int,
) -> list[CandidateMatch]:
    """Build the shortlist via the chosen retriever, scoped to Resume.csv."""
    if retriever == "semantic":
        return await shortlist_semantic(session, vac_embedding, n)
    if retriever == "tfidf":
        return await shortlist_tfidf(session, vac_text, n)
    if retriever == "rrf":
        return await shortlist_rrf(session, vac_text, vac_embedding, n)
    raise ValueError(f"unknown retriever {retriever!r}")


async def run_evaluation(
    *,
    subset: int | None,
    qualitative: bool,
    retriever: str = "semantic",
    top_k: int = 10,
) -> None:
    matcher = LlmMatcher()  # rerank logic only; we build shortlist ourselves
    shortlist_n = matcher.shortlist_size
    log.info("eval_llm_retriever", retriever=retriever, shortlist=shortlist_n)

    # Load eval vacancies (with embeddings).
    eval_filenames = {p.name for p in EVAL_VACANCIES_DIR.glob("*.json")}
    async with SessionLocal() as session:
        eval_rows = (
            await session.execute(
                select(
                    Vacancy.id,
                    Vacancy.title,
                    Vacancy.description,
                    Vacancy.source_filename,
                    Vacancy.embedding,
                )
                .where(Vacancy.embedding.is_not(None))
                .where(Vacancy.source_filename.in_(eval_filenames))
                .order_by(Vacancy.source_filename)
            )
        ).all()

    if not eval_rows:
        log.error("no_eval_vacancies_found — run without --skip-import first")
        return

    if subset is not None:
        eval_rows = list(eval_rows)[:subset]

    # Per-category counts for the NDCG denominator (and random baseline).
    async with SessionLocal() as session:
        cand_cats = (
            await session.execute(
                select(Candidate.parsed_json).where(
                    Candidate.source_file.like(f"{RESUME_CSV_PREFIX}%")
                )
            )
        ).scalars().all()
    category_counts = Counter(
        pj["category"] for pj in cand_cats if pj and "category" in pj
    )
    total_candidates = sum(category_counts.values())
    log.info(
        "eval_llm_start",
        vacancies=len(eval_rows),
        total_candidates=total_candidates,
        shortlist=shortlist_n,
        top_k=top_k,
    )

    metric_rows: list[dict] = []
    qualitative_dump: list[tuple[str, str, list[CandidateMatch]]] = []

    async with SessionLocal() as session:
        for vac_id, title, description, source_filename, vac_embedding in eval_rows:
            expected_cat = source_to_category(source_filename)
            n_relevant = category_counts.get(expected_cat, 0)

            vac_text = f"{title}\n\n{description}"
            shortlist = await _shortlist(
                session,
                retriever=retriever,
                vac_text=vac_text,
                vac_embedding=vac_embedding,
                n=shortlist_n,
            )
            reranked = await matcher.rerank(
                session,
                vacancy_title=title or "",
                vacancy_description=description or "",
                shortlist=shortlist,
                top_k=top_k,
            )

            hits = [
                (r.parsed_json or {}).get("category") == expected_cat
                for r in reranked
            ]
            top1_cat = (
                (reranked[0].parsed_json or {}).get("category", "?")
                if reranked
                else "—"
            )

            metric_rows.append({
                "category": expected_cat,
                "title": title,
                "n_relevant": n_relevant,
                "p@1":    p_at_k(hits, 1),
                "p@3":    p_at_k(hits, 3),
                "p@5":    p_at_k(hits, 5),
                "p@10":   p_at_k(hits, 10),
                "mrr":    mrr(hits),
                "ndcg@5": ndcg_at_k(hits, 5, n_relevant),
                "top1_cat": top1_cat,
            })
            if qualitative:
                qualitative_dump.append((expected_cat, title or "?", reranked[:5]))

    print_metric_table(metric_rows, total_candidates)

    if qualitative:
        print("\n" + "═" * 80)
        print("QUALITATIVE: top-5 LLM-reranked candidates with explanations")
        print("═" * 80)
        for expected_cat, title, top5 in qualitative_dump:
            print(f"\n┌─ {expected_cat}  —  {title}")
            for i, m in enumerate(top5, 1):
                cat = (m.parsed_json or {}).get("category", "?")
                hit = "✓" if cat == expected_cat else "✗"
                print(f"│ {i}. [{hit}] score={m.score:.2f}  Resume.csv→{cat}")
                if m.explanation:
                    print(f"│    {m.explanation}")
                else:
                    print(f"│    (no explanation — LLM call failed)")
            print("└─")


async def main_async(
    *,
    skip_import: bool,
    subset: int | None,
    qualitative: bool,
    retriever: str,
) -> None:
    if not skip_import:
        log.info("step1_import_eval_vacancies")
        await import_vacancies(EVAL_VACANCIES_DIR)
        log.info("step2_embed_vacancies")
        await embed_vacancies()
    await run_evaluation(
        subset=subset, qualitative=qualitative, retriever=retriever,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate LlmMatcher on Resume.csv (24-category benchmark)."
    )
    ap.add_argument("--skip-import", action="store_true")
    ap.add_argument("--subset", type=int, default=None,
                    help="Evaluate only the first N vacancies (cheap smoke test).")
    ap.add_argument("--qualitative", action="store_true",
                    help="Dump top-5 candidates with explanations after the metrics.")
    ap.add_argument(
        "--retriever",
        choices=["semantic", "tfidf", "rrf"],
        default="semantic",
        help="Which retriever feeds the LLM rerank shortlist (default: semantic).",
    )
    args = ap.parse_args()
    configure_logging(level="INFO", json_output=False)
    asyncio.run(main_async(
        skip_import=args.skip_import,
        subset=args.subset,
        qualitative=args.qualitative,
        retriever=args.retriever,
    ))


if __name__ == "__main__":
    main()
