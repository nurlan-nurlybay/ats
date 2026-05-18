"""Evaluate the RRF matcher on the 24-category Resume.csv benchmark.

Pipeline (per eval vacancy):
  1. pgvector cosine top-N scoped to Resume.csv → semantic ranking
  2. TF-IDF cosine top-N scoped to Resume.csv   → tfidf ranking
  3. Reciprocal Rank Fusion of the two rankings → top-K
  4. Score against the Resume.csv category label

The two scoped rankings reuse the shortlist helpers from eval_common; the
fusion uses RrfMatcher._fuse (same code path as the production matcher).

Run:
    PYTHONPATH=src python -m ats.utils.eval_rrf
    PYTHONPATH=src python -m ats.utils.eval_rrf --skip-import
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from pathlib import Path

from sqlalchemy import select

from ats.core.logger import configure_logging, get_logger
from ats.db.base import SessionLocal
from ats.db.models import Candidate, Vacancy
from ats.matching.rrf import DEFAULT_K, RrfMatcher
from ats.utils.embed_vacancies import embed_vacancies
from ats.utils.eval_common import (
    RESUME_CSV_PREFIX,
    mrr,
    ndcg_at_k,
    p_at_k,
    print_metric_table,
    shortlist_semantic,
    shortlist_tfidf,
    source_to_category,
)
from ats.utils.import_vacancies import import_vacancies

log = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_VACANCIES_DIR = PROJECT_ROOT / "data" / "eval" / "vacancies"


async def run_evaluation(top_k: int = 10, fan: int = 30, k: int = DEFAULT_K) -> None:
    eval_filenames = {p.name for p in EVAL_VACANCIES_DIR.glob("*.json")}
    async with SessionLocal() as session:
        eval_rows = (await session.execute(
            select(
                Vacancy.id, Vacancy.title, Vacancy.description,
                Vacancy.source_filename, Vacancy.embedding,
            )
            .where(Vacancy.embedding.is_not(None))
            .where(Vacancy.source_filename.in_(eval_filenames))
            .order_by(Vacancy.source_filename)
        )).all()

    if not eval_rows:
        log.error("no_eval_vacancies_found — run without --skip-import first")
        return

    async with SessionLocal() as session:
        cand_cats = (await session.execute(
            select(Candidate.parsed_json)
            .where(Candidate.source_file.like(f"{RESUME_CSV_PREFIX}%"))
        )).scalars().all()
    category_counts = Counter(
        pj["category"] for pj in cand_cats if pj and "category" in pj
    )
    total_candidates = sum(category_counts.values())
    log.info(
        "eval_rrf_start",
        vacancies=len(eval_rows),
        total_candidates=total_candidates,
        fan=fan, k=k, top_k=top_k,
    )

    fuser = RrfMatcher(k=k)
    metric_rows: list[dict] = []

    async with SessionLocal() as session:
        for vac_id, title, description, source_filename, vac_embedding in eval_rows:
            expected_cat = source_to_category(source_filename)
            n_relevant = category_counts.get(expected_cat, 0)

            vac_text = f"{title}\n\n{description}"
            sem_ranking = await shortlist_semantic(session, vac_embedding, fan)
            tfi_ranking = await shortlist_tfidf(session, vac_text, fan)
            fused = fuser._fuse([sem_ranking, tfi_ranking], top_k)

            hits = [
                (r.parsed_json or {}).get("category") == expected_cat
                for r in fused
            ]
            top1_cat = (
                (fused[0].parsed_json or {}).get("category", "?")
                if fused else "—"
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

    print_metric_table(metric_rows, total_candidates)


async def main_async(*, skip_import: bool) -> None:
    if not skip_import:
        log.info("step1_import_eval_vacancies")
        await import_vacancies(EVAL_VACANCIES_DIR)
        log.info("step2_embed_vacancies")
        await embed_vacancies()
    await run_evaluation()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate RrfMatcher on Resume.csv (24-category benchmark)."
    )
    ap.add_argument("--skip-import", action="store_true")
    args = ap.parse_args()
    configure_logging(level="INFO", json_output=False)
    asyncio.run(main_async(skip_import=args.skip_import))


if __name__ == "__main__":
    main()
