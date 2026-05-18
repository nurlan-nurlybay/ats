"""Evaluate the semantic matcher on the 24-category Resume.csv benchmark.

Pipeline (every step is idempotent — safe to re-run):
  1. Import 24 eval vacancies from data/eval/vacancies/ into `vacancies`.
  2. Embed them with bge-m3 (embed_vacancies covers NULL rows only).
  3. Bulk-encode all 2,484 Resume.csv resumes and upsert as `candidates`
     rows with source_file='resume_csv_{id}' and parsed_json={'category': ...}.
  4. For each eval vacancy, query top-K using pgvector cosine distance
     (scoped to Resume.csv candidates only — real CVs are excluded).
  5. Score against ground truth and report P@1 / P@3 / P@5 / P@10 / MRR.

Run:
    HF_HUB_OFFLINE=1 PYTHONPATH=src python -m ats.utils.eval_semantic
    HF_HUB_OFFLINE=1 PYTHONPATH=src python -m ats.utils.eval_semantic --skip-import
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from ats.core.logger import configure_logging, get_logger
from ats.db.base import SessionLocal
from ats.db.models import Candidate, Vacancy
from ats.ingestion.parser.embed import embed_texts
from ats.utils.embed_vacancies import embed_vacancies
from ats.utils.eval_common import (
    mrr,
    ndcg_at_k,
    p_at_k,
    print_metric_table,
    source_to_category,
)
from ats.utils.import_vacancies import import_vacancies

log = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_VACANCIES_DIR = PROJECT_ROOT / "data" / "eval" / "vacancies"
RESUME_CSV_PATH = PROJECT_ROOT / "data" / "eval" / "Resume.csv"
RESUME_CSV_PREFIX = "resume_csv_"


# ---------------------------------------------------------------------------
# import step
# ---------------------------------------------------------------------------

async def import_resume_csv_candidates(batch_size: int = 8) -> int:
    """Encode Resume.csv rows and insert into `candidates`. Idempotent.

    Uses max_seq_length=512 for the bulk encode to avoid CUDA OOM on the
    RTX 5060 (7.5 GiB). Resume.csv texts average ~600 tokens; 512 captures
    the most discriminative content (skills, job titles, summary).
    """
    df = pd.read_csv(RESUME_CSV_PATH, on_bad_lines="skip")
    log.info("resume_csv_loaded", rows=len(df))

    # Find which IDs are already in DB
    async with SessionLocal() as session:
        existing = {
            row[0]
            for row in (
                await session.execute(
                    select(Candidate.source_file).where(
                        Candidate.source_file.like(f"{RESUME_CSV_PREFIX}%")
                    )
                )
            ).all()
        }

    df["_source_file"] = RESUME_CSV_PREFIX + df["ID"].astype(str)
    new_df = df[~df["_source_file"].isin(existing)].reset_index(drop=True)

    if new_df.empty:
        log.info("resume_csv_already_imported", skipped=len(existing))
        return 0

    log.info("resume_csv_encoding", to_insert=len(new_df), batch_size=batch_size)
    texts = new_df["Resume_str"].fillna("").tolist()

    # Temporarily cap max_seq_length so each batch fits in 7.5 GiB VRAM
    from ats.ingestion.parser.embed import get_embedder
    embedder = get_embedder()
    original_max = embedder.max_seq_length
    embedder.max_seq_length = 512
    try:
        vectors = embed_texts(texts, batch_size=batch_size)
    finally:
        embedder.max_seq_length = original_max

    inserted = 0
    async with SessionLocal() as session:
        for i, row in new_df.iterrows():
            session.add(
                Candidate(
                    source_file=row["_source_file"],
                    raw_text=row["Resume_str"] or "",
                    embedding=vectors[int(i)],  # type: ignore[arg-type]
                    parsed_json={"category": row["Category"]},
                )
            )
            inserted += 1
            if inserted % 200 == 0:
                await session.flush()
                log.info("resume_csv_progress", inserted=inserted, total=len(new_df))
        await session.commit()

    log.info("resume_csv_imported", inserted=inserted)
    return inserted


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

async def run_evaluation(top_k: int = 10) -> None:
    """Run semantic matching against Resume.csv candidates, print metrics."""

    # 1. Load eval vacancies (only those with embeddings from eval_vacancies/)
    eval_filenames = {p.name for p in EVAL_VACANCIES_DIR.glob("*.json")}

    async with SessionLocal() as session:
        eval_rows = (
            await session.execute(
                select(Vacancy.id, Vacancy.title, Vacancy.source_filename, Vacancy.embedding)
                .where(Vacancy.embedding.is_not(None))
                .where(Vacancy.source_filename.in_(eval_filenames))
                .order_by(Vacancy.source_filename)
            )
        ).all()

    if not eval_rows:
        log.error("no_eval_vacancies_found — run without --skip-import first")
        return

    # 2. Count Resume.csv candidates per category (for random baseline + NDCG)
    async with SessionLocal() as session:
        cand_cats = (
            await session.execute(
                select(Candidate.parsed_json).where(
                    Candidate.source_file.like(f"{RESUME_CSV_PREFIX}%")
                )
            )
        ).scalars().all()

    category_counts: dict[str, int] = {}
    for pj in cand_cats:
        if pj and "category" in pj:
            cat = pj["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

    total_candidates = sum(category_counts.values())
    log.info(
        "eval_start",
        vacancies=len(eval_rows),
        total_candidates=total_candidates,
        top_k=top_k,
    )

    # 3. Evaluate each vacancy
    metric_rows: list[dict] = []

    async with SessionLocal() as session:
        for vac_id, title, source_filename, vac_embedding in eval_rows:
            expected_cat = source_to_category(source_filename)
            n_relevant = category_counts.get(expected_cat, 0)

            dist_col = Candidate.embedding.cosine_distance(list(vac_embedding)).label("dist")
            stmt = (
                select(Candidate.parsed_json, dist_col)
                .where(Candidate.source_file.like(f"{RESUME_CSV_PREFIX}%"))
                .where(Candidate.embedding.is_not(None))
                .order_by(dist_col)
                .limit(top_k)
            )
            top_results = (await session.execute(stmt)).all()

            hits = [
                (pj or {}).get("category") == expected_cat
                for pj, _dist in top_results
            ]
            top1_cat = (top_results[0][0] or {}).get("category", "?") if top_results else "—"

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


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

async def main_async(skip_import: bool) -> None:
    if not skip_import:
        log.info("step1_import_eval_vacancies")
        await import_vacancies(EVAL_VACANCIES_DIR)
        log.info("step2_embed_vacancies")
        await embed_vacancies()
        log.info("step3_import_resume_csv")
        await import_resume_csv_candidates()
    await run_evaluation()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate SemanticMatcher on Resume.csv (24-category benchmark)."
    )
    ap.add_argument(
        "--skip-import",
        action="store_true",
        help="Skip import/embed steps (safe to use after first run).",
    )
    args = ap.parse_args()
    configure_logging(level="INFO", json_output=False)
    asyncio.run(main_async(args.skip_import))


if __name__ == "__main__":
    main()
