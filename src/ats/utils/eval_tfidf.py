"""Evaluate the TF-IDF + LogisticRegression matcher on the 24-category Resume.csv benchmark.

Mirrors eval_semantic.py but uses sklearn TF-IDF cosine + a
LogisticRegression category-agreement boost (+0.10 if vacancy and
candidate share a predicted Resume.csv category). The boost is skipped
for Russian vacancies and Russian candidates (none here — Resume.csv is
English-only). Fits the vectorizer on Resume.csv only so the benchmark
is comparable across runs and doesn't drift with the DB.

Run:
    PYTHONPATH=src python -m ats.utils.eval_tfidf
    PYTHONPATH=src python -m ats.utils.eval_tfidf --skip-import
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select

from ats.core.config import settings
from ats.core.logger import configure_logging, get_logger
from ats.db.base import SessionLocal
from ats.db.models import Vacancy
from ats.ingestion.parser.keywords import _stopwords
from ats.matching.tfidf import CATEGORY_BOOST, _is_russian
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


async def run_evaluation(top_k: int = 10) -> None:
    # 1. Load Resume.csv — the candidate pool *and* the classifier's training set.
    df = pd.read_csv(RESUME_CSV_PATH, on_bad_lines="skip").reset_index(drop=True)
    resume_texts = df["Resume_str"].fillna("").tolist()
    resume_cats = df["Category"].tolist()
    log.info("eval_tfidf_loaded_csv", rows=len(df))

    # 2. Fit TF-IDF on Resume.csv only.
    vectorizer = TfidfVectorizer(
        max_features=settings.models.tfidf.max_features,
        ngram_range=settings.models.tfidf.ngram_range,
        min_df=settings.models.tfidf.min_df,
        sublinear_tf=True,
        lowercase=True,
        stop_words=_stopwords(),
    )
    cand_vecs = vectorizer.fit_transform(resume_texts)
    log.info("eval_tfidf_fit_done", vocab_size=len(vectorizer.vocabulary_))

    classifier = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1)
    classifier.fit(cand_vecs, resume_cats)
    cand_cats_pred = classifier.predict(cand_vecs)
    log.info("eval_tfidf_classifier_trained", classes=len(classifier.classes_))

    category_counts = dict(Counter(resume_cats))
    total_candidates = len(resume_cats)

    # 3. Load eval vacancies from DB.
    eval_filenames = {p.name for p in EVAL_VACANCIES_DIR.glob("*.json")}
    async with SessionLocal() as session:
        eval_rows = (
            await session.execute(
                select(Vacancy.title, Vacancy.description, Vacancy.source_filename)
                .where(Vacancy.source_filename.in_(eval_filenames))
                .order_by(Vacancy.source_filename)
            )
        ).all()

    if not eval_rows:
        log.error("no_eval_vacancies_found — run without --skip-import first")
        return

    log.info(
        "eval_tfidf_start",
        vacancies=len(eval_rows),
        total_candidates=total_candidates,
        top_k=top_k,
    )

    # 4. Run both evaluation settings.
    results = {}
    for run_name, use_boost in [("Standalone TF-IDF", False), ("TF-IDF + LogReg Boost", True)]:
        metric_rows = []
        for title, description, source_filename in eval_rows:
            expected_cat = source_to_category(source_filename)
            n_relevant = category_counts.get(expected_cat, 0)

            vac_text = f"{title}\n\n{description}"
            vac_vec = vectorizer.transform([vac_text])
            sims = cosine_similarity(vac_vec, cand_vecs).ravel()

            if use_boost and not _is_russian(vac_text):
                vac_cat = classifier.predict(vac_vec)[0]
                sims = sims + CATEGORY_BOOST * (cand_cats_pred == vac_cat)

            top_idx = np.argsort(-sims)[:top_k]
            hits = [resume_cats[i] == expected_cat for i in top_idx]
            top1_cat = resume_cats[top_idx[0]] if len(top_idx) else "—"

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
        
        # Calculate macro averages
        n_cats = len(metric_rows)
        averages = {}
        for metric in ("p@1", "p@3", "p@5", "p@10", "mrr", "ndcg@5"):
            averages[metric] = sum(row[metric] for row in metric_rows) / n_cats if n_cats else 0.0
        results[run_name] = (metric_rows, averages)

    # 5. Print the full detailed table for the Boosted run
    print("\nDetailed Per-Category Metrics for [TF-IDF + LogReg Boost]:")
    print_metric_table(results["TF-IDF + LogReg Boost"][0], total_candidates)

    # 6. Print the side-by-side comparison table
    standalone_avg = results["Standalone TF-IDF"][1]
    boosted_avg = results["TF-IDF + LogReg Boost"][1]

    print("\n" + "=" * 76)
    print("📊 COMPARISON: Standalone TF-IDF vs. TF-IDF + LogReg Category Boost")
    print("=" * 76)
    print(f"{'Metric':<10} {'Standalone TF-IDF':<22} {'TF-IDF + LogReg Boost':<24} {'Delta'}")
    print("-" * 76)
    for metric in ("p@1", "p@3", "p@5", "p@10", "mrr", "ndcg@5"):
        v_std = standalone_avg[metric]
        v_bst = boosted_avg[metric]
        diff = v_bst - v_std
        if v_std > 0:
            pct = (diff / v_std) * 100
            pct_str = f" ({pct:+.1f}%)"
        else:
            pct_str = ""
        print(f"{metric.upper():<10} {v_std:<22.4f} {v_bst:<24.4f} {diff:+.4f}{pct_str}")
    print("=" * 76)


async def main_async(skip_import: bool) -> None:
    if not skip_import:
        log.info("step1_import_eval_vacancies")
        await import_vacancies(EVAL_VACANCIES_DIR)
    await run_evaluation()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate TfidfMatcher on Resume.csv (24-category benchmark)."
    )
    ap.add_argument(
        "--skip-import",
        action="store_true",
        help="Skip the eval-vacancy import step (safe after the first run).",
    )
    args = ap.parse_args()
    configure_logging(level="INFO", json_output=False)
    asyncio.run(main_async(args.skip_import))


if __name__ == "__main__":
    main()
