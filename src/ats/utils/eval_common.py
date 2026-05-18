"""Shared helpers for the Resume.csv 24-category benchmark.

Used by eval_semantic.py (pgvector cosine), eval_tfidf.py (TF-IDF + LR),
eval_llm.py (LLM rerank), and eval_rrf.py (RRF fusion). The metrics +
presentation helpers are at the top; the scoped retrievers (semantic,
tfidf) live at the bottom and are used by eval_llm/eval_rrf as
interchangeable shortlist sources for ablation studies.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from ats.matching.base import CandidateMatch


def source_to_category(source_filename: str) -> str:
    """Map eval vacancy filename to Resume.csv category label.

    '3_information_technology.json' → 'INFORMATION-TECHNOLOGY'
    '10_bpo.json'                   → 'BPO'
    '21_public_relations.json'      → 'PUBLIC-RELATIONS'
    """
    stem = source_filename.removesuffix(".json")
    _, cat_raw = stem.split("_", 1)
    return cat_raw.replace("_", "-").upper()


def p_at_k(hits: list[bool], k: int) -> float:
    return sum(hits[:k]) / k if k and hits else 0.0


def mrr(hits: list[bool]) -> float:
    for i, h in enumerate(hits, 1):
        if h:
            return 1.0 / i
    return 0.0


def ndcg_at_k(hits: list[bool], k: int, n_relevant: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, h in enumerate(hits[:k]) if h)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, n_relevant)))
    return dcg / idcg if idcg else 0.0


def print_metric_table(rows: list[dict], total_candidates: int) -> None:
    """Render the per-category table + macro-averages + random baselines.

    `rows` items must contain: category, title, n_relevant,
    p@1, p@3, p@5, p@10, mrr, ndcg@5, top1_cat.
    """
    rows = sorted(rows, key=lambda r: -r["p@5"])

    col_w = 26
    hdr = (
        f"{'Category':<{col_w}} {'P@1':>5} {'P@3':>5} {'P@5':>5} "
        f"{'P@10':>6} {'MRR':>6} {'NDCG@5':>7}  Top-1 returned"
    )
    sep = "─" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)
    for r in rows:
        print(
            f"{r['category']:<{col_w}} {r['p@1']:>5.2f} {r['p@3']:>5.2f} "
            f"{r['p@5']:>5.2f} {r['p@10']:>6.2f} {r['mrr']:>6.3f} "
            f"{r['ndcg@5']:>7.3f}  {r['top1_cat']}"
        )
    print(sep)

    n = len(rows)
    print(f"\nMacro-average  ({n} categories, {total_candidates} Resume.csv candidates):")
    for k in ("p@1", "p@3", "p@5", "p@10", "mrr", "ndcg@5"):
        avg = sum(r[k] for r in rows) / n if n else 0.0
        print(f"  {k:<10}: {avg:.4f}")

    # Random baseline: expected fraction of relevant docs at rank K under
    # uniform random sampling. The k * avg_pos / total formula is only valid
    # when all categories are evaluated (n_categories == |distinct labels|);
    # otherwise it overcounts. Clamp to [0, 1] either way — P@K can never
    # exceed 1.
    avg_pos = total_candidates / n if n else 1
    for k in (1, 5, 10):
        rand_pk = k * avg_pos / total_candidates if total_candidates else 0.0
        print(f"  Random P@{k:<2} : {min(1.0, rand_pk):.4f}")


# ─── Scoped-to-Resume.csv retrievers ──────────────────────────────────────
#
# Production matchers exclude `resume_csv_%` rows (see ats/matching/base.py:
# RESUME_CSV_PREFIX). For the benchmark we need the OPPOSITE — score the
# eval-only rows. These helpers do that, returning CandidateMatch objects
# that downstream eval scripts can pass to the LLM rerank or to RRF fusion.

RESUME_CSV_PREFIX = "resume_csv_"


async def shortlist_semantic(
    session: "AsyncSession",
    vac_embedding: list[float],
    n: int,
) -> list["CandidateMatch"]:
    """Top-N Resume.csv candidates by pgvector cosine on the given vacancy
    embedding."""
    from sqlalchemy import select

    from ats.db.models import Candidate
    from ats.matching.base import CandidateMatch

    dist_col = Candidate.embedding.cosine_distance(list(vac_embedding)).label("dist")
    rows = (
        await session.execute(
            select(
                Candidate.id, Candidate.name, Candidate.email,
                Candidate.source_file, Candidate.parsed_json, dist_col,
            )
            .where(Candidate.source_file.like(f"{RESUME_CSV_PREFIX}%"))
            .where(Candidate.embedding.is_not(None))
            .order_by(dist_col)
            .limit(n)
        )
    ).all()
    return [
        CandidateMatch(
            candidate_id=r.id,
            name=r.name,
            email=r.email,
            source_file=r.source_file,
            score=max(0.0, min(1.0, 1.0 - float(r.dist))),
            strategy="semantic",
            parsed_json=r.parsed_json,
        )
        for r in rows
    ]


# ─── TF-IDF scoped retriever (cached fit) ─────────────────────────────────
# eval_tfidf.py fits the vectorizer on Resume.csv once per process. We
# stash the fit in module-level state so eval_llm / eval_rrf can reuse it
# across vacancies without re-fitting.

_tfidf_state: tuple | None = None


def _fit_tfidf_resume_csv() -> tuple:
    """Fit a TF-IDF vectorizer + LR classifier on Resume.csv. Cached per process.

    The LR classifier matches production `TfidfMatcher` behavior: it
    contributes a +CATEGORY_BOOST signal when the vacancy's predicted
    category equals the candidate's predicted category. Without this,
    eval_rrf would fuse pure-cosine TF-IDF (weaker than production) with
    semantic, under-representing RRF's real-world performance.
    """
    global _tfidf_state
    if _tfidf_state is not None:
        return _tfidf_state

    from pathlib import Path

    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    from ats.core.config import settings
    from ats.ingestion.parser.keywords import _stopwords

    project_root = Path(__file__).resolve().parents[3]
    csv = project_root / "data" / "eval" / "Resume.csv"
    df = pd.read_csv(csv, on_bad_lines="skip").reset_index(drop=True)
    texts = df["Resume_str"].fillna("").tolist()
    cats = df["Category"].tolist()

    vectorizer = TfidfVectorizer(
        max_features=settings.models.tfidf.max_features,
        ngram_range=settings.models.tfidf.ngram_range,
        min_df=settings.models.tfidf.min_df,
        sublinear_tf=True,
        lowercase=True,
        stop_words=_stopwords(),
    )
    cand_vecs = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1)
    classifier.fit(cand_vecs, cats)
    cand_cats_pred = classifier.predict(cand_vecs)
    _tfidf_state = (vectorizer, cand_vecs, df, classifier, cand_cats_pred)
    return _tfidf_state


async def shortlist_tfidf(
    session: "AsyncSession",
    vac_text: str,
    n: int,
) -> list["CandidateMatch"]:
    """Top-N Resume.csv candidates by TF-IDF cosine + LR category boost.

    Mirrors production `TfidfMatcher`: cosine similarity plus a
    CATEGORY_BOOST when the vacancy's LR-predicted category matches the
    candidate's. Boost is skipped for Russian vacancies (LR is trained on
    English Resume.csv).

    Note: the in-memory vectorizer is keyed by row index in Resume.csv; we
    map back to the DB via `source_file = resume_csv_{ID}`.
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sqlalchemy import select

    from ats.db.models import Candidate
    from ats.matching.base import CandidateMatch
    from ats.matching.tfidf import CATEGORY_BOOST, _is_russian

    vectorizer, cand_vecs, df, classifier, cand_cats_pred = _fit_tfidf_resume_csv()
    vac_vec = vectorizer.transform([vac_text])
    sims = cosine_similarity(vac_vec, cand_vecs).ravel()
    if not _is_russian(vac_text):
        vac_cat = classifier.predict(vac_vec)[0]
        sims = sims + CATEGORY_BOOST * (cand_cats_pred == vac_cat)
    top_idx = np.argsort(-sims)[:n]
    source_files = [f"{RESUME_CSV_PREFIX}{df.iloc[int(i)]['ID']}" for i in top_idx]

    # Resolve back to DB rows for the name/email/parsed_json metadata.
    rows = (await session.execute(
        select(
            Candidate.id, Candidate.name, Candidate.email,
            Candidate.source_file, Candidate.parsed_json,
        ).where(Candidate.source_file.in_(source_files))
    )).all()
    by_source = {r.source_file: r for r in rows}

    out: list[CandidateMatch] = []
    for rank, i in enumerate(top_idx):
        sf = source_files[rank]
        r = by_source.get(sf)
        if r is None:
            continue
        out.append(CandidateMatch(
            candidate_id=r.id, name=r.name, email=r.email,
            source_file=r.source_file,
            score=max(0.0, min(1.0, float(sims[int(i)]))),
            strategy="tfidf",
            parsed_json=r.parsed_json,
        ))
    return out


async def shortlist_rrf(
    session: "AsyncSession",
    vac_text: str,
    vac_embedding: list[float],
    n: int,
    *,
    k: int = 60,
    over_fetch: int = 3,
) -> list["CandidateMatch"]:
    """Fuse semantic + tfidf scoped rankings via RRF, return top-n."""
    from ats.matching.rrf import RrfMatcher

    fan = n * over_fetch
    semantic_ranking = await shortlist_semantic(session, vac_embedding, fan)
    tfidf_ranking = await shortlist_tfidf(session, vac_text, fan)
    fuser = RrfMatcher(k=k)
    return fuser._fuse([semantic_ranking, tfidf_ranking], n)
