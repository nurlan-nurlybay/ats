"""Streamlit render helpers — candidate cards, parsed_json drill-down, score bar.

Kept thin: each function takes plain dicts (the JSON shape returned by FastAPI)
and writes to the streamlit canvas. No global state, no caching, no API calls.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st


# ── score bar ────────────────────────────────────────────────────────────────

def score_bar(score: float) -> str:
    """●●●●○ 0.78 — 5 buckets at 0.2 thresholds."""
    filled = max(0, min(5, int(round(score * 5))))
    return "●" * filled + "○" * (5 - filled) + f"  {score:.2f}"


def strategy_badge(strategy: str) -> str:
    emoji = {"semantic": "🧠", "tfidf": "📊", "llm": "✨"}.get(strategy, "🔹")
    return f"{emoji} `{strategy}`"


# ── vacancy preview ──────────────────────────────────────────────────────────

def render_vacancy_preview(v: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown(f"### {v.get('title') or '(untitled)'}")
        cols = st.columns([1, 3])
        with cols[0]:
            if v.get("experience"):
                st.caption(f"**Experience:** {v['experience']}")
        with cols[1]:
            st.caption(f"**Source:** `{v.get('source_filename') or '?'}`")
        descr = v.get("description") or ""
        if len(descr) > 600:
            with st.expander(f"Description ({len(descr)} chars) — click to expand"):
                st.write(descr)
        else:
            st.write(descr)


# ── parsed_json drill-down ───────────────────────────────────────────────────

def _render_skills(skills: list[str]) -> None:
    if not skills:
        st.caption("_(no skills parsed)_")
        return
    chips = " ".join(f"`{s}`" for s in skills[:30])
    st.markdown(chips)


def _render_experience(items: list[dict]) -> None:
    if not items:
        st.caption("_(no experience parsed)_")
        return
    rows = [
        {
            "Role": it.get("role") or "?",
            "Org": it.get("organization") or "?",
            "Period": f"{it.get('start') or '?'} – {it.get('end') or '?'}",
        }
        for it in items[:8]
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_education(items: list[dict]) -> None:
    if not items:
        st.caption("_(no education parsed)_")
        return
    rows = [
        {
            "Degree": it.get("degree") or "?",
            "Institution": it.get("institution") or "?",
            "Graduated": it.get("end") or "?",
        }
        for it in items[:5]
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)


def render_parsed_json(pj: dict[str, Any] | None, raw_excerpt: str | None) -> None:
    """Three tabs inside the expander: Skills | Experience | Education | Raw."""
    if pj is None and not raw_excerpt:
        st.caption("_(no parsed data — candidate may be from Resume.csv import)_")
        return
    pj = pj or {}

    # If parsed_json only carries a category (Resume.csv import), say so plainly.
    if set(pj.keys()) <= {"category"}:
        if pj.get("category"):
            st.caption(f"**Source category:** `{pj['category']}` (Resume.csv label)")
        if raw_excerpt:
            with st.expander("Raw resume text"):
                st.text(raw_excerpt)
        return

    tabs = st.tabs(["🛠️ Skills", "💼 Experience", "🎓 Education", "📝 Raw"])
    with tabs[0]:
        _render_skills(pj.get("skills") or [])
    with tabs[1]:
        _render_experience(pj.get("experience") or [])
    with tabs[2]:
        _render_education(pj.get("education") or [])
    with tabs[3]:
        if raw_excerpt:
            st.text(raw_excerpt)
        else:
            st.caption("_(raw text not available)_")


# ── candidate card ───────────────────────────────────────────────────────────

def render_candidate_card(
    rank: int,
    match: dict[str, Any],
    detail: dict[str, Any] | None,
) -> None:
    with st.container(border=True):
        short = Path(match.get("source_file") or "").name or "(no source file)"
        name = match.get("name") or "Not Found"
        email = match.get("email")
        strategy = match.get("strategy", "?")
        score = float(match.get("score", 0.0))

        # Header row: rank · score bar · strategy badge
        cols = st.columns([1, 2, 2])
        with cols[0]:
            st.markdown(f"### #{rank}")
        with cols[1]:
            st.markdown(score_bar(score))
        with cols[2]:
            st.markdown(strategy_badge(strategy))

        # Identity row
        ident = f"**{name}**"
        if email:
            ident += f"  ·  `{email}`"
        ident += f"  ·  `{short}`"
        st.markdown(ident)

        # LLM explanation (only present for strategy=llm)
        if match.get("explanation"):
            st.info(f"💡 {match['explanation']}")

        # Drill-down: parsed_json + raw text excerpt
        with st.expander("▾ Show details"):
            if detail is None:
                st.caption("_(candidate detail unavailable)_")
            else:
                render_parsed_json(
                    detail.get("parsed_json"),
                    detail.get("raw_text_excerpt"),
                )
