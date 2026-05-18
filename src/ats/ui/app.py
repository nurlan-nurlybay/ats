"""Streamlit single-page UI for the ATS.

Sidebar: mode (vacancy vs ad-hoc), vacancy picker / text input, strategy
radio, top-K slider, run button.

Main: vacancy preview (if applicable) + top-K candidate cards with LLM
explanations and parsed_json drill-downs.

Run locally:
    streamlit run src/ats/ui/app.py
In docker:
    streamlit run src/ats/ui/app.py --server.address 0.0.0.0 --server.port 8501
"""
from __future__ import annotations

import os
from typing import Any

import streamlit as st

from ats.ui.api_client import ApiClient, ApiError
from ats.ui.components import (
    render_candidate_card,
    render_vacancy_preview,
)

st.set_page_config(
    page_title="AI Recruiting Agent",
    page_icon="🤖",
    layout="wide",
)


# ─── api client ────────────────────────────────────────────────────────────

@st.cache_resource
def _api() -> ApiClient:
    return ApiClient(os.environ.get("ATS_API_URL"))


@st.cache_data(ttl=60)
def _vacancies() -> list[dict[str, Any]]:
    return _api().list_vacancies()


@st.cache_data(ttl=300)
def _vacancy(vid: int) -> dict[str, Any]:
    return _api().get_vacancy(vid)


@st.cache_data(ttl=300)
def _candidate(cid: int) -> dict[str, Any]:
    return _api().get_candidate(cid)


# ─── sidebar ───────────────────────────────────────────────────────────────

def _sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.markdown("## 🤖 AI Recruiting Agent")
        st.caption("Stage 4 — FastAPI + Streamlit + Docker")
        st.divider()

        mode = st.radio(
            "Mode",
            ["By vacancy", "Ad-hoc text"],
            help="Pick a stored vacancy or paste a job description.",
        )

        if mode == "By vacancy":
            try:
                vacs = _vacancies()
            except ApiError as e:
                st.error(f"Couldn't load vacancies: {e}")
                vacs = []
            if not vacs:
                st.warning("No vacancies in the database yet.")
                chosen_id = None
                text = title = None
            else:
                # Honor a preselect set by the All Vacancies page (clicked
                # the "🔍 Match" button there).
                preselect_id = st.session_state.pop("preselect_vacancy_id", None)
                default_idx = 0
                if preselect_id is not None:
                    for i, v in enumerate(vacs):
                        if v["id"] == preselect_id:
                            default_idx = i
                            break
                chosen = st.selectbox(
                    "Vacancy",
                    options=vacs,
                    index=default_idx,
                    format_func=lambda v: f"{v['id']}. {v['title']}",
                )
                chosen_id = chosen["id"]
                text = title = None
        else:
            chosen_id = None
            title = st.text_input("Title (optional)", placeholder="Senior Python Engineer")
            text = st.text_area(
                "Vacancy text",
                height=200,
                placeholder=(
                    "Paste a job description here. The matcher will search "
                    "all parsed candidates against this text."
                ),
            )

        st.divider()
        strategy = st.radio(
            "Matching strategy",
            ["semantic", "tfidf", "rrf", "llm"],
            captions=[
                "bge-m3 + pgvector cosine",
                "TF-IDF + LR classifier boost",
                "Reciprocal Rank Fusion (semantic + tfidf)",
                "Qwen rerank with explanations",
            ],
            help="Semantic is fastest. LLM produces natural-language explanations.",
        )
        # When strategy=llm, let the user pick which retriever feeds the
        # shortlist to the LLM rerank. RRF is the recommended default.
        retriever = None
        if strategy == "llm":
            retriever = st.selectbox(
                "LLM shortlist retriever",
                ["rrf", "semantic", "tfidf"],
                index=0,
                help=(
                    "RRF fuses semantic + tfidf rankings before the LLM "
                    "reranks them. Pick `semantic` or `tfidf` for ablation."
                ),
            )
        top_k = st.slider("Top-K", min_value=1, max_value=10, value=5)
        run = st.button("▶ Run match", type="primary", use_container_width=True)

        st.divider()
        with st.expander("ℹ About"):
            st.caption(
                "Four matching strategies side-by-side, with bilingual (RU/EN) "
                "support. RRF fuses semantic + TF-IDF; the LLM strategy "
                "reranks a shortlist (RRF by default) and adds natural-language "
                "explanations from Qwen."
            )

    return {
        "mode": mode,
        "chosen_id": chosen_id,
        "text": text,
        "title": title,
        "strategy": strategy,
        "retriever": retriever,
        "top_k": top_k,
        "run": run,
    }


# ─── main pane ─────────────────────────────────────────────────────────────

def _show_recommendations(resp: dict[str, Any]) -> None:
    count = resp.get("count", 0)
    strategy = resp.get("strategy", "?")
    st.subheader(f"Top-{count} candidates  ·  via `{strategy}`")
    if count == 0:
        st.warning("No candidates returned. The database may be empty.")
        return
    for rank, match in enumerate(resp["results"], 1):
        cid = match.get("candidate_id")
        detail = None
        if cid is not None:
            try:
                detail = _candidate(cid)
            except ApiError:
                detail = None
        render_candidate_card(rank, match, detail)


def _run_match(state: dict[str, Any]) -> None:
    api = _api()
    spinner = f"Matching via `{state['strategy']}` (this can take ~15s for LLM)…"

    if state["mode"] == "By vacancy":
        if state["chosen_id"] is None:
            st.error("Select a vacancy first.")
            return
        try:
            v = _vacancy(state["chosen_id"])
            render_vacancy_preview(v)
        except ApiError as e:
            st.error(f"Couldn't load vacancy: {e}")
            return

        with st.spinner(spinner):
            try:
                resp = api.get_recommendations(
                    job_id=state["chosen_id"],
                    strategy=state["strategy"],
                    top_k=state["top_k"],
                    retriever=state.get("retriever"),
                )
            except ApiError as e:
                st.error(f"API call failed: {e}")
                return
    else:
        if not (state.get("text") or "").strip():
            st.error("Enter some vacancy text first.")
            return
        with st.spinner(spinner):
            try:
                resp = api.get_recommendations(
                    text=state["text"],
                    title=state.get("title") or None,
                    strategy=state["strategy"],
                    top_k=state["top_k"],
                    retriever=state.get("retriever"),
                )
            except ApiError as e:
                st.error(f"API call failed: {e}")
                return

    _show_recommendations(resp)


def main() -> None:
    state = _sidebar()
    if not state["run"]:
        st.title("🤖 AI Recruiting Agent")
        st.markdown(
            "Pick a vacancy (or paste one) on the left, choose a matching "
            "strategy, then press **▶ Run match**."
        )
        st.markdown(
            "**Strategies:**\n"
            "- 🧠 **semantic** — dense embedding similarity (bge-m3 + pgvector). "
            "Fast, multilingual, best for top-1 precision.\n"
            "- 📊 **tfidf** — sparse term overlap + LR classifier boost. "
            "Stronger recall at depth (P@5+).\n"
            "- ✨ **llm** — Qwen rerank of the semantic top-15, with a "
            "natural-language explanation for each candidate."
        )
        # Quick health probe so the user knows whether the backend is reachable.
        try:
            _api().health()
            st.success(f"API reachable at `{_api().base_url}` ✓")
        except ApiError as e:
            st.error(f"API unreachable at `{_api().base_url}`: {e}")
        return

    _run_match(state)


if __name__ == "__main__":
    main()
