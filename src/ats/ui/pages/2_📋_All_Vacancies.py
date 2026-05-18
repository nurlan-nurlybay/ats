"""Streamlit page: read-only overview of every vacancy in the DB.

A recruiter using the main `Match` page has to pick a vacancy from the
sidebar dropdown — there's no way to scan the full catalog. This page
plugs that gap.

Each vacancy renders as a card with title, experience requirement, source
filename, and a click-to-expand full description. A "🔍 Match candidates"
button stashes the chosen vacancy id in `st.session_state` and switches
back to the main page, which reads the preselect on rerun.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

from ats.ui.api_client import ApiClient, ApiError

st.set_page_config(
    page_title="All Vacancies — ATS",
    page_icon="📋",
    layout="wide",
)


@st.cache_resource
def _api() -> ApiClient:
    return ApiClient(os.environ.get("ATS_API_URL"))


@st.cache_data(ttl=60)
def _vacancies() -> list[dict[str, Any]]:
    return _api().list_vacancies()


@st.cache_data(ttl=300)
def _vacancy(vid: int) -> dict[str, Any]:
    return _api().get_vacancy(vid)


def main() -> None:
    st.title("📋 All Vacancies")
    st.caption(
        "Read-only overview of every job ad in the database. "
        "Click **🔍 Match candidates** on any card to jump to the matching page."
    )

    try:
        vacancies = _vacancies()
    except ApiError as e:
        st.error(f"Couldn't load vacancies: {e}")
        return

    if not vacancies:
        st.info("No vacancies in the database yet.")
        return

    st.markdown(f"**{len(vacancies)} vacancies** in the catalog.")

    for v in vacancies:
        with st.container(border=True):
            cols = st.columns([0.7, 0.2, 0.1])
            cols[0].markdown(f"### #{v['id']} · {v['title']}")
            cols[1].caption(
                f"**Experience:** {v.get('experience') or '—'}\n\n"
                f"`{v['source_filename']}`"
            )
            with cols[2]:
                if st.button(
                    "🔍 Match",
                    key=f"match_{v['id']}",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state["preselect_vacancy_id"] = v["id"]
                    # Streamlit ≥1.28 supports st.switch_page
                    try:
                        st.switch_page(str(Path(__file__).parent.parent / "app.py"))
                    except Exception:
                        st.info(
                            "Selected. Switch to the **Match** page in the "
                            "sidebar to see results."
                        )
            # Lazy-load description on expand to avoid 32 round-trips on page load.
            with st.expander("Description"):
                try:
                    full = _vacancy(v["id"])
                except ApiError as e:
                    st.error(f"Failed to load detail: {e}")
                else:
                    st.write(full["description"])


main()
