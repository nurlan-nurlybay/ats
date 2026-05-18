"""Streamlit page: CRUD admin for vacancies and candidates.

Two tabs:
  📋 Vacancies — create / edit / delete (text-based form)
  👤 Candidates — upload PDF/DOCX, list, delete

All operations hit the FastAPI surface; nothing here talks to Postgres
directly. The page is intentionally lean — no validation duplication.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

from ats.ui.api_client import ApiClient, ApiError

st.set_page_config(
    page_title="Manage — ATS",
    page_icon="⚙️",
    layout="wide",
)


@st.cache_resource
def _api() -> ApiClient:
    return ApiClient(os.environ.get("ATS_API_URL"))


def _refresh() -> None:
    """Drop the page-level caches and rerun."""
    st.cache_data.clear()
    st.rerun()


# ─── Vacancies tab ──────────────────────────────────────────────────────────

def _vacancies_tab() -> None:
    api = _api()
    st.subheader("Add a new vacancy")
    with st.form("vac_create", clear_on_submit=True):
        title = st.text_input("Title *", placeholder="Senior Python Engineer")
        experience = st.text_input(
            "Experience (free-text, e.g. '2–4 years')", placeholder=""
        )
        description = st.text_area(
            "Description *",
            height=200,
            placeholder="Responsibilities, stack, must-haves, nice-to-haves…",
        )
        if st.form_submit_button("➕ Create", type="primary"):
            if not title.strip() or not description.strip():
                st.error("Title and description are required.")
            else:
                try:
                    v = api.create_vacancy(
                        title=title.strip(),
                        description=description.strip(),
                        experience=(experience.strip() or None),
                    )
                    st.success(f"Created #{v['id']} — {v['title']}.")
                    _refresh()
                except ApiError as e:
                    st.error(str(e))

    st.divider()
    st.subheader("Existing vacancies")
    try:
        vacancies = api.list_vacancies()
    except ApiError as e:
        st.error(str(e))
        return
    if not vacancies:
        st.info("No vacancies yet.")
        return

    edit_id = st.session_state.get("edit_vac_id")
    for v in vacancies:
        with st.container(border=True):
            cols = st.columns([0.55, 0.25, 0.10, 0.10])
            cols[0].markdown(f"**#{v['id']} · {v['title']}**")
            cols[1].caption(
                f"{v.get('experience') or '—'} · `{v['source_filename']}`"
            )
            if cols[2].button("✏️ Edit", key=f"edit_{v['id']}"):
                st.session_state["edit_vac_id"] = v["id"]
                st.rerun()
            if cols[3].button("🗑️", key=f"del_v_{v['id']}", type="secondary"):
                try:
                    api.delete_vacancy(v["id"])
                    st.toast(f"Deleted vacancy #{v['id']}", icon="🗑️")
                    _refresh()
                except ApiError as e:
                    st.error(str(e))

            if edit_id == v["id"]:
                try:
                    full = api.get_vacancy(v["id"])
                except ApiError as e:
                    st.error(str(e))
                    continue
                with st.form(f"edit_form_{v['id']}", clear_on_submit=False):
                    e_title = st.text_input("Title", value=full["title"])
                    e_exp = st.text_input(
                        "Experience", value=full.get("experience") or ""
                    )
                    e_desc = st.text_area(
                        "Description", value=full["description"], height=200,
                    )
                    save, cancel = st.columns(2)
                    if save.form_submit_button("💾 Save", type="primary"):
                        try:
                            api.update_vacancy(
                                v["id"],
                                title=e_title.strip() or None,
                                description=e_desc.strip() or None,
                                experience=e_exp.strip() or None,
                            )
                            st.session_state.pop("edit_vac_id", None)
                            st.success("Updated.")
                            _refresh()
                        except ApiError as e:
                            st.error(str(e))
                    if cancel.form_submit_button("Cancel"):
                        st.session_state.pop("edit_vac_id", None)
                        st.rerun()


# ─── Candidates tab ─────────────────────────────────────────────────────────

def _candidates_tab() -> None:
    api = _api()
    st.subheader("Upload a CV")
    uploaded = st.file_uploader(
        "PDF or DOCX",
        type=["pdf", "docx"],
        accept_multiple_files=False,
        help="The file is parsed inline (10–30 s) and embedded with bge-m3.",
    )
    if uploaded is not None:
        if st.button("📤 Upload & parse", type="primary"):
            with st.spinner(
                "Parsing — this can take 10-30 s on a CPU-only container…"
            ):
                try:
                    detail = api.upload_candidate(uploaded, filename=uploaded.name)
                    st.success(
                        f"Parsed #{detail['id']} — {detail['name'] or '?'}"
                    )
                    _refresh()
                except ApiError as e:
                    st.error(str(e))

    st.divider()
    st.subheader("📥 Pull CVs from Gmail")
    st.caption(
        "CVs are pulled automatically every 5 minutes. "
        "Click below to trigger an immediate pull."
    )
    if st.button("📥 Pull CVs from Gmail", type="secondary"):
        try:
            resp = api.trigger_pull()
            task_id = resp.get("task_id")
            if not task_id:
                st.error("No task ID returned.")
            else:
                with st.spinner(
                    "Pulling from Gmail and parsing — this may take a minute…"
                ):
                    import time

                    for _ in range(60):  # poll for up to 2 minutes
                        time.sleep(2)
                        status_resp = api.poll_pull_status(task_id)
                        if status_resp.get("status") in ("SUCCESS", "FAILURE"):
                            break
                    else:
                        st.warning("Task is still running. Check back shortly.")
                        return

                    if status_resp.get("status") == "SUCCESS":
                        result = status_resp.get("result", {})
                        pulled = result.get("pulled", 0)
                        parsed = result.get("parsed", 0)
                        errors = result.get("errors", [])
                        if pulled == 0:
                            st.info("No new emails found.")
                        else:
                            st.success(
                                f"Pulled {pulled} file(s), parsed {parsed} "
                                f"new candidate(s)."
                            )
                        if errors:
                            for err in errors:
                                st.warning(f"⚠️ {err}")
                        if parsed > 0:
                            _refresh()
                    else:
                        st.error(
                            f"Task failed: {status_resp.get('error', 'unknown')}"
                        )
        except ApiError as e:
            st.error(str(e))

    st.divider()
    st.subheader("Existing candidates")
    try:
        candidates = api.list_candidates(include_eval=False)
    except ApiError as e:
        st.error(str(e))
        return
    if not candidates:
        st.info("No candidates yet. Upload a CV above.")
        return

    for c in candidates:
        with st.container(border=True):
            cols = st.columns([0.4, 0.4, 0.1, 0.1])
            cols[0].markdown(f"**{c.get('name') or '(no name)'}**")
            cols[1].caption(
                f"{c.get('email') or 'no email'} · "
                f"`{Path(c['source_file']).name}`"
            )
            cols[2].caption(f"id={c['id']}")
            if cols[3].button("🗑️", key=f"del_c_{c['id']}", type="secondary"):
                try:
                    api.delete_candidate(c["id"])
                    st.toast(f"Deleted candidate #{c['id']}", icon="🗑️")
                    _refresh()
                except ApiError as e:
                    st.error(str(e))


def main() -> None:
    st.title("⚙️ Manage")
    st.caption(
        "Create, edit, and delete vacancies and candidates. "
        "All operations are immediate — no draft mode."
    )
    tab_v, tab_c = st.tabs(["📋 Vacancies", "👤 Candidates"])
    with tab_v:
        _vacancies_tab()
    with tab_c:
        _candidates_tab()


main()
