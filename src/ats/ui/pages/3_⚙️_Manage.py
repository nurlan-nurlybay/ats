"""Streamlit page: CRUD admin for vacancies and candidates.

Two tabs:
  📋 Vacancies — create / edit / delete (text-based form)
  👤 Candidates — upload PDF/DOCX, list, delete

All operations hit the FastAPI surface; nothing here talks to Postgres
directly. The page is intentionally lean — no validation duplication.
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import streamlit as st

from ats.ui.api_client import ApiClient, ApiError
from ats.ui.components import render_parsed_json

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


@st.cache_data(show_spinner=False)
def _candidate_detail(cid: int) -> dict[str, Any]:
    """Fetch + cache full candidate detail. Invalidated by `_refresh()`."""
    return _api().get_candidate(cid)


def _bulk_delete_bar(
    kind: str,                       # "vacancy" or "candidate"
    all_ids: list[int],
    delete_fn,                       # api.delete_vacancy / api.delete_candidate
) -> None:
    """Render the bulk-delete control + confirmation flow above a list.

    Checkboxes per row live in session_state under `bulk_{kind}_{id}` and are
    set by `_bulk_checkbox(...)` next to each row. This helper reads them,
    shows a count, asks for confirm, then loops the delete API.
    """
    selected = [i for i in all_ids if st.session_state.get(f"bulk_{kind}_{i}", False)]
    confirm_key = f"confirm_bulk_{kind}"
    n = len(selected)

    if not st.session_state.get(confirm_key, False):
        if st.button(
            f"🗑️ Удалить выбранное ({n})",
            disabled=n == 0,
            type="primary",
            key=f"bulk_btn_{kind}",
        ):
            st.session_state[confirm_key] = True
            st.rerun()
        return

    st.warning(f"Удалить {n} элемент(ов)? Действие необратимо.")
    yes, no = st.columns([1, 1])
    if yes.button("✅ Да, удалить", key=f"bulk_yes_{kind}", type="primary"):
        ok, errs = 0, []
        for i in selected:
            try:
                delete_fn(i)
                ok += 1
            except ApiError as e:
                errs.append((i, str(e)))
        for i in all_ids:
            st.session_state.pop(f"bulk_{kind}_{i}", None)
        st.session_state.pop(confirm_key, None)
        st.toast(f"Удалено: {ok}; ошибок: {len(errs)}", icon="🗑️")
        for i, msg in errs:
            st.error(f"id={i}: {msg}")
        _refresh()
    if no.button("❌ Отмена", key=f"bulk_no_{kind}"):
        st.session_state.pop(confirm_key, None)
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

    _bulk_delete_bar("vacancy", [v["id"] for v in vacancies], api.delete_vacancy)

    edit_id = st.session_state.get("edit_vac_id")
    for v in vacancies:
        sel_col, row_col = st.columns([0.04, 0.96])
        with sel_col:
            st.checkbox(
                "select", key=f"bulk_vacancy_{v['id']}",
                label_visibility="collapsed",
            )
        with row_col, st.container(border=True):
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

    _bulk_delete_bar(
        "candidate", [c["id"] for c in candidates], api.delete_candidate,
    )

    # Group by email so a person who submitted multiple CVs appears once.
    NO_EMAIL = "(нет email)"
    by_email: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_email[c.get("email") or NO_EMAIL].append(c)
    # emails with addresses first (alphabetical), no-email bucket last
    ordered_keys = sorted(
        by_email.keys(), key=lambda k: (k == NO_EMAIL, k.lower()),
    )

    edit_id = st.session_state.get("edit_cand_id")
    for email_key in ordered_keys:
        rows = by_email[email_key]
        with st.container(border=True):
            st.markdown(
                f"**{email_key}** · {len(rows)} резюме"
            )
            for c in rows:
                _render_candidate_row(api, c, edit_id)


def _render_candidate_row(
    api: ApiClient, c: dict[str, Any], edit_id: int | None,
) -> None:
    sel_col, row_col = st.columns([0.04, 0.96])
    with sel_col:
        st.checkbox(
            "select", key=f"bulk_candidate_{c['id']}",
            label_visibility="collapsed",
        )
    with row_col, st.container(border=True):
        cols = st.columns([0.45, 0.35, 0.10, 0.05, 0.05])
        cols[0].markdown(f"**{c.get('name') or 'Not Found'}**")
        cols[1].caption(f"`{Path(c['source_file']).name}`")
        cols[2].caption(f"id={c['id']}")
        if cols[3].button("✏️", key=f"edit_c_{c['id']}", help="Edit name/email"):
            st.session_state["edit_cand_id"] = c["id"]
            st.rerun()
        if cols[4].button("🗑️", key=f"del_c_{c['id']}", type="secondary"):
            try:
                api.delete_candidate(c["id"])
                st.toast(f"Deleted candidate #{c['id']}", icon="🗑️")
                _refresh()
            except ApiError as e:
                st.error(str(e))

        # Inline edit form — only the row whose ✏️ was clicked
        if edit_id == c["id"]:
            with st.form(f"edit_cand_form_{c['id']}", clear_on_submit=False):
                e_name = st.text_input(
                    "Name", value=c.get("name") or "",
                    help="Leave empty to clear (e.g. parser picked garbage).",
                )
                e_email = st.text_input(
                    "Email", value=c.get("email") or "",
                )
                save_col, cancel_col = st.columns(2)
                if save_col.form_submit_button("💾 Save", type="primary"):
                    try:
                        api.update_candidate(
                            c["id"], name=e_name.strip(), email=e_email.strip(),
                        )
                        st.session_state.pop("edit_cand_id", None)
                        st.success("Saved.")
                        _refresh()
                    except ApiError as e:
                        st.error(str(e))
                if cancel_col.form_submit_button("Cancel"):
                    st.session_state.pop("edit_cand_id", None)
                    st.rerun()

        with st.expander("▾ Показать детали"):
            try:
                detail = _candidate_detail(c["id"])
            except ApiError as e:
                st.error(str(e))
            else:
                render_parsed_json(
                    detail.get("parsed_json"),
                    detail.get("raw_text_excerpt"),
                )


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
