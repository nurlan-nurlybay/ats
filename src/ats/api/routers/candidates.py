"""Candidate read + CRUD.

GET endpoints feed the Manage page's list and the Match page's drill-down.
POST takes a multipart PDF/DOCX upload, runs the Stage 2 parser inline,
and inserts a fully-populated row. DELETE removes the row and (by default)
the file on disk.

Resume.csv eval rows (`source_file LIKE 'resume_csv_%'`) are hidden from
the default list — the UI is for real CVs only. Pass `?include_eval=true`
on the list endpoint to override (mostly for debugging).
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ats.api.dependencies import get_session
from ats.api.schemas import CandidateDetail, CandidateSummary
from ats.core.config import settings
from ats.core.logger import get_logger
from ats.db.models import Candidate
from ats.ingestion.parser import parse_resume
from ats.matching.base import RESUME_CSV_PREFIX

log = get_logger(__name__)
router = APIRouter(prefix="/candidates", tags=["candidates"])

_MAX_RAW_TEXT_CHARS = 3000
_MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
_ALLOWED_EXTS = {".pdf", ".docx"}
_UNSAFE_CHARS = re.compile(r"[^\w.\-]+")


def _sanitize_filename(name: str) -> str:
    """Mirror the Stage 1 mail_client sanitizer — preserves Cyrillic word chars."""
    name = name.strip().replace(" ", "_")
    return _UNSAFE_CHARS.sub("_", name) or "uploaded.bin"


def _to_detail(c: Candidate) -> CandidateDetail:
    excerpt = None
    if c.raw_text:
        excerpt = c.raw_text[:_MAX_RAW_TEXT_CHARS]
        if len(c.raw_text) > _MAX_RAW_TEXT_CHARS:
            excerpt += "… [truncated]"
    return CandidateDetail(
        id=c.id,
        name=c.name,
        email=c.email,
        source_file=c.source_file,
        parsed_json=c.parsed_json,
        raw_text_excerpt=excerpt,
        created_at=c.created_at,
    )


# ─── READ ───────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=list[CandidateSummary],
    summary="List candidates (Resume.csv rows hidden by default)",
)
async def list_candidates(
    include_eval: bool = Query(
        False,
        description="Include `resume_csv_*` mock candidates (default: hidden).",
    ),
    limit: int = Query(200, ge=1, le=2000),
    session: AsyncSession = Depends(get_session),
) -> list[CandidateSummary]:
    stmt = select(
        Candidate.id, Candidate.name, Candidate.email, Candidate.source_file
    )
    if not include_eval:
        stmt = stmt.where(~Candidate.source_file.like(f"{RESUME_CSV_PREFIX}%"))
    stmt = stmt.order_by(Candidate.id.desc()).limit(limit)
    rows = (await session.execute(stmt)).all()
    return [
        CandidateSummary(
            id=r.id, name=r.name, email=r.email, source_file=r.source_file,
        )
        for r in rows
    ]


@router.get(
    "/{candidate_id}",
    response_model=CandidateDetail,
    responses={404: {"description": "Candidate not found"}},
)
async def get_candidate(
    candidate_id: int,
    session: AsyncSession = Depends(get_session),
) -> CandidateDetail:
    c = await session.get(Candidate, candidate_id)
    if c is None:
        raise HTTPException(404, f"candidate id={candidate_id} not found")
    return _to_detail(c)


# ─── WRITE ──────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=CandidateDetail,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF/DOCX CV; parses + embeds inline",
    responses={
        400: {"description": "Bad file (unsupported extension or empty)"},
        409: {"description": "File already exists on disk"},
        413: {"description": "File too large"},
    },
)
async def upload_candidate(
    file: UploadFile = File(..., description="PDF or DOCX resume"),
    session: AsyncSession = Depends(get_session),
) -> CandidateDetail:
    if not file.filename:
        raise HTTPException(400, "no filename provided")
    ext = Path(file.filename).suffix.lower()
    if ext not in _ALLOWED_EXTS:
        raise HTTPException(400, f"unsupported file type: {ext!r}")

    payload = await file.read(_MAX_UPLOAD_BYTES + 1)
    if len(payload) > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file exceeds {_MAX_UPLOAD_BYTES} bytes")
    if not payload:
        raise HTTPException(400, "uploaded file is empty")

    target_dir = settings.paths.cvs
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / _sanitize_filename(file.filename)
    if target.exists():
        raise HTTPException(409, f"file already exists: {target.name}")

    target.write_bytes(payload)
    log.info("candidate_upload_saved", path=str(target), bytes=len(payload))

    try:
        parsed = parse_resume(target)
    except Exception as exc:
        # Roll back the disk write if parsing fails — we don't want orphan files.
        target.unlink(missing_ok=True)
        log.warning("candidate_parse_failed", err=str(exc))
        raise HTTPException(400, f"parse failed: {exc}") from exc

    cand = Candidate(
        source_file=str(target),
        name=parsed.name,
        email=parsed.email,
        raw_text=parsed.raw_text,
        embedding=parsed.embedding,
        parsed_json=parsed.model_dump(exclude={"raw_text", "embedding"}),
    )
    session.add(cand)
    await session.commit()
    await session.refresh(cand)
    log.info("candidate_uploaded", id=cand.id, name=cand.name, source_file=cand.source_file)
    return _to_detail(cand)


@router.delete(
    "/{candidate_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete candidate row (and its file on disk by default)",
    responses={404: {"description": "Candidate not found"}},
)
async def delete_candidate(
    candidate_id: int,
    keep_file: bool = Query(False, description="If true, leave the file on disk."),
    session: AsyncSession = Depends(get_session),
) -> None:
    c = await session.get(Candidate, candidate_id)
    if c is None:
        raise HTTPException(404, f"candidate id={candidate_id} not found")
    source_path = c.source_file
    await session.delete(c)
    await session.commit()
    if not keep_file:
        try:
            p = Path(source_path)
            # Only delete if it's actually under settings.paths.cvs — never
            # follow arbitrary paths from old eval imports.
            if p.exists() and p.is_file() and settings.paths.cvs in p.parents:
                p.unlink()
                log.info("candidate_file_deleted", path=str(p))
        except Exception as e:
            log.warning("candidate_file_delete_failed", err=str(e))
    log.info("candidate_deleted", id=candidate_id)
