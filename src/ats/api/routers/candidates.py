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
from ats.api.schemas import CandidateDetail, CandidateSummary, CandidateUpdate
from ats.core.config import settings
from ats.core.logger import get_logger
from ats.db.models import Candidate
from ats.ingestion.cv_store import (
    compute_hash,
    final_path_for,
    sanitize_filename,
    temp_path_for,
)
from ats.ingestion.parser import parse_resume
from ats.matching.base import RESUME_CSV_PREFIX

log = get_logger(__name__)
router = APIRouter(prefix="/candidates", tags=["candidates"])

_MAX_RAW_TEXT_CHARS = 3000
_MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
_ALLOWED_EXTS = {".pdf", ".docx"}


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
        400: {"description": "Bad file (unsupported extension, empty, or parse failure)"},
        409: {"description": "Duplicate content — already uploaded as candidate id=X"},
        413: {"description": "File too large"},
    },
)
async def upload_candidate(
    file: UploadFile = File(..., description="PDF or DOCX resume"),
    session: AsyncSession = Depends(get_session),
) -> CandidateDetail:
    """Upload + parse + insert flow.

    Insert-then-rename pattern:
      1. read bytes, hash them
      2. SHA-256 lookup in DB; 409 if seen before (regardless of filename)
      3. save to a temp name in `data/cvs/`
      4. parse_resume(temp_path)
      5. insert candidate row with source_file=temp_path + content_hash
      6. rename temp → `{id}_{sanitized}.{ext}` and UPDATE source_file
    """
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

    # Content-level dedup. Identical bytes → 409, regardless of filename.
    digest = compute_hash(payload)
    dupe_id = await session.scalar(
        select(Candidate.id).where(Candidate.content_hash == digest)
    )
    if dupe_id is not None:
        raise HTTPException(
            409,
            f"duplicate content (already uploaded as candidate id={dupe_id})",
        )

    cvs_dir = settings.paths.cvs
    cvs_dir.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_filename(file.filename)
    temp_path = temp_path_for(cvs_dir, sanitized)

    temp_path.write_bytes(payload)
    log.info("candidate_upload_temp_saved", path=str(temp_path), bytes=len(payload))

    try:
        parsed = parse_resume(temp_path)
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        log.warning("candidate_parse_failed", err=str(exc))
        raise HTTPException(400, f"parse failed: {exc}") from exc

    cand = Candidate(
        source_file=str(temp_path),
        content_hash=digest,
        name=parsed.name,
        email=parsed.email,
        raw_text=parsed.raw_text,
        embedding=parsed.embedding,
        parsed_json=parsed.model_dump(exclude={"raw_text", "embedding"}),
    )
    session.add(cand)
    await session.commit()
    await session.refresh(cand)

    # Now we know the id — promote temp file to the canonical name.
    final_path = final_path_for(cvs_dir, cand.id, sanitized)
    try:
        temp_path.rename(final_path)
    except OSError as exc:
        # The DB row exists with a temp path; not fatal, but flag it loudly so
        # the operator can run a one-off rename. We don't roll back the insert.
        log.error(
            "candidate_rename_failed",
            id=cand.id, temp=str(temp_path), target=str(final_path), err=str(exc),
        )
    else:
        cand.source_file = str(final_path)
        await session.commit()
        await session.refresh(cand)

    log.info(
        "candidate_uploaded",
        id=cand.id, name=cand.name, source_file=cand.source_file,
        hash=digest[:12],
    )
    return _to_detail(cand)


@router.put(
    "/{candidate_id}",
    response_model=CandidateDetail,
    summary="Edit candidate name/email (HR override of parser output)",
    responses={404: {"description": "Candidate not found"}},
)
async def update_candidate(
    candidate_id: int,
    body: CandidateUpdate,
    session: AsyncSession = Depends(get_session),
) -> CandidateDetail:
    """Partial update of name and/or email.

    Use `null`/omit to leave a field unchanged. Pass an empty string to clear
    a wrong NER-extracted value. Does NOT re-parse or re-embed — content is
    parser-managed.
    """
    c = await session.get(Candidate, candidate_id)
    if c is None:
        raise HTTPException(404, f"candidate id={candidate_id} not found")

    changed: list[str] = []
    if body.name is not None:
        new_name = body.name.strip() or None
        if new_name != c.name:
            c.name = new_name
            changed.append("name")
    if body.email is not None:
        new_email = body.email.strip() or None
        if new_email != c.email:
            c.email = new_email
            changed.append("email")

    if changed:
        await session.commit()
        await session.refresh(c)
        log.info("candidate_updated", id=c.id, fields=changed)
    return _to_detail(c)


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
