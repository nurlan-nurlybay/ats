"""Shared helpers for CV file storage.

Both the API upload endpoint and the Gmail-ingestion Celery task converge on
the same on-disk layout:

    {settings.paths.cvs}/{candidate_id}_{sanitized_original}.{ext}

The flow is **insert-then-rename**:

    1. compute sha256(payload)
    2. save to a temp name to avoid clobbering anything mid-flight
    3. check `content_hash` against the DB; if collide, raise/return early
    4. parse + insert candidate row (with source_file = temp path,
       content_hash = the digest)
    5. rename temp → `{id}_{sanitized}.{ext}` and UPDATE source_file

These helpers stay synchronous and pure (no DB calls) so the two callers
can use them in whatever async session they already hold.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path

# Mirrors the Stage 1 mail_client sanitizer — preserves Cyrillic word chars
# and `.` `_` `-`, replaces every other byte with `_`. Stable across the
# api + worker.
_UNSAFE_CHARS = re.compile(r"[^\w.\-]+")


def sanitize_filename(name: str) -> str:
    """Drop everything but `\\w.-`, collapse whitespace to `_`, lowercase ext.

    `name` may be a bare filename or a path; we only keep the basename.
    Returns a stem+suffix, never raises. Falls back to `'uploaded.bin'`
    if the input collapses to empty.
    """
    base = Path(name).name.strip().replace(" ", "_")
    cleaned = _UNSAFE_CHARS.sub("_", base) or "uploaded.bin"
    # Lowercase the suffix; the stem is left as-is (allows Cyrillic).
    p = Path(cleaned)
    return f"{p.stem}{p.suffix.lower()}"


def compute_hash(payload: bytes) -> str:
    """SHA-256 hex digest. 64 chars, lowercase."""
    return hashlib.sha256(payload).hexdigest()


def temp_path_for(cvs_dir: Path, sanitized: str) -> Path:
    """A unique temp filename in `cvs_dir` for the in-flight write.

    Prefix `_tmp_` so the worker's CV-file scanner skips them, and an
    8-char hex tag for uniqueness even if two uploads collide in time.
    """
    tag = uuid.uuid4().hex[:8]
    return cvs_dir / f"_tmp_{tag}_{sanitized}"


def final_path_for(cvs_dir: Path, candidate_id: int, sanitized: str) -> Path:
    """`{cvs_dir}/{candidate_id}_{sanitized}` — the canonical post-insert name."""
    return cvs_dir / f"{candidate_id}_{sanitized}"
