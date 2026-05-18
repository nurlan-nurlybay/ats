"""IMAP ingestion: pull unseen resume attachments from the mailbox.

Connects with credentials from `settings.imap`, fetches UNSEEN messages,
saves `.pdf` / `.docx` attachments to `settings.paths.cvs` (under a temp
name — the Celery task that called us renames them to `{id}_{name}.{ext}`
after the DB row is inserted), and marks the source messages as SEEN.

Content-level dedup is **DB-backed** since Stage 7. Before saving an
attachment to disk we hash its bytes and look up `candidates.content_hash`;
if a row already has the same hash, the attachment is skipped silently.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from imap_tools import AND, MailBox
from imap_tools.message import MailMessage
from sqlalchemy import select

from ats.core.config import settings
from ats.core.logger import bind_context, clear_context, get_logger
from ats.db.base import SessionLocal
from ats.db.models import Candidate
from ats.ingestion.cv_store import compute_hash, sanitize_filename, temp_path_for

log = get_logger(__name__)

ALLOWED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx")


@dataclass(frozen=True)
class SavedAttachment:
    """One PDF/DOCX successfully written to disk under a temp filename.

    The Celery task receives these, parses them, inserts a Candidate row,
    and renames the file to `{candidate_id}_{sanitized}.{ext}`.
    """

    uid: str
    sender: str
    subject: str
    path: Path
    content_hash: str
    sanitized_basename: str


class EmailIngestionService:
    """Fetch new resume attachments from the configured IMAP mailbox.

    Stateless on disk — no more `seen_hashes.json`. Each `fetch_new_resumes()`
    call opens a DB session, checks `candidates.content_hash` per attachment,
    and only writes the new ones to disk under a `_tmp_*` filename. The
    caller (Celery task) is responsible for parsing + insert + rename.
    """

    def __init__(self, save_dir: Path | None = None) -> None:
        self.save_dir = save_dir or settings.paths.cvs
        self.save_dir.mkdir(parents=True, exist_ok=True)

    async def _hash_already_in_db(self, digest: str) -> int | None:
        """Return candidate id matching this hash, or None."""
        async with SessionLocal() as session:
            return await session.scalar(
                select(Candidate.id).where(Candidate.content_hash == digest)
            )

    async def fetch_new_resumes(self) -> list[SavedAttachment]:
        """Download attachments from every UNSEEN message; mark them SEEN.

        Returns the list of saved attachments (under temp filenames). Errors
        processing a single message are logged and do not abort the rest of
        the batch.
        """
        cfg = settings.imap
        saved: list[SavedAttachment] = []

        log.info("imap_connect", server=cfg.server, port=cfg.port, user=cfg.user)
        with MailBox(cfg.server, port=cfg.port).login(
            cfg.user, cfg.password.get_secret_value()
        ) as box:
            for msg in box.fetch(AND(seen=False), mark_seen=True):
                bind_context(uid=msg.uid, subject=msg.subject)
                try:
                    found = await self._save_attachments(msg)
                    saved.extend(found)
                    log.info(
                        "email_processed",
                        sender=msg.from_,
                        attachments_found=len(found),
                    )
                except Exception:
                    log.exception("email_processing_failed")
                finally:
                    clear_context()

        log.info(
            "ingestion_complete",
            emails_with_attachments=len({s.uid for s in saved}),
            files_saved=len(saved),
        )
        return saved

    async def _save_attachments(self, msg: MailMessage) -> list[SavedAttachment]:
        saved: list[SavedAttachment] = []
        for att in msg.attachments:
            if not att.filename:
                continue
            ext = Path(att.filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                log.debug(
                    "attachment_skipped",
                    filename=att.filename,
                    reason="unsupported_extension",
                )
                continue

            digest = compute_hash(att.payload)
            dupe_id = await self._hash_already_in_db(digest)
            if dupe_id is not None:
                log.info(
                    "attachment_skipped",
                    filename=att.filename,
                    reason="duplicate_content",
                    existing_candidate_id=dupe_id,
                )
                continue

            sanitized = sanitize_filename(att.filename)
            temp_path = temp_path_for(self.save_dir, sanitized)
            temp_path.write_bytes(att.payload)

            log.info(
                "attachment_saved_temp",
                temp=temp_path.name,
                size_bytes=len(att.payload),
                hash=digest[:12],
            )
            saved.append(
                SavedAttachment(
                    uid=msg.uid or "unknown_uid",
                    sender=msg.from_,
                    subject=msg.subject,
                    path=temp_path,
                    content_hash=digest,
                    sanitized_basename=sanitized,
                )
            )
        return saved


if __name__ == "__main__":
    import asyncio

    from ats.core.logger import configure_logging

    configure_logging(level="INFO", json_output=False)
    service = EmailIngestionService()
    results = asyncio.run(service.fetch_new_resumes())
    log.info("script_done", count=len(results))
    for r in results:
        log.info(
            "saved",
            path=str(r.path),
            sender=r.sender,
            subject=r.subject,
            sanitized=r.sanitized_basename,
        )
