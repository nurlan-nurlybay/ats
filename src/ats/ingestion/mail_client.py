"""IMAP ingestion: pull unseen resume attachments from the mailbox.

Connects with credentials from `settings.imap`, fetches UNSEEN messages,
saves any `.pdf` / `.docx` attachments to `settings.paths.cvs`, and
marks the source messages as SEEN so they are not processed twice.

Hash self-healing: on init, if the number of CV files on disk does not
match the number of entries in `seen_hashes.json`, the hash set is wiped
and recalculated from the files currently on disk.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from imap_tools import AND, MailBox
from imap_tools.message import MailMessage

from ats.core.config import settings
from ats.core.logger import bind_context, clear_context, get_logger

log = get_logger(__name__)

ALLOWED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx")
_UNSAFE_CHARS = re.compile(r"[^\w.-]+")


def _safe_filename(name: str) -> str:
    """Reduce an attachment filename to its sanitized basename."""
    p = Path(name)
    stem = _UNSAFE_CHARS.sub("_", p.stem).strip("._") or "attachment"
    return f"{stem}{p.suffix.lower()}"


@dataclass(frozen=True)
class SavedAttachment:
    uid: str
    sender: str
    subject: str
    path: Path


class EmailIngestionService:
    """Fetch new resume attachments from the configured IMAP mailbox."""

    def __init__(self, save_dir: Path | None = None) -> None:
        self.save_dir = save_dir or settings.paths.cvs
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.hashes_file = self.save_dir / "seen_hashes.json"
        self._load_or_heal_hashes()

    def _cv_files(self) -> list[Path]:
        """Return all CV files on disk in the save directory."""
        return [
            p for p in self.save_dir.iterdir()
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
        ]

    def _load_or_heal_hashes(self) -> None:
        """Load hashes from disk. Self-heal if count mismatches files on disk."""
        files = self._cv_files()

        if self.hashes_file.exists():
            try:
                stored = set(json.loads(self.hashes_file.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, TypeError):
                stored = set()

            if len(stored) == len(files):
                self.seen_hashes = stored
                return
            else:
                log.warning(
                    "hash_mismatch_healing",
                    stored_hashes=len(stored),
                    files_on_disk=len(files),
                )

        # Rebuild from scratch
        self.seen_hashes: set[str] = set()
        for p in files:
            self.seen_hashes.add(hashlib.sha256(p.read_bytes()).hexdigest())
        self._save_hashes()
        log.info("hashes_rebuilt", count=len(self.seen_hashes))

    def _save_hashes(self) -> None:
        self.hashes_file.write_text(json.dumps(list(self.seen_hashes)), encoding="utf-8")

    def fetch_new_resumes(self) -> list[SavedAttachment]:
        """Download attachments from every UNSEEN message; mark them SEEN.

        Returns the list of saved attachments. Errors processing a single
        message are logged and do not abort the rest of the batch.
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
                    found = self._save_attachments(msg)
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

        log.info("ingestion_complete", emails_with_attachments=len({s.uid for s in saved}), files_saved=len(saved))
        return saved

    def _save_attachments(self, msg: MailMessage) -> list[SavedAttachment]:
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

            filename = f"{msg.uid}_{_safe_filename(att.filename)}"
            path = self.save_dir / filename

            payload_hash = hashlib.sha256(att.payload).hexdigest()
            if payload_hash in self.seen_hashes:
                log.info(
                    "attachment_skipped",
                    filename=att.filename,
                    reason="duplicate_content",
                )
                continue

            path.write_bytes(att.payload)
            self.seen_hashes.add(payload_hash)
            self._save_hashes()

            log.info("attachment_saved", filename=filename, size_bytes=len(att.payload))
            saved.append(
                SavedAttachment(
                    uid=msg.uid or "unknown_uid",
                    sender=msg.from_,
                    subject=msg.subject,
                    path=path,
                )
            )
        return saved


if __name__ == "__main__":
    from ats.core.logger import configure_logging

    configure_logging(level="INFO", json_output=False)
    service = EmailIngestionService()
    results = service.fetch_new_resumes()
    log.info("script_done", count=len(results))
    for r in results:
        log.info("saved", path=str(r.path), sender=r.sender, subject=r.subject)
