"""add_candidate_content_hash

Revision ID: c1a2b3d4e5f6
Revises: 686ec35ba1cc
Create Date: 2026-05-18 13:00:00.000000

Adds a SHA-256 hex column for content-level dedup, consolidating the
mail_client `seen_hashes.json` and the API upload's filename-only check
into a single unique DB column. Backfills existing rows by reading each
`source_file` from disk.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c1a2b3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "686ec35ba1cc"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "candidates",
        sa.Column("content_hash", sa.String(64), nullable=True),
    )
    op.create_index(
        "ix_candidates_content_hash",
        "candidates",
        ["content_hash"],
        unique=True,
    )

    # Backfill — small N (≤ a few thousand). Read each file, hash its bytes.
    # Rows whose source_file is missing on disk are left NULL; future inserts
    # will populate them, and the unique index tolerates NULLs.
    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT id, source_file FROM candidates")
    ).fetchall()
    for row_id, src in rows:
        p = Path(src)
        if not p.exists() or not p.is_file():
            continue
        try:
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            continue
        conn.execute(
            sa.text(
                "UPDATE candidates SET content_hash = :h "
                "WHERE id = :i AND content_hash IS NULL"
            ),
            {"h": digest, "i": row_id},
        )


def downgrade() -> None:
    op.drop_index("ix_candidates_content_hash", table_name="candidates")
    op.drop_column("candidates", "content_hash")
