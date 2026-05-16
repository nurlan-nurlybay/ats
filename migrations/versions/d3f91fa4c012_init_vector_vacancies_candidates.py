"""init_vector_vacancies_candidates

Revision ID: d3f91fa4c012
Revises:
Create Date: 2026-05-15 22:14:41.917449

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "d3f91fa4c012"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "vacancies",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("experience", sa.String(128), nullable=True),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("source_filename", sa.String(512), nullable=False),
        sa.Column("embedding", Vector(1024), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("source_filename", name="uq_vacancies_source_filename"),
    )
    op.create_index(
        "ix_vacancies_source_filename", "vacancies", ["source_filename"]
    )

    op.create_table(
        "candidates",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(256), nullable=True),
        sa.Column("email", sa.String(256), nullable=True),
        sa.Column("source_file", sa.String(1024), nullable=False),
        sa.Column("raw_text", sa.Text, nullable=False),
        sa.Column("embedding", Vector(1024), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("source_file", name="uq_candidates_source_file"),
    )
    op.create_index("ix_candidates_email", "candidates", ["email"])
    op.create_index("ix_candidates_source_file", "candidates", ["source_file"])


def downgrade() -> None:
    op.drop_index("ix_candidates_source_file", table_name="candidates")
    op.drop_index("ix_candidates_email", table_name="candidates")
    op.drop_table("candidates")
    op.drop_index("ix_vacancies_source_filename", table_name="vacancies")
    op.drop_table("vacancies")
    # Keep the `vector` extension — other DBs in the cluster may use it.
