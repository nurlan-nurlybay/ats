"""add_candidate_parsed_json

Revision ID: 686ec35ba1cc
Revises: d3f91fa4c012
Create Date: 2026-05-16 15:45:12.390106

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "686ec35ba1cc"
down_revision: Union[str, Sequence[str], None] = "d3f91fa4c012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "candidates",
        sa.Column("parsed_json", JSONB, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("candidates", "parsed_json")
