"""add memory graph columns

Revision ID: 9f1d8c2a4b11
Revises: cb896e9e1284
Create Date: 2026-04-27 19:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9f1d8c2a4b11"
down_revision: Union[str, Sequence[str], None] = "cb896e9e1284"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _existing_columns(bind, table_name: str) -> set[str]:
    insp = sa.inspect(bind)
    return {c["name"] for c in insp.get_columns(table_name)}


def upgrade() -> None:
    bind = op.get_bind()
    cols = _existing_columns(bind, "user_memory")

    if "weight" not in cols:
        op.add_column("user_memory", sa.Column("weight", sa.Float(), nullable=True, server_default=sa.text("1.0")))
    if "last_used" not in cols:
        if bind.dialect.name == "sqlite":
            op.add_column("user_memory", sa.Column("last_used", sa.Text(), nullable=True, server_default=sa.text("(datetime('now'))")))
        else:
            op.add_column("user_memory", sa.Column("last_used", sa.DateTime(), nullable=True, server_default=sa.text("NOW()")))
    if "recency_score" not in cols:
        op.add_column("user_memory", sa.Column("recency_score", sa.Float(), nullable=True, server_default=sa.text("1.0")))


def downgrade() -> None:
    bind = op.get_bind()
    cols = _existing_columns(bind, "user_memory")

    if "recency_score" in cols:
        op.drop_column("user_memory", "recency_score")
    if "last_used" in cols:
        op.drop_column("user_memory", "last_used")
    if "weight" in cols:
        op.drop_column("user_memory", "weight")
