"""job tracking: celery task id + progress

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-22
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("celery_task_id", sa.String(length=64)))
    op.add_column("jobs", sa.Column("progress", sa.String(length=255)))


def downgrade() -> None:
    op.drop_column("jobs", "progress")
    op.drop_column("jobs", "celery_task_id")
