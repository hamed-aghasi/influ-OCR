"""job liveness: last_progress_at, so the reaper keys off activity

Before this, fail_stale_jobs compared created_at, which fails a job for
waiting in the queue: with a small worker pool and a large backlog, healthy
queued jobs were marked failed and then completed anyway. Nullable, and every
read COALESCEs to created_at, so existing rows behave as before.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-20
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("last_progress_at", sa.TIMESTAMP()))
    # Existing unfinished rows get a starting point; without it every one of
    # them looks stale the first time the new reaper runs.
    op.execute("UPDATE jobs SET last_progress_at = created_at WHERE last_progress_at IS NULL")
    op.create_index("ix_jobs_status_last_progress", "jobs", ["status", "last_progress_at"])


def downgrade() -> None:
    op.drop_index("ix_jobs_status_last_progress", table_name="jobs")
    op.drop_column("jobs", "last_progress_at")
