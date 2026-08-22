"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-05-05
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(length=100), primary_key=True),
        sa.Column("campaign_date", sa.Date(), nullable=False),
        sa.Column("campaign_name", sa.String(length=255), nullable=False),
        sa.Column("product_name", sa.String(length=255), nullable=False),
        sa.Column("company", sa.String(length=255), nullable=False),
        sa.Column("filename", sa.String(length=255)),
        sa.Column("file_type", sa.String(length=50)),
        sa.Column("status", sa.String(length=50), server_default="pending"),
        sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("NOW()")),
        sa.Column("completed_at", sa.TIMESTAMP()),
        sa.Column("error_message", sa.Text()),
    )
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_index("ix_jobs_created_at", "jobs", ["created_at"])

    op.create_table(
        "job_metrics",
        sa.Column("job_id", sa.String(length=100), primary_key=True),
        sa.Column("total_frames", sa.Integer()),
        sa.Column("good_frames", sa.Integer()),
        sa.Column("bad_frames", sa.Integer()),
        sa.Column("processing_time_seconds", sa.Integer()),
        sa.Column("metrics_json", sa.dialects.postgresql.JSONB()),
        sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("NOW()")),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
    )

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("username", sa.String(length=100), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("NOW()")),
    )


def downgrade() -> None:
    op.drop_table("users")
    op.drop_table("job_metrics")
    op.drop_index("ix_jobs_created_at", table_name="jobs")
    op.drop_index("ix_jobs_status", table_name="jobs")
    op.drop_table("jobs")
