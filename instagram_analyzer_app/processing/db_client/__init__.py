"""Database client.

PostgreSQL via a threaded connection pool when DATABASE_URL is set;
otherwise an in-memory store guarded by a lock for local/dev use.

Schema is owned by Alembic. This module no longer creates tables at
runtime — run `alembic upgrade head` before serving traffic.
"""


from . import export, jobs, memory, pool, users  # noqa: F401  (eager: same import footprint)
from .export import _METRIC_KEYS, export_to_excel
from .jobs import (
    active_job_ids, create_job, fail_stale_jobs, get_all_jobs, get_job_by_id, save_job_metrics,
    set_job_task_id, update_job_progress, update_job_status,
)
from .memory import _memory_jobs, _memory_lock, _memory_metrics, _memory_users
from .pool import _conn, _get_pool, _pool_lock, is_database_available   # not _pool: rebound via `global`
from .users import (
    _hash_password, _is_legacy_hash, _update_user_password, _verify_password, create_user,
    get_user_count, verify_user,
)

__all__ = [
    "create_job", "set_job_task_id", "update_job_progress", "fail_stale_jobs", "active_job_ids",
    "update_job_status", "save_job_metrics", "get_job_by_id", "get_all_jobs",
    "export_to_excel", "_METRIC_KEYS",
    "create_user", "verify_user", "get_user_count", "_hash_password", "_verify_password",
    "_is_legacy_hash", "_update_user_password",
    "is_database_available", "_get_pool", "_conn", "_pool_lock",
    "_memory_lock", "_memory_jobs", "_memory_metrics", "_memory_users",
]
