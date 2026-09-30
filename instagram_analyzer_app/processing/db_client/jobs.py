"""Job rows and job_metrics: create, track progress, reap, fetch."""

import json
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Set

from ..logger import get_logger
from .memory import _memory_jobs, _memory_lock, _memory_metrics
from .pool import _conn, is_database_available

logger = get_logger("processing.db_client")   # same name as the flat module: same Logger

def create_job(
    job_id: str,
    campaign_date: date,
    campaign_name: str,
    product_name: str,
    company: str,
    filename: str,
    file_type: str,
) -> bool:
    if not is_database_available():
        with _memory_lock:
            _memory_jobs[job_id] = {
                "id": job_id,
                "campaign_date": campaign_date,
                "campaign_name": campaign_name,
                "product_name": product_name,
                "company": company,
                "filename": filename,
                "file_type": file_type,
                "status": "queued",
                "celery_task_id": None,
                "progress": None,
                "created_at": datetime.now(),
                "last_progress_at": None,
                "completed_at": None,
                "error_message": None,
            }
        return True

    try:
        with _conn() as c, c.cursor() as cur:
            cur.execute(
                """
                INSERT INTO jobs (id, campaign_date, campaign_name, product_name,
                                  company, filename, file_type, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'queued')
                """,
                (job_id, campaign_date, campaign_name, product_name, company, filename, file_type),
            )
            return True
    except Exception as exc:  # noqa: BLE001
        logger.error("create_job failed: %s", exc)
        return False


def set_job_task_id(job_id: str, task_id: str) -> bool:
    if not is_database_available():
        with _memory_lock:
            job = _memory_jobs.get(job_id)
            if job is None:
                return False
            job["celery_task_id"] = task_id
            return True
    try:
        with _conn() as c, c.cursor() as cur:
            cur.execute("UPDATE jobs SET celery_task_id=%s WHERE id=%s", (task_id, job_id))
            return True
    except Exception as exc:  # noqa: BLE001
        logger.error("set_job_task_id failed: %s", exc)
        return False


def update_job_progress(job_id: str, progress: str) -> bool:
    """Record a step, and stamp the job as alive.

    `last_progress_at` is what the reaper measures; see fail_stale_jobs.
    """
    if not is_database_available():
        with _memory_lock:
            job = _memory_jobs.get(job_id)
            if job is None:
                return False
            job["progress"] = progress
            job["last_progress_at"] = datetime.now()
            return True
    try:
        with _conn() as c, c.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET progress=%s, last_progress_at=NOW() WHERE id=%s",
                (progress, job_id),
            )
            return True
    except Exception as exc:  # noqa: BLE001
        logger.error("update_job_progress failed: %s", exc)
        return False


def fail_stale_jobs(max_age_seconds: int, queued_max_age_seconds: Optional[int] = None) -> int:
    """Fail jobs that have genuinely stopped, measured by activity not by age.

    Two different things are being caught, so they get two different clocks:

    - **processing**: no progress for `max_age_seconds`. Measured from
      `last_progress_at` (COALESCEd to `created_at` for rows written before
      migration 0003), which `update_job_progress` and `update_job_status`
      stamp. A long job that keeps reporting steps is never reaped.
    - **queued**: still waiting after `queued_max_age_seconds`, which should be
      far longer. Waiting is not a symptom — with a small worker pool and a
      backlog of thousands it is the normal state — so this only catches a
      broker message that was genuinely lost. Keying this on `created_at` is
      correct: a queued job has no activity to measure.

    The old rule failed both after the same 2 hours, keyed on `created_at`, so
    a healthy queued job was marked failed and then completed anyway.

    Returns rows affected.
    """
    if queued_max_age_seconds is None:
        queued_max_age_seconds = max_age_seconds
    stalled = f"Job stalled (no progress for {max_age_seconds // 60} minutes)"
    never_started = (
        f"Job never started (queued for {queued_max_age_seconds // 60} minutes; "
        "its worker message was most likely lost)"
    )

    if not is_database_available():
        from datetime import timedelta

        now = datetime.now()
        processing_cutoff = now - timedelta(seconds=max_age_seconds)
        queued_cutoff = now - timedelta(seconds=queued_max_age_seconds)
        count = 0
        with _memory_lock:
            for job in _memory_jobs.values():
                if job["status"] == "processing":
                    last = job.get("last_progress_at") or job["created_at"]
                    stale, message = last < processing_cutoff, stalled
                elif job["status"] == "queued":
                    stale, message = job["created_at"] < queued_cutoff, never_started
                else:
                    continue
                if stale:
                    job["status"] = "failed"
                    job["completed_at"] = now
                    job["error_message"] = message
                    count += 1
        return count

    try:
        with _conn() as c, c.cursor() as cur:
            cur.execute(
                """
                UPDATE jobs SET status='failed', completed_at=NOW(), error_message=%s
                WHERE status = 'processing'
                  AND COALESCE(last_progress_at, created_at)
                      < NOW() - (%s * INTERVAL '1 second')
                """,
                (stalled, max_age_seconds),
            )
            count = cur.rowcount
            cur.execute(
                """
                UPDATE jobs SET status='failed', completed_at=NOW(), error_message=%s
                WHERE status = 'queued'
                  AND created_at < NOW() - (%s * INTERVAL '1 second')
                """,
                (never_started, queued_max_age_seconds),
            )
            return count + cur.rowcount
    except Exception as exc:  # noqa: BLE001
        logger.error("fail_stale_jobs failed: %s", exc)
        return 0


def active_job_ids() -> Set[str]:
    """Ids of jobs that are still queued or processing.

    The sweeper uses this as its do-not-touch list; see tasks.sweep_orphans.
    An empty set on a database error would let the sweeper delete a live job's
    frames, so the error is re-raised rather than swallowed like the writes.
    """
    if not is_database_available():
        with _memory_lock:
            return {j["id"] for j in _memory_jobs.values()
                    if j["status"] in {"queued", "processing"}}

    with _conn() as c, c.cursor() as cur:
        cur.execute("SELECT id FROM jobs WHERE status IN ('queued', 'processing')")
        return {row[0] for row in cur.fetchall()}


def update_job_status(job_id: str, status: str, error_message: Optional[str] = None) -> bool:
    if not is_database_available():
        with _memory_lock:
            job = _memory_jobs.get(job_id)
            if job is None:
                return False
            job["status"] = status
            job["last_progress_at"] = datetime.now()
            if status in {"completed", "failed"}:
                job["completed_at"] = datetime.now()
            if error_message:
                job["error_message"] = error_message
            return True

    try:
        with _conn() as c, c.cursor() as cur:
            if status in {"completed", "failed"}:
                cur.execute(
                    "UPDATE jobs SET status=%s, completed_at=NOW(), error_message=%s, "
                    "last_progress_at=NOW() WHERE id=%s",
                    (status, error_message, job_id),
                )
            else:
                # A status change is activity: it keeps the reaper off a job
                # that has just started and not yet reported a step.
                cur.execute(
                    "UPDATE jobs SET status=%s, last_progress_at=NOW() WHERE id=%s",
                    (status, job_id),
                )
            return True
    except Exception as exc:  # noqa: BLE001
        logger.error("update_job_status failed: %s", exc)
        return False


def save_job_metrics(job_id: str, metrics: dict) -> bool:
    if not is_database_available():
        with _memory_lock:
            _memory_metrics[job_id] = {
                "job_id": job_id,
                "total_frames": metrics.get("total_frames", 0),
                "good_frames": metrics.get("good_frames", 0),
                "bad_frames": metrics.get("bad_frames", 0),
                "processing_time_seconds": metrics.get("processing_time_seconds"),
                "metrics_json": metrics.get("metrics_json", {}),
                "created_at": datetime.now(),
            }
        return True

    try:
        with _conn() as c, c.cursor() as cur:
            cur.execute(
                """
                INSERT INTO job_metrics (job_id, total_frames, good_frames, bad_frames,
                                         processing_time_seconds, metrics_json)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (job_id) DO UPDATE SET
                    total_frames = EXCLUDED.total_frames,
                    good_frames = EXCLUDED.good_frames,
                    bad_frames = EXCLUDED.bad_frames,
                    processing_time_seconds = EXCLUDED.processing_time_seconds,
                    metrics_json = EXCLUDED.metrics_json
                """,
                (
                    job_id,
                    metrics.get("total_frames", 0),
                    metrics.get("good_frames", 0),
                    metrics.get("bad_frames", 0),
                    metrics.get("processing_time_seconds"),
                    json.dumps(metrics.get("metrics_json", {})),
                ),
            )
            return True
    except Exception as exc:  # noqa: BLE001
        logger.error("save_job_metrics failed: %s", exc)
        return False


def get_job_by_id(job_id: str) -> Optional[Dict[str, Any]]:
    if not is_database_available():
        with _memory_lock:
            job = _memory_jobs.get(job_id)
            if not job:
                return None
            merged = job.copy()
            if job_id in _memory_metrics:
                merged.update(_memory_metrics[job_id])
            return merged

    try:
        from psycopg2.extras import RealDictCursor

        with _conn() as c, c.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT j.*, m.total_frames, m.good_frames, m.bad_frames,
                       m.processing_time_seconds, m.metrics_json
                FROM jobs j LEFT JOIN job_metrics m ON j.id = m.job_id
                WHERE j.id = %s
                """,
                (job_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    except Exception as exc:  # noqa: BLE001
        logger.error("get_job_by_id failed: %s", exc)
        return None


def get_all_jobs(limit: int = 100, offset: int = 0, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Newest first. `offset` is threaded through both stores but no caller
    passes it (the jobs page and the export use the default 0)."""
    if not is_database_available():
        with _memory_lock:
            jobs = []
            for job_id, job in _memory_jobs.items():
                if status_filter and job.get("status") != status_filter:
                    continue
                merged = job.copy()
                if job_id in _memory_metrics:
                    merged.update(_memory_metrics[job_id])
                jobs.append(merged)
        jobs.sort(key=lambda j: j.get("created_at", datetime.min), reverse=True)
        return jobs[offset : offset + limit]

    try:
        from psycopg2.extras import RealDictCursor

        with _conn() as c, c.cursor(cursor_factory=RealDictCursor) as cur:
            base = """
                SELECT j.*, m.total_frames, m.good_frames, m.bad_frames,
                       m.processing_time_seconds, m.metrics_json
                FROM jobs j LEFT JOIN job_metrics m ON j.id = m.job_id
            """
            if status_filter:
                cur.execute(
                    base + " WHERE j.status = %s ORDER BY j.created_at DESC LIMIT %s OFFSET %s",
                    (status_filter, limit, offset),
                )
            else:
                cur.execute(
                    base + " ORDER BY j.created_at DESC LIMIT %s OFFSET %s",
                    (limit, offset),
                )
            return [dict(row) for row in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001
        logger.error("get_all_jobs failed: %s", exc)
        return []
