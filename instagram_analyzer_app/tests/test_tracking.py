"""Job tracking: queued lifecycle, task-id storage, progress, stale reaper."""

from datetime import date, datetime, timedelta


def _fresh_job(job_id: str):
    from processing.db_client import create_job

    assert create_job(job_id, date(2026, 1, 1), "camp", "prod", "co", "f.mp4", "video")


def test_job_starts_queued_and_tracks_task_id_and_progress():
    from processing.db_client import (
        _memory_jobs,
        get_job_by_id,
        set_job_task_id,
        update_job_progress,
    )

    _memory_jobs.clear()
    _fresh_job("j-track")

    job = get_job_by_id("j-track")
    assert job["status"] == "queued"
    assert job["celery_task_id"] is None

    assert set_job_task_id("j-track", "celery-abc-123")
    assert update_job_progress("j-track", "OCR batch 1/3 done")

    job = get_job_by_id("j-track")
    assert job["celery_task_id"] == "celery-abc-123"
    assert job["progress"] == "OCR batch 1/3 done"


def test_tracking_updates_fail_for_unknown_job():
    from processing.db_client import _memory_jobs, set_job_task_id, update_job_progress

    _memory_jobs.clear()
    assert not set_job_task_id("ghost", "x")
    assert not update_job_progress("ghost", "y")


def test_reaper_fails_only_stale_active_jobs():
    from processing.db_client import _memory_jobs, fail_stale_jobs, get_job_by_id

    _memory_jobs.clear()
    _fresh_job("stale-queued")
    _fresh_job("fresh-processing")
    _fresh_job("stale-completed")

    two_hours_ago = datetime.now() - timedelta(hours=2)
    _memory_jobs["stale-queued"]["created_at"] = two_hours_ago
    _memory_jobs["fresh-processing"]["status"] = "processing"
    _memory_jobs["stale-completed"]["created_at"] = two_hours_ago
    _memory_jobs["stale-completed"]["status"] = "completed"

    assert fail_stale_jobs(max_age_seconds=3600) == 1

    assert get_job_by_id("stale-queued")["status"] == "failed"
    assert "timed out" in get_job_by_id("stale-queued")["error_message"].lower()
    assert get_job_by_id("fresh-processing")["status"] == "processing"
    assert get_job_by_id("stale-completed")["status"] == "completed"
