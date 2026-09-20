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


def test_reaper_fails_a_processing_job_with_no_progress():
    from processing.db_client import _memory_jobs, fail_stale_jobs, get_job_by_id

    _memory_jobs.clear()
    _fresh_job("stalled")
    _fresh_job("done")

    two_hours_ago = datetime.now() - timedelta(hours=2)
    _memory_jobs["stalled"]["status"] = "processing"
    _memory_jobs["stalled"]["last_progress_at"] = two_hours_ago
    _memory_jobs["done"]["status"] = "completed"
    _memory_jobs["done"]["last_progress_at"] = two_hours_ago

    assert fail_stale_jobs(max_age_seconds=3600, queued_max_age_seconds=86400) == 1

    assert get_job_by_id("stalled")["status"] == "failed"
    assert "no progress" in get_job_by_id("stalled")["error_message"].lower()
    assert get_job_by_id("done")["status"] == "completed"


def test_reaper_spares_a_long_job_that_is_still_reporting_progress():
    """The bug this fixes: the old rule keyed on created_at, so a job that had
    been running for longer than the limit was failed even while it was
    actively working."""
    from processing.db_client import _memory_jobs, fail_stale_jobs, get_job_by_id

    _memory_jobs.clear()
    _fresh_job("long-but-alive")
    _memory_jobs["long-but-alive"]["status"] = "processing"
    _memory_jobs["long-but-alive"]["created_at"] = datetime.now() - timedelta(hours=9)
    _memory_jobs["long-but-alive"]["last_progress_at"] = datetime.now() - timedelta(minutes=2)

    assert fail_stale_jobs(max_age_seconds=3600, queued_max_age_seconds=86400) == 0
    assert get_job_by_id("long-but-alive")["status"] == "processing"


def test_reaper_does_not_fail_a_job_for_waiting_in_the_queue():
    """The other half of the bug: with two worker slots and a backlog of
    thousands, queued jobs routinely wait longer than the processing limit.
    They were being marked failed and then completing anyway."""
    from processing.db_client import _memory_jobs, fail_stale_jobs, get_job_by_id

    _memory_jobs.clear()
    _fresh_job("waiting")
    _memory_jobs["waiting"]["created_at"] = datetime.now() - timedelta(hours=5)

    assert fail_stale_jobs(max_age_seconds=3600, queued_max_age_seconds=86400) == 0
    assert get_job_by_id("waiting")["status"] == "queued"


def test_reaper_fails_a_job_queued_past_the_long_limit():
    """The safety net still exists: a broker message that was genuinely lost
    leaves a job queued forever, and that is what the long limit catches."""
    from processing.db_client import _memory_jobs, fail_stale_jobs, get_job_by_id

    _memory_jobs.clear()
    _fresh_job("lost")
    _memory_jobs["lost"]["created_at"] = datetime.now() - timedelta(hours=30)

    assert fail_stale_jobs(max_age_seconds=3600, queued_max_age_seconds=86400) == 1
    assert get_job_by_id("lost")["status"] == "failed"
    assert "never started" in get_job_by_id("lost")["error_message"].lower()


def test_progress_and_status_updates_stamp_liveness():
    from processing.db_client import (
        _memory_jobs,
        fail_stale_jobs,
        update_job_progress,
        update_job_status,
    )

    _memory_jobs.clear()
    _fresh_job("ticking")
    _memory_jobs["ticking"]["status"] = "processing"
    _memory_jobs["ticking"]["last_progress_at"] = datetime.now() - timedelta(hours=3)

    assert update_job_progress("ticking", "OCR: 2/5 batches done")
    assert fail_stale_jobs(max_age_seconds=3600, queued_max_age_seconds=86400) == 0

    _memory_jobs["ticking"]["last_progress_at"] = datetime.now() - timedelta(hours=3)
    assert update_job_status("ticking", "processing")
    assert fail_stale_jobs(max_age_seconds=3600, queued_max_age_seconds=86400) == 0


def test_reaper_falls_back_to_created_at_for_rows_predating_the_column():
    """Rows written before migration 0003 have last_progress_at NULL; the
    in-memory store models that with None."""
    from processing.db_client import _memory_jobs, fail_stale_jobs, get_job_by_id

    _memory_jobs.clear()
    _fresh_job("legacy")
    _memory_jobs["legacy"]["status"] = "processing"
    _memory_jobs["legacy"]["created_at"] = datetime.now() - timedelta(hours=4)
    _memory_jobs["legacy"]["last_progress_at"] = None

    assert fail_stale_jobs(max_age_seconds=3600, queued_max_age_seconds=86400) == 1
    assert get_job_by_id("legacy")["status"] == "failed"
