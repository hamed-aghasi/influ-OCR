"""Job working-directory cleanup, the orphan sweeper, and classifier failure
reported as its own error rather than as an upload with no Insights screens."""

import time
from pathlib import Path

import pytest


def _make_job_dir(root: Path, job_id: str) -> Path:
    d = root / job_id / "frames"
    d.mkdir(parents=True)
    (d / "frame_000001.jpg").write_bytes(b"not-a-real-jpeg")
    (d / "frame_000002.jpg").write_bytes(b"not-a-real-jpeg")
    return root / job_id


def test_cleanup_job_dir_removes_extracted_frames(tmp_path, monkeypatch):
    from processing.config import settings
    import tasks

    monkeypatch.setattr(settings, "processing_dir", tmp_path)
    job_dir = _make_job_dir(tmp_path, "job-1")
    assert job_dir.exists()

    tasks._cleanup_job_dir(str(job_dir))

    assert not job_dir.exists()
    assert tmp_path.exists()  # the root itself survives


def test_cleanup_job_dir_refuses_paths_outside_processing_dir(tmp_path, monkeypatch):
    from processing.config import settings
    import tasks

    processing_root = tmp_path / "processing"
    processing_root.mkdir()
    monkeypatch.setattr(settings, "processing_dir", processing_root)

    outsider = tmp_path / "somewhere_else"
    outsider.mkdir()
    (outsider / "precious.txt").write_text("do not delete")

    tasks._cleanup_job_dir(str(outsider))
    assert (outsider / "precious.txt").exists()

    # the processing root itself is not a job dir and must survive too
    tasks._cleanup_job_dir(str(processing_root))
    assert processing_root.exists()


def test_cleanup_job_dir_honours_keep_job_frames(tmp_path, monkeypatch):
    from processing.config import settings
    import tasks

    monkeypatch.setattr(settings, "processing_dir", tmp_path)
    monkeypatch.setattr(settings, "keep_job_frames", True)
    job_dir = _make_job_dir(tmp_path, "job-keep")

    tasks._cleanup_job_dir(str(job_dir))

    assert (job_dir / "frames" / "frame_000001.jpg").exists()


def test_cleanup_job_dir_tolerates_missing_and_empty(tmp_path, monkeypatch):
    from processing.config import settings
    import tasks

    monkeypatch.setattr(settings, "processing_dir", tmp_path)
    tasks._cleanup_job_dir("")  # no output_dir recorded in context
    tasks._cleanup_job_dir(str(tmp_path / "never-existed"))


def test_classifier_total_failure_is_reported_as_an_error(tmp_path, monkeypatch):
    """A broken model must not look like 'no Insights screens in this upload'."""
    from processing import frame_classifier

    frames = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    for f in frames:
        f.write_bytes(b"garbage")

    monkeypatch.setattr(frame_classifier, "load_model", lambda: object())
    monkeypatch.setattr(
        frame_classifier, "_classify_one", lambda p, m: (None, 0.0, "inference failed: boom")
    )

    result = frame_classifier.classify_frames(frames, job_id="j-broken")

    assert result["error"] == "classifier failed on all 2 frames"
    assert result["good_paths"] == []
    assert len(result["failed_frames"]) == 2
    assert result["failed_frames"][0]["error"] == "inference failed: boom"


def test_classifier_error_fails_the_job_with_a_distinct_message(tmp_path, monkeypatch):
    import tasks

    monkeypatch.setattr(tasks, "update_job_status", lambda *a, **k: True)
    monkeypatch.setattr(tasks, "update_job_progress", lambda *a, **k: True)
    monkeypatch.setattr(
        tasks, "classify_frames", lambda *a, **k: {"error": "model could not be loaded"}
    )

    upload = tmp_path / "shot.png"
    upload.write_bytes(b"x")
    monkeypatch.setattr(tasks.settings, "processing_dir", tmp_path / "processing")

    result = tasks.run_job("j-x", str(upload), "image")

    assert result["status"] == "failed"
    assert "classifier unavailable" in result["error"].lower()
    assert "no instagram insights screens" not in result["error"].lower()


def test_healthy_classification_does_not_set_an_error(tmp_path, monkeypatch):
    """Zero good frames with no failures is a real 'no screens' result, not an
    infrastructure error — the two must stay distinguishable."""
    from processing import frame_classifier

    frames = [tmp_path / "a.jpg"]
    frames[0].write_bytes(b"garbage")

    monkeypatch.setattr(frame_classifier, "load_model", lambda: object())
    monkeypatch.setattr(frame_classifier, "_classify_one", lambda p, m: ("BAD", 0.1, ""))

    result = frame_classifier.classify_frames(frames, job_id="j-nogood")

    assert "error" not in result
    assert result["good_paths"] == []
    assert len(result["bad_frames"]) == 1


# ----- the sweeper: files left by jobs that died without settling -----
#
# finalize_job's `finally` is the only cleanup on the normal path, and a hard
# Celery time limit, an OOM kill or a lost chord message all skip it. These
# pin that the sweeper collects what is left without ever touching live work.

def _age(path: Path, seconds: float) -> None:
    """Backdate everything in a tree, so it looks untouched."""
    import os

    stamp = time.time() - seconds
    for p in [path, *path.rglob("*")]:
        os.utime(p, (stamp, stamp))


def _sweeper_env(tmp_path, monkeypatch, active=()):
    from processing.config import settings
    import tasks

    processing = tmp_path / "processing"
    uploads = tmp_path / "uploads"
    processing.mkdir()
    uploads.mkdir()
    monkeypatch.setattr(settings, "processing_dir", processing)
    monkeypatch.setattr(settings, "upload_dir", uploads)
    monkeypatch.setattr(settings, "keep_job_frames", False)
    monkeypatch.setattr(settings, "orphan_max_age_seconds", 3600)
    monkeypatch.setattr(tasks, "active_job_ids", lambda: set(active))
    return processing, uploads


def test_sweeper_removes_an_orphaned_job_directory(tmp_path, monkeypatch):
    import tasks

    processing, _ = _sweeper_env(tmp_path, monkeypatch)
    orphan = _make_job_dir(processing, "dead-job")
    _age(orphan, 7200)

    result = tasks.sweep_orphans()

    assert not orphan.exists()
    assert result["dirs"] == 1
    assert processing.exists()  # the root survives


def test_sweeper_keeps_a_directory_belonging_to_a_live_job(tmp_path, monkeypatch):
    """A job can be mid-OCR with its frames untouched for hours; deleting them
    would destroy work in flight."""
    import tasks

    processing, _ = _sweeper_env(tmp_path, monkeypatch, active={"running-job"})
    live = _make_job_dir(processing, "running-job")
    _age(live, 7200)

    result = tasks.sweep_orphans()

    assert live.exists()
    assert result["dirs"] == 0


def test_sweeper_keeps_recently_touched_directories(tmp_path, monkeypatch):
    import tasks

    processing, _ = _sweeper_env(tmp_path, monkeypatch)
    recent = _make_job_dir(processing, "just-started")

    assert tasks.sweep_orphans()["dirs"] == 0
    assert recent.exists()


def test_sweeper_notices_new_frames_inside_an_old_directory(tmp_path, monkeypatch):
    """Writing frames into <job>/frames/ leaves <job>/ itself untouched, so a
    directory mtime alone would call a busy job an orphan."""
    import os
    import tasks

    processing, _ = _sweeper_env(tmp_path, monkeypatch)
    job = _make_job_dir(processing, "busy-job")
    _age(job, 7200)
    fresh = job / "frames" / "frame_000003.jpg"
    fresh.write_bytes(b"just written")
    os.utime(fresh, None)

    assert tasks.sweep_orphans()["dirs"] == 0
    assert job.exists()


def test_sweeper_removes_orphaned_uploads(tmp_path, monkeypatch):
    import tasks

    _, uploads = _sweeper_env(tmp_path, monkeypatch, active={"live-job"})
    dead = uploads / "dead-job.mp4"
    dead.write_bytes(b"x" * 1024)
    live = uploads / "live-job.mp4"
    live.write_bytes(b"x" * 1024)
    _age(uploads, 7200)

    result = tasks.sweep_orphans()

    assert not dead.exists()
    assert live.exists()          # its job is still queued or processing
    assert result["uploads"] == 1
    assert result["bytes"] >= 1024


def test_sweeper_is_a_noop_when_keep_job_frames_is_set(tmp_path, monkeypatch):
    from processing.config import settings
    import tasks

    processing, _ = _sweeper_env(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "keep_job_frames", True)
    kept = _make_job_dir(processing, "evaluation-job")
    _age(kept, 7200)

    assert tasks.sweep_orphans()["status"] == "disabled"
    assert kept.exists()


def test_sweeper_keeps_everything_when_it_cannot_read_active_jobs(tmp_path, monkeypatch):
    """Without the active list it cannot tell a live job from a dead one, and
    deleting a live one is unrecoverable."""
    import tasks

    processing, _ = _sweeper_env(tmp_path, monkeypatch)

    def _boom():
        raise RuntimeError("database down")

    monkeypatch.setattr(tasks, "active_job_ids", _boom)
    orphan = _make_job_dir(processing, "dead-job")
    _age(orphan, 7200)

    assert tasks.sweep_orphans()["status"] == "error"
    assert orphan.exists()


def test_sweeper_never_follows_a_symlink_out_of_the_processing_dir(tmp_path, monkeypatch):
    import tasks

    processing, _ = _sweeper_env(tmp_path, monkeypatch)
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "keep.txt").write_text("do not delete")
    link = processing / "sneaky-job"
    link.symlink_to(outside, target_is_directory=True)
    _age(outside, 7200)

    tasks.sweep_orphans()

    assert (outside / "keep.txt").exists()
