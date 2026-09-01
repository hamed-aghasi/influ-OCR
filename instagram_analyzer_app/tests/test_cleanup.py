"""Job working-directory cleanup, and classifier failure reported as its own
error rather than as an upload with no Insights screens."""

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


def _real_jpeg() -> bytes:
    import cv2
    import numpy as np

    ok, encoded = cv2.imencode(".jpg", np.zeros((10, 10, 3), np.uint8))
    assert ok
    return encoded.tobytes()


def test_classifier_total_failure_is_reported_as_an_error(tmp_path, monkeypatch):
    """A broken model must not look like 'no Insights screens in this upload'."""
    from processing import frame_classifier

    frames = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    payload = _real_jpeg()
    for f in frames:
        f.write_bytes(payload)

    def _boom(batch):
        raise RuntimeError("boom")

    monkeypatch.setattr(frame_classifier, "load_model", lambda: _boom)

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
    import numpy as np

    from processing import frame_classifier

    frames = [tmp_path / "a.jpg"]
    frames[0].write_bytes(_real_jpeg())

    monkeypatch.setattr(
        frame_classifier, "load_model", lambda: lambda batch: np.full((len(batch), 1), 0.1, np.float32)
    )

    result = frame_classifier.classify_frames(frames, job_id="j-nogood")

    assert "error" not in result
    assert result["good_paths"] == []
    assert len(result["bad_frames"]) == 1
