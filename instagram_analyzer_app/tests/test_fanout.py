"""Chord fan-out pieces: chunking, assembly, and finalize_job semantics."""

from datetime import date, datetime
from pathlib import Path


def test_chunk_batches_splits_evenly():
    from processing.gemini_processor import chunk_batches

    paths = [Path(f"{i}.jpg") for i in range(30)]
    batches = chunk_batches(paths, batch_size=12)
    assert [len(b) for b in batches] == [12, 12, 6]
    assert batches[0][0] == Path("0.jpg")
    assert chunk_batches([], batch_size=12) == []


def test_assemble_metrics_reports_batch_failures():
    from processing.gemini_processor import assemble_metrics

    m = assemble_metrics(
        [{"metrics": {"views": 100}}, {"metrics": {"views": 250, "likes": 9}}],
        total_frames=8,
        unique_frames=5,
        duplicate_frames=3,
        batches_total=3,
        batches_failed=1,
    )
    assert m["ocr_batches_total"] == 3
    assert m["ocr_batches_failed"] == 1
    assert m["summary"] == {"views": 250, "likes": 9}


def _context(tmp_path: Path) -> dict:
    return {
        "started_at": datetime.now().isoformat(),
        "upload_path": str(tmp_path / "upload.bin"),
        "output_dir": str(tmp_path / "out"),
        "total_frames": 10,
        "good_frames": 8,
        "bad_frames": 2,
        "ocr_input_frames": 8,
        "unique_frames": 5,
        "duplicate_frames": 3,
        "batches_total": 2,
    }


def test_finalize_job_fails_job_when_batches_dropped(tmp_path):
    from processing.db_client import _memory_jobs, _memory_metrics, create_job, get_job_by_id
    from tasks import finalize_job

    _memory_jobs.clear()
    _memory_metrics.clear()
    assert create_job("j-drop", date(2026, 1, 1), "c", "p", "co", "f.mp4", "video")

    outcomes = [
        {"batch_index": 0, "failed": False, "results": [{"frame_index": 0, "metrics": {"views": 100}}]},
        {"batch_index": 1, "failed": True, "results": []},
    ]
    result = finalize_job(outcomes, "j-drop", _context(tmp_path))

    assert result["status"] == "failed"
    job = get_job_by_id("j-drop")
    assert job["status"] == "failed"
    assert "1/2 OCR batches failed" in job["error_message"]
    # Partial metrics are still persisted for inspection.
    assert job["metrics_json"]["summary"] == {"views": 100}
    assert (tmp_path / "out" / "instagram_metrics.json").exists()


def test_finalize_job_completes_when_all_batches_ok(tmp_path):
    from processing.db_client import _memory_jobs, _memory_metrics, create_job, get_job_by_id
    from tasks import finalize_job

    _memory_jobs.clear()
    _memory_metrics.clear()
    assert create_job("j-ok", date(2026, 1, 1), "c", "p", "co", "f.mp4", "video")

    outcomes = [
        {"batch_index": 1, "failed": False, "results": [{"frame_index": 0, "metrics": {"views": 250}}]},
        {"batch_index": 0, "failed": False, "results": [{"frame_index": 0, "metrics": {"views": 100, "likes": 4}}]},
    ]
    result = finalize_job(outcomes, "j-ok", _context(tmp_path))

    assert result["status"] == "completed"
    job = get_job_by_id("j-ok")
    assert job["status"] == "completed"
    assert job["metrics_json"]["summary"] == {"views": 250, "likes": 4}
    assert job["metrics_json"]["ocr_batches_failed"] == 0
