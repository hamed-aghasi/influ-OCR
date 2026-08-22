"""Celery tasks: long-running ingest pipeline.

The HTTP layer creates the job row and enqueues `run_job` here. The worker
extracts frames → classifies → dedupes → OCRs → persists metrics. State is
written through the same db_client/s3 helpers used by the API.
"""

from datetime import datetime
from pathlib import Path
from typing import List

from celery_app import celery
from processing.config import settings
from processing.db_client import save_job_metrics, update_job_status
from processing.frame_classifier import classify_frames
from processing.frame_extractor import extract_frames_from_video, process_campaign_zip
from processing.gemini_processor import extract_metrics_from_paths
from processing.logger import job_logger


@celery.task(name="run_job", bind=True, max_retries=0)
def run_job(self, job_id: str, file_path: str, file_type: str) -> dict:
    log = job_logger("tasks", job_id)
    start_time = datetime.now()
    file_p = Path(file_path)
    output_dir = settings.processing_dir / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        update_job_status(job_id, "processing")

        if file_type == "zip":
            extract = process_campaign_zip(file_p, settings.processing_dir, job_id)
            frame_paths: List[Path] = [Path(p) for p in extract.get("frame_paths", [])]
            output_dir = Path(extract.get("output_directory", output_dir))
        elif file_type == "video":
            _, frame_paths = extract_frames_from_video(file_p, output_dir / "frames", job_id=job_id)
        else:
            frame_paths = [file_p]

        log.info("Extracted %d frames", len(frame_paths))

        classification = classify_frames(frame_paths, output_dir=output_dir, job_id=job_id) if frame_paths else {
            "good_paths": [],
            "good_frames": [],
            "bad_frames": [],
        }
        good_paths: List[Path] = classification.get("good_paths", [])
        log.info("Classified: good=%d bad=%d", len(good_paths), len(classification.get("bad_frames", [])))

        ocr = extract_metrics_from_paths(good_paths, output_dir, job_id) if good_paths else {}

        elapsed = int((datetime.now() - start_time).total_seconds())
        save_job_metrics(
            job_id,
            {
                "total_frames": len(frame_paths),
                "good_frames": len(classification.get("good_frames", [])),
                "bad_frames": len(classification.get("bad_frames", [])),
                "processing_time_seconds": elapsed,
                "metrics_json": ocr,
            },
        )
        update_job_status(job_id, "completed")
        log.info("Job complete in %ds", elapsed)
        return {"status": "completed", "elapsed_seconds": elapsed}

    except Exception as exc:  # noqa: BLE001 — we want to record any failure
        log.exception("Job failed: %s", exc)
        update_job_status(job_id, "failed", str(exc))
        return {"status": "failed", "error": str(exc)}
    finally:
        try:
            if file_p.exists():
                file_p.unlink()
        except OSError:
            pass
