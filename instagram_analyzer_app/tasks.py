"""Celery tasks: long-running ingest pipeline.

The HTTP layer creates the job row (status 'queued') and enqueues `run_job`.
`run_job` prepares frames (extract → classify → dedup → chunk) and fans the
batches out as a chord of `ocr_batch` tasks; `finalize_job` fires once all
batches are done, aggregates, persists, and sets the final status. Batches
run in parallel across whatever worker capacity exists — parallelism is a
deploy knob (worker concurrency / replicas), not code.

`reap_stale_jobs` runs on beat and fails jobs stuck in queued/processing,
so lost broker messages can't leave immortal spinners.
"""

import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from celery import chord

from celery_app import celery
from processing import ingest
from processing.config import settings
from processing.db_client import (
    create_job,
    fail_stale_jobs,
    save_job_metrics,
    set_job_task_id,
    update_job_progress,
    update_job_status,
)
from processing.naming import generate_job_id
from processing.dedup import dedupe_frames
from processing.frame_classifier import classify_frames
from processing.frame_extractor import extract_frames_from_video, process_campaign_zip
from processing.ocr_processor import (
    assemble_metrics,
    chunk_batches,
    ocr_single_batch,
    persist_metrics,
)
from processing.logger import job_logger


def _cleanup_upload(file_path: str) -> None:
    try:
        p = Path(file_path)
        if p.exists():
            p.unlink()
    except OSError:
        pass


def _cleanup_job_dir(dir_path: str) -> None:
    """Delete a job's working directory (extracted frames, summaries).

    Without this the processing volume grows without bound — one long video at
    frame_interval=3 leaves hundreds of JPEGs behind, and once the disk fills
    every subsequent job fails at extraction. Metrics are already persisted to
    Postgres (and S3 when configured) before this runs, so nothing unique is
    lost. Refuses to touch anything outside settings.processing_dir.
    """
    if not dir_path:
        return
    if settings.keep_job_frames:
        job_logger("tasks").info("KEEP_JOB_FRAMES set — retaining %s", dir_path)
        return
    try:
        target = Path(dir_path).resolve()
        root = settings.processing_dir.resolve()
        if target == root or root not in target.parents:
            job_logger("tasks").warning("Refusing to clean up %s: outside %s", target, root)
            return
        shutil.rmtree(target, ignore_errors=True)
    except OSError as exc:
        job_logger("tasks").warning("Could not clean up %s: %s", dir_path, exc)


@celery.task(name="run_job", bind=True, max_retries=0)
def run_job(self, job_id: str, file_path: str, file_type: str, object_key: str = None) -> dict:
    log = job_logger("tasks", job_id)
    file_p = Path(file_path)
    output_dir = settings.processing_dir / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        update_job_status(job_id, "processing")
        update_job_progress(job_id, "Extracting frames")

        if file_type == "zip":
            extract = process_campaign_zip(file_p, settings.processing_dir, job_id)
            frame_paths: List[Path] = [Path(p) for p in extract.get("frame_paths", [])]
            output_dir = Path(extract.get("output_directory", output_dir))
        elif file_type == "video":
            _, frame_paths = extract_frames_from_video(file_p, output_dir / "frames", job_id=job_id)
        else:
            frame_paths = [file_p]

        log.info("Extracted %d frames", len(frame_paths))
        update_job_progress(job_id, f"Classifying {len(frame_paths)} frames")

        classification = classify_frames(frame_paths, output_dir=output_dir, job_id=job_id) if frame_paths else {
            "good_paths": [],
            "good_frames": [],
            "bad_frames": [],
        }
        if classification.get("error"):
            # An infrastructure failure (model missing, keras absent, OOM) — not
            # an upload with no Insights screens. Reporting them alike sends
            # debugging in the wrong direction.
            raise RuntimeError(f"Frame classifier unavailable: {classification['error']}")

        good_paths: List[Path] = classification.get("good_paths", [])
        log.info("Classified: good=%d bad=%d", len(good_paths), len(classification.get("bad_frames", [])))

        update_job_progress(job_id, "Deduplicating frames")
        unique_paths, duplicate_paths = dedupe_frames(good_paths)
        batches = chunk_batches(unique_paths)
        log.info(
            "Dedup: %d unique / %d duplicates from %d good frames -> %d OCR batches",
            len(unique_paths), len(duplicate_paths), len(good_paths), len(batches),
        )

        context = {
            "started_at": datetime.now().isoformat(),
            "upload_path": file_path,
            "object_key": object_key,
            "output_dir": str(output_dir),
            "total_frames": len(frame_paths),
            "good_frames": len(classification.get("good_frames", [])),
            "bad_frames": len(classification.get("bad_frames", [])),
            "ocr_input_frames": len(good_paths),
            "unique_frames": len(unique_paths),
            "duplicate_frames": len(duplicate_paths),
            "batches_total": len(batches),
        }

        if not batches:
            # Nothing to OCR — finalize directly (chord over an empty group misbehaves).
            finalize_job.delay([], job_id, context)
            return {"status": "dispatched", "batches": 0}

        update_job_progress(job_id, f"OCR: 0/{len(batches)} batches done")
        chord(
            [
                ocr_batch.s(job_id, [str(p) for p in batch], i, len(batches))
                for i, batch in enumerate(batches)
            ],
            finalize_job.s(job_id, context),
        ).delay()
        return {"status": "dispatched", "batches": len(batches)}

    except Exception as exc:  # noqa: BLE001 — we want to record any failure
        log.exception("Job preparation failed: %s", exc)
        update_job_status(job_id, "failed", str(exc))
        _cleanup_upload(file_path)
        _cleanup_job_dir(str(output_dir))
        if object_key:
            ingest.settle_object(object_key, success=False)
        return {"status": "failed", "error": str(exc)}


@celery.task(name="ocr_batch", bind=True, max_retries=0)
def ocr_batch(self, job_id: str, paths: List[str], batch_index: int, batches_total: int) -> Dict:
    """OCR one batch. Never raises: a raised exception inside a chord header
    would abort the callback, so failures are returned as data instead."""
    log = job_logger("tasks", job_id)
    results: Optional[List[Dict]] = None
    try:
        results = ocr_single_batch([Path(p) for p in paths], job_id)
    except Exception as exc:  # noqa: BLE001
        log.exception("OCR batch %d crashed: %s", batch_index, exc)

    failed = results is None
    update_job_progress(
        job_id,
        f"OCR batch {batch_index + 1}/{batches_total} {'failed' if failed else 'done'}",
    )
    return {"batch_index": batch_index, "failed": failed, "results": results or []}


@celery.task(name="finalize_job", bind=True, max_retries=0)
def finalize_job(self, batch_outcomes: List[Dict], job_id: str, context: Dict) -> dict:
    log = job_logger("tasks", job_id)
    job_succeeded = False
    try:
        update_job_progress(job_id, "Finalizing")
        outcomes = sorted(batch_outcomes, key=lambda o: o.get("batch_index", 0))
        all_results = [r for o in outcomes for r in o.get("results", [])]
        batches_failed = sum(1 for o in outcomes if o.get("failed"))

        final_metrics = assemble_metrics(
            all_results,
            total_frames=context["ocr_input_frames"],
            unique_frames=context["unique_frames"],
            duplicate_frames=context["duplicate_frames"],
            batches_total=context["batches_total"],
            batches_failed=batches_failed,
        )
        persist_metrics(final_metrics, Path(context["output_dir"]), job_id)

        started = datetime.fromisoformat(context["started_at"])
        elapsed = int((datetime.now() - started).total_seconds())
        save_job_metrics(
            job_id,
            {
                "total_frames": context["total_frames"],
                "good_frames": context["good_frames"],
                "bad_frames": context["bad_frames"],
                "processing_time_seconds": elapsed,
                "metrics_json": final_metrics,
            },
        )

        if batches_failed:
            message = (
                f"{batches_failed}/{context['batches_total']} OCR batches failed; "
                "extracted metrics are incomplete"
            )
            update_job_status(job_id, "failed", message)
            log.error("Job finished with dropped OCR batches: %s", message)
            return {"status": "failed", "error": message, "elapsed_seconds": elapsed}

        if context["ocr_input_frames"] == 0:
            # Zero frames classified as Insights screens: nothing was extracted,
            # so a green status would misreport an empty result as success.
            message = "No Instagram Insights screens detected in this upload"
            update_job_status(job_id, "failed", message)
            log.error(message)
            return {"status": "failed", "error": message, "elapsed_seconds": elapsed}

        update_job_status(job_id, "completed")
        log.info("Job complete in %ds", elapsed)
        job_succeeded = True
        return {"status": "completed", "elapsed_seconds": elapsed}

    except Exception as exc:  # noqa: BLE001
        log.exception("Finalize failed: %s", exc)
        update_job_status(job_id, "failed", str(exc))
        return {"status": "failed", "error": str(exc)}
    finally:
        _cleanup_upload(context.get("upload_path", ""))
        _cleanup_job_dir(context.get("output_dir", ""))
        if context.get("object_key"):
            ingest.settle_object(context["object_key"], success=job_succeeded)


@celery.task(name="reap_stale_jobs")
def reap_stale_jobs() -> int:
    count = fail_stale_jobs(settings.job_stale_seconds)
    if count:
        job_logger("tasks").warning("Reaper failed %d stale job(s)", count)
    return count


@celery.task(name="poll_minio", bind=True, max_retries=0)
def poll_minio(self) -> dict:
    """Scan incoming/ in the configured MinIO bucket, claim + enqueue new files.

    Runs on beat every MINIO_POLL_INTERVAL_SECONDS; no-ops when unconfigured.
    """
    if not ingest.is_ingest_configured():
        return {"status": "unconfigured"}

    log = job_logger("tasks")
    enqueued, rejected = 0, 0

    try:
        objects = ingest.list_incoming()
    except Exception as exc:  # noqa: BLE001 — endpoint down must not kill beat
        log.error("MinIO poll failed to list incoming/: %s", exc)
        return {"status": "error", "error": str(exc)}

    for obj in objects:
        key, size = obj["key"], obj["size"]
        meta = ingest.parse_object_key(key)

        if meta is None or size > settings.max_upload_bytes:
            reason = "bad path/extension" if meta is None else f"too large ({size} bytes)"
            log.warning("Rejecting %s: %s", key, reason)
            ingest.reject_object(key)
            rejected += 1
            continue

        claimed_key = ingest.claim_object(key)
        if claimed_key is None:
            continue  # another poller got it, or transient error — retry next tick

        job_id = generate_job_id(meta["company"], meta["campaign_name"])
        upload_path = settings.upload_dir / f"{job_id}{meta['ext']}"

        try:
            ingest.download_object(claimed_key, upload_path)
        except Exception as exc:  # noqa: BLE001
            log.error("Download failed for %s: %s", claimed_key, exc)
            ingest.settle_object(claimed_key, success=False)
            continue

        create_job(
            job_id,
            date.today(),
            meta["campaign_name"],
            meta["product_name"],
            meta["company"],
            key,  # original object key recorded as the job's filename
            meta["file_type"],
        )
        result = run_job.delay(job_id, str(upload_path), meta["file_type"], claimed_key)
        set_job_task_id(job_id, result.id)
        enqueued += 1
        log.info("Ingested %s as job %s", key, job_id)

    if enqueued or rejected:
        log.info("MinIO poll: %d enqueued, %d rejected", enqueued, rejected)
    return {"status": "ok", "enqueued": enqueued, "rejected": rejected}
