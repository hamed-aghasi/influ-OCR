"""Batching, persistence, and the legacy sequential pipeline (reachable only
through processing._EXPORTS; the Celery chord in tasks.py is the live path)."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..config import settings
from ..errors import APIError
from ..logger import job_logger
from ..s3_storage import is_s3_configured, upload_json
from .aggregate import assemble_metrics
from .client import ocr_single_batch

def persist_metrics(final_metrics: Dict, output_dir: Optional[Path], job_id: Optional[str]) -> None:
    log = job_logger("processing.ocr_processor", job_id)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(output_dir / "instagram_metrics.json", "w", encoding="utf-8") as f:
                json.dump(final_metrics, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            log.warning("Could not write metrics file: %s", exc)

    if job_id and is_s3_configured():
        upload_json(job_id, final_metrics)


def chunk_batches(paths: List[Path], batch_size: Optional[int] = None) -> List[List[Path]]:
    size = batch_size or settings.ocr_batch_size
    return [paths[i : i + size] for i in range(0, len(paths), size)]


def process_frames(
    frame_paths: List[Path],
    output_dir: Optional[Path] = None,
    job_id: Optional[str] = None,
    skip_dedup: bool = False,
) -> Dict:
    """Sequential pipeline (dedup → batched OCR → persist). The Celery chord
    path in tasks.py runs the same pieces with batches in parallel."""
    log = job_logger("processing.ocr_processor", job_id)

    if not settings.openrouter_api_key:
        raise APIError(503, "OPENROUTER_API_KEY not configured")

    if skip_dedup:
        unique_paths = list(frame_paths)
        duplicate_paths: List[Path] = []
    else:
        from ..dedup import dedupe_frames  # local: only this legacy path needs imagehash

        unique_paths, duplicate_paths = dedupe_frames(frame_paths)
        log.info("Dedup: %d unique / %d duplicates from %d input", len(unique_paths), len(duplicate_paths), len(frame_paths))

    all_results: List[Dict] = []
    batches = chunk_batches(unique_paths)
    batches_failed = 0

    for i, batch in enumerate(batches):
        log.info("OCR batch %d/%d (%d frames)", i + 1, len(batches), len(batch))
        results = ocr_single_batch(batch, job_id)
        if results is None:
            batches_failed += 1
            continue
        all_results.extend(results)
        if i + 1 < len(batches):
            time.sleep(settings.ocr_delay_seconds)

    final_metrics = assemble_metrics(
        all_results,
        total_frames=len(frame_paths),
        unique_frames=len(unique_paths),
        duplicate_frames=len(duplicate_paths),
        batches_total=len(batches),
        batches_failed=batches_failed,
    )
    persist_metrics(final_metrics, output_dir, job_id)

    log.info("OCR complete: %d frame results, %d summary fields", len(all_results), len(final_metrics["summary"]))
    return final_metrics


def extract_metrics_from_paths(
    frame_paths: List[Path],
    output_dir: Optional[Path] = None,
    job_id: Optional[str] = None,
) -> Dict:
    """Convenience entry point used by the worker."""
    if not frame_paths:
        return {"error": "No frames", "total_frames": 0}
    return process_frames(frame_paths, output_dir, job_id)
