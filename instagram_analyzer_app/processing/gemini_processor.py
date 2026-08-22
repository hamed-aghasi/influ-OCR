"""OCR via Gemini through OpenRouter.

Frames are deduplicated before this stage by `dedup.py`, so the LLM
no longer needs to do that. Each batch is asked only to read metric
values off the (already-unique) frames it gets.
"""

import base64
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from pydantic import BaseModel, Field

from .config import settings
from .dedup import dedupe_frames
from .errors import APIError
from .logger import job_logger
from .s3_storage import is_s3_configured, upload_json


_METRIC_FIELDS = (
    "views", "followers", "non_followers", "accounts_reached",
    "interactions", "likes", "replies", "shares",
    "links_clicks", "sticker_taps", "navigation",
    "forward", "next_story", "back", "exited",
    "profile_activity", "profile_visits", "external_link_taps", "follows",
)


class Metrics(BaseModel):
    views: Optional[int] = None
    followers: Optional[float] = None
    non_followers: Optional[float] = None
    accounts_reached: Optional[int] = None
    interactions: Optional[int] = None
    likes: Optional[int] = None
    replies: Optional[int] = None
    shares: Optional[int] = None
    links_clicks: Optional[int] = None
    sticker_taps: Optional[int] = None
    navigation: Optional[int] = None
    forward: Optional[int] = None
    next_story: Optional[int] = None
    back: Optional[int] = None
    exited: Optional[int] = None
    profile_activity: Optional[int] = None
    profile_visits: Optional[int] = None
    external_link_taps: Optional[int] = None
    follows: Optional[int] = None


class Metadata(BaseModel):
    language: Optional[str] = None
    date_range: Optional[str] = None
    content_type: Optional[str] = None


class FrameResult(BaseModel):
    frame_index: int = Field(ge=0)
    metrics: Optional[Metrics] = None
    metadata: Optional[Metadata] = None
    actual_frame: Optional[str] = None


EXTRACTION_PROMPT = """\
You are reading Instagram Insights screenshots. For each frame, extract any
metric values visible. Numbers may be in English digits or Persian digits
(۰-۹ map to 0-9). If a metric is not visible, omit it.

Metric keys to extract (omit any not present):
- views, followers, non_followers, accounts_reached
- interactions, likes, replies, shares
- links_clicks, sticker_taps, navigation
- forward, next_story, back, exited
- profile_activity, profile_visits, external_link_taps, follows

Return ONLY a JSON array, one entry per frame, in input order:
[
  {
    "frame_index": 0,
    "metrics": {"views": 1234, "likes": 30},
    "metadata": {"language": "fa", "content_type": "story"}
  }
]
"""


def _encode(paths: List[Path], log) -> List[Tuple[str, str]]:
    encoded = []
    for path in paths:
        try:
            with open(path, "rb") as f:
                encoded.append((path.name, base64.b64encode(f.read()).decode("utf-8")))
        except OSError as exc:
            log.warning("Could not read %s: %s", path.name, exc)
    return encoded


def _strip_fences(text: str) -> str:
    if "```json" in text:
        return text.split("```json", 1)[1].split("```", 1)[0]
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0]
    return text


def _call_api(encoded: List[Tuple[str, str]], log) -> Optional[List[Dict]]:
    if not settings.openrouter_api_key:
        log.error("OPENROUTER_API_KEY not configured")
        return None

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    content = [{"type": "text", "text": EXTRACTION_PROMPT}]
    for i, (name, b64) in enumerate(encoded):
        content.append({"type": "text", "text": f"\nFrame {i}: {name}"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    body = {
        "model": settings.openrouter_model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        "max_tokens": 4000,
    }

    for attempt in range(settings.ocr_max_retries):
        try:
            response = requests.post(
                settings.openrouter_url,
                headers=headers,
                json=body,
                timeout=settings.ocr_request_timeout,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            log.warning("Attempt %d transport error: %s", attempt + 1, exc)
            time.sleep(5)
            continue

        if response.status_code == 200:
            text = response.json()["choices"][0]["message"]["content"]
            cleaned = _strip_fences(text).strip()
            try:
                raw = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                log.error("JSON parse failed: %s", exc)
                return None
            validated = []
            for item in raw:
                try:
                    validated.append(FrameResult(**item).model_dump())
                except Exception:  # noqa: BLE001 — keep partial parses
                    validated.append(item)
            return validated

        if response.status_code == 429:
            wait = (attempt + 1) * 10
            log.warning("Rate limited; waiting %ds", wait)
            time.sleep(wait)
            continue

        log.error("API error %d: %s", response.status_code, response.text[:300])
        time.sleep(5)

    return None


def _aggregate(results: List[Dict]) -> Dict[str, int]:
    """For each metric, return the largest observed value across frames.

    Insights screens show running totals, so max ≈ final value. Frames are
    already deduplicated, but the same metric can appear on multiple
    different screens — taking the max is robust to either.
    """
    summary: Dict[str, int] = {}
    for metric in _METRIC_FIELDS:
        values = []
        for r in results:
            metrics = r.get("metrics") or {}
            v = metrics.get(metric)
            if v is not None:
                values.append(v)
        if values:
            summary[metric] = max(values)
    return summary


def process_frames(
    frame_paths: List[Path],
    output_dir: Optional[Path] = None,
    job_id: Optional[str] = None,
    skip_dedup: bool = False,
) -> Dict:
    log = job_logger(__name__, job_id)

    if not settings.openrouter_api_key:
        raise APIError(503, "OPENROUTER_API_KEY not configured")

    if skip_dedup:
        unique_paths = list(frame_paths)
        duplicate_paths: List[Path] = []
    else:
        unique_paths, duplicate_paths = dedupe_frames(frame_paths)
        log.info("Dedup: %d unique / %d duplicates from %d input", len(unique_paths), len(duplicate_paths), len(frame_paths))

    all_results: List[Dict] = []
    batch_size = settings.ocr_batch_size

    for start in range(0, len(unique_paths), batch_size):
        batch = unique_paths[start : start + batch_size]
        log.info("OCR batch %d-%d (%d frames)", start, start + len(batch) - 1, len(batch))

        encoded = _encode(batch, log)
        if not encoded:
            continue

        results = _call_api(encoded, log)
        if results is None:
            log.error("OCR batch failed; skipping batch")
            continue

        for r in results:
            idx = r.get("frame_index")
            if isinstance(idx, int) and 0 <= idx < len(batch):
                r["actual_frame"] = batch[idx].name
            all_results.append(r)

        if start + batch_size < len(unique_paths):
            time.sleep(settings.ocr_delay_seconds)

    final_metrics = {
        "extraction_date": datetime.now().isoformat(),
        "total_frames": len(frame_paths),
        "unique_frames": len(unique_paths),
        "duplicate_frames": len(duplicate_paths),
        "all_frames_data": all_results,
        "summary": _aggregate(all_results),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(output_dir / "instagram_metrics.json", "w", encoding="utf-8") as f:
                json.dump(final_metrics, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            log.warning("Could not write metrics file: %s", exc)

    if job_id and is_s3_configured():
        upload_json(job_id, final_metrics)

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
