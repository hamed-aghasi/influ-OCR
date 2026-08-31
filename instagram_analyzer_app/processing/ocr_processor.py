"""OCR through OpenRouter.

Model-agnostic: the request is a plain OpenAI-compatible chat completion
(messages + base64 image_url parts + a json_schema response_format), so the
model is a config value (`OPENROUTER_MODEL`, currently qwen/qwen3.8-max) and
nothing here is specific to any provider. Named gemini_processor.py until
2026-08-31, when the default had already been qwen for over a week.

Frames are deduplicated before this stage by `dedup.py`, so the LLM
no longer needs to do that. Each batch is asked only to read metric
values off the (already-unique) frames it gets.
"""

import base64
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from pydantic import BaseModel, Field, field_validator

from .config import settings
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


# No real Instagram metric exceeds this; observed model failure mode is
# digit-repetition hallucinations hundreds of digits long.
_MAX_PLAUSIBLE_VALUE = 10_000_000_000


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

    @field_validator("*", mode="after")
    @classmethod
    def _drop_implausible(cls, v, info):
        if v is None:
            return None
        if info.field_name in {"followers", "non_followers"}:  # percentages
            return v if 0 <= v <= 100 else None
        return v if 0 <= v <= _MAX_PLAUSIBLE_VALUE else None


class Metadata(BaseModel):
    language: Optional[str] = None
    date_range: Optional[str] = None
    content_type: Optional[str] = None


# NB: no docstring on the wire models below — pydantic copies class docstrings
# into the schema as `description`, which then ships to the model on every
# request. `actual_frame` is likewise absent by design: it is filled in locally
# in ocr_single_batch, so asking the model to invent a filename it cannot know
# would cost tokens per frame and be overwritten regardless.
class FrameResult(BaseModel):
    frame_index: int = Field(ge=0)
    metrics: Optional[Metrics] = None
    metadata: Optional[Metadata] = None


class FrameResults(BaseModel):
    # Response envelope enforced via OpenRouter structured outputs.
    frames: List[FrameResult] = []


# Validation keywords that OpenAI-style `strict` structured outputs reject.
# Dropping them from the *wire* schema costs nothing: the pydantic models
# still enforce every one of them locally in _parse_content.
_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
        "minLength", "maxLength", "pattern", "minItems", "maxItems", "format", "default",
    }
)


def _strict_schema(node):
    """Rewrite a pydantic JSON schema to satisfy `strict: true`.

    Strict mode requires every object to set `additionalProperties: false` and
    to list *all* its properties in `required`, and rejects the validation
    keywords above. pydantic emits none of that, so declaring strict over a raw
    model_json_schema() is a latent 400 — it happens to pass with the current
    model and would fail on the next one, retried 5x before failing the batch.
    """
    if isinstance(node, list):
        return [_strict_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    out = {k: _strict_schema(v) for k, v in node.items() if k not in _UNSUPPORTED_SCHEMA_KEYS}
    if out.get("type") == "object":
        out["additionalProperties"] = False
        out["required"] = list(out.get("properties") or {})
    return out


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "frame_results",
        "strict": True,
        "schema": _strict_schema(FrameResults.model_json_schema()),
    },
}


EXTRACTION_PROMPT = """\
You are reading Instagram Insights screenshots. For each frame, extract any
metric values visible. Numbers may be in English digits or Persian digits
(۰-۹ map to 0-9). If a metric is not visible, omit it (leave it null).

Metric keys to extract (omit any not present):
- views, followers, non_followers, accounts_reached
- interactions, likes, replies, shares
- links_clicks, sticker_taps, navigation
- forward, next_story, back, exited
- profile_activity, profile_visits, external_link_taps, follows

On-screen label mapping (English / Persian):
- "Viewers" / بینندگان -> accounts_reached (NOT views; the "Views" donut or
  Summary row is views)
- "Comments" / نظرها -> replies
- "Bio link clicks" / کلیک‌های پیوند بیو -> links_clicks
- followers / non_followers are the percentage split under the Views donut —
  report the percentage number (e.g. 62.2)

Also read the icon strip at the top of Reel insights screens (heart=likes,
speech bubble=replies, paper plane=shares) and the Engagement tab rows.
Expand K/M abbreviations to full numbers: 8.6K -> 8600, 1.2M -> 1200000.
Read carefully digit by digit; do not merge a number with a neighboring
number, and never repeat digits beyond what is on screen.

Respond with one entry in `frames` per input frame, in input order, using
each frame's zero-based position as `frame_index`.
"""


_thread_local = threading.local()


def _session() -> requests.Session:
    """One pooled Session per thread. requests.post() opens a fresh TCP+TLS
    connection per call; batches are megabytes of base64 and the round trip to
    openrouter.ai is long, so reusing the connection is worth real time.
    Per-thread rather than global because Session is not thread-safe.
    """
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def _encode(paths: List[Path], log) -> List[Tuple[str, str]]:
    encoded = []
    for path in paths:
        try:
            with open(path, "rb") as f:
                encoded.append((path.name, base64.b64encode(f.read()).decode("utf-8")))
        except OSError as exc:
            log.warning("Could not read %s: %s", path.name, exc)
    return encoded


def _parse_content(text: str) -> Optional[List[Dict]]:
    """Parse + validate a structured-output response body.

    Returns None on any malformed/truncated payload — callers treat that
    as a batch failure to retry, never as partial data to keep.
    """
    try:
        return [f.model_dump() for f in FrameResults.model_validate_json(text).frames]
    except Exception:  # noqa: BLE001 — invalid JSON and schema mismatch alike
        return None


def _split_on_truncation(encoded: List[Tuple[str, str]], log) -> Optional[List[Dict]]:
    """Retry a truncated batch as two halves instead of resending it unchanged.

    A response cut off at max_tokens will be cut off again for an identical
    request, so the old behaviour burned every remaining retry — five full
    multi-image vision calls — to reproduce one failure. Halving the frame
    count halves the output the model has to produce. frame_index is
    zero-based per request, so the second half's indices are rebased onto the
    original batch before the halves are joined.
    """
    if len(encoded) < 2:
        log.error("Response truncated on a single frame; batch cannot be split further")
        return None

    mid = len(encoded) // 2
    log.warning("Truncated at %d tokens; splitting %d frames into %d + %d",
                settings.ocr_max_tokens, len(encoded), mid, len(encoded) - mid)

    first = _call_api(encoded[:mid], log)
    if first is None:
        return None
    second = _call_api(encoded[mid:], log)
    if second is None:
        return None

    for r in second:
        if isinstance(r.get("frame_index"), int):
            r["frame_index"] += mid
    return first + second


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
        "max_tokens": settings.ocr_max_tokens,
        "response_format": RESPONSE_FORMAT,
    }

    for attempt in range(settings.ocr_max_retries):
        try:
            response = _session().post(
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
            # OpenRouter reports provider-side failures as HTTP 200 with an
            # `error` object and no `choices`. Reaching into the envelope
            # unguarded raised straight past this retry loop and killed the
            # batch — and, on the sequential path, the whole job.
            try:
                payload = response.json()
                choice = payload["choices"][0]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                log.warning(
                    "Attempt %d returned no usable choice (%s): %s",
                    attempt + 1, exc, response.text[:300],
                )
                time.sleep(5)
                continue

            # One parseable line per billable call; Grafana/Loki aggregates
            # these into the AI-cost dashboard. Logged before the truncation
            # check because truncated calls cost money too.
            usage = payload.get("usage") or {}
            log.info(
                "ai_call model=%s frames=%d prompt_tokens=%s completion_tokens=%s cost_usd=%s",
                settings.openrouter_model,
                len(encoded),
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                usage.get("cost"),
            )

            if choice.get("finish_reason") == "length":
                return _split_on_truncation(encoded, log)

            validated = _parse_content((choice.get("message") or {}).get("content") or "")
            if validated is None:
                log.warning("Attempt %d returned unparseable payload; retrying", attempt + 1)
                time.sleep(5)
                continue
            return validated

        if response.status_code == 429:
            wait = (attempt + 1) * 10
            log.warning("Rate limited; waiting %ds", wait)
            time.sleep(wait)
            continue

        log.error("API error %d: %s", response.status_code, response.text[:300])
        time.sleep(5)

    return None


def _aggregate(results: List[Dict]) -> Tuple[Dict[str, float], Dict[str, Dict]]:
    """Consensus per metric across frames: modal value if any read repeats,
    otherwise the median. Replaces max(), which let a single inflated
    misread win (observed in production: follows read as 3154, true 31).

    The no-agreement median is median_LOW, not median_high: with exactly two
    disagreeing reads — the common case, since most metrics survive dedup on
    only one or two screens — median_high returns the larger value and is
    therefore identical to the max() this was meant to replace. Misreads
    inflate (digit repetition, see _MAX_PLAUSIBLE_VALUE) far more often than
    they deflate, so ties break downward. For 3+ reads both pick the true
    middle element and the choice is immaterial.

    Returns (summary, quality) where quality records reads/min/max and a
    disputed flag for metrics whose reads disagree materially.
    """
    from collections import Counter
    from statistics import median_low

    summary: Dict[str, float] = {}
    quality: Dict[str, Dict] = {}
    for metric in _METRIC_FIELDS:
        values = []
        for r in results:
            metrics = r.get("metrics") or {}
            v = metrics.get(metric)
            if v is not None:
                values.append(v)
        if not values:
            continue
        top, freq = Counter(values).most_common(1)[0]
        chosen = top if freq > 1 else median_low(values)
        lo, hi = min(values), max(values)
        summary[metric] = chosen
        quality[metric] = {
            "reads": len(values),
            "min": lo,
            "max": hi,
            "disputed": (hi - lo) > max(0.05 * abs(hi), 1),
        }
    return summary, quality


def ocr_single_batch(batch: List[Path], job_id: Optional[str] = None) -> Optional[List[Dict]]:
    """OCR one batch of frames. Returns frame results with `actual_frame`
    attached, or None if the batch failed after retries."""
    log = job_logger(__name__, job_id)

    if not settings.openrouter_api_key:
        raise APIError(503, "OPENROUTER_API_KEY not configured")

    encoded = _encode(batch, log)
    if not encoded:
        log.error("OCR batch had no readable frames; counted as failed")
        return None

    results = _call_api(encoded, log)
    if results is None:
        log.error("OCR batch failed after retries; metrics will be incomplete")
        return None

    for r in results:
        idx = r.get("frame_index")
        # Always set the key: it is no longer part of the wire schema, so
        # downstream consumers would otherwise see it appear and disappear.
        r["actual_frame"] = batch[idx].name if isinstance(idx, int) and 0 <= idx < len(batch) else None
    return results


def assemble_metrics(
    all_results: List[Dict],
    total_frames: int,
    unique_frames: int,
    duplicate_frames: int,
    batches_total: int,
    batches_failed: int,
) -> Dict:
    summary, summary_quality = _aggregate(all_results)
    return {
        "extraction_date": datetime.now().isoformat(),
        "total_frames": total_frames,
        "unique_frames": unique_frames,
        "duplicate_frames": duplicate_frames,
        "ocr_batches_total": batches_total,
        "ocr_batches_failed": batches_failed,
        "all_frames_data": all_results,
        "summary": summary,
        "summary_quality": summary_quality,
    }


def persist_metrics(final_metrics: Dict, output_dir: Optional[Path], job_id: Optional[str]) -> None:
    log = job_logger(__name__, job_id)
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
    log = job_logger(__name__, job_id)

    if not settings.openrouter_api_key:
        raise APIError(503, "OPENROUTER_API_KEY not configured")

    if skip_dedup:
        unique_paths = list(frame_paths)
        duplicate_paths: List[Path] = []
    else:
        from .dedup import dedupe_frames  # local: only this legacy path needs imagehash

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
