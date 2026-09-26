"""The OpenRouter call: one pooled session per thread, retries, truncation splitting,
and the parse of the structured response."""

import base64
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from ..config import settings
from ..errors import APIError
from ..logger import job_logger
from .prompt import EXTRACTION_PROMPT
from .schema import RESPONSE_FORMAT, FrameResults

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


_KEY_NOISE = re.compile(r"^[^A-Za-z0-9_]+|[^A-Za-z0-9_]+$")


def _normalise_keys(node: Any) -> Any:
    """Strip stray punctuation from object keys, recursively.

    The strict schema is not enforced by every provider. qwen3.8-max-0902
    answered a Final_Test batch with every key written as `": views"`, which
    pydantic silently ignored — 28 nulls per frame, reported as success while
    the numbers were right there under the noisy keys."""
    if isinstance(node, dict):
        return {_KEY_NOISE.sub("", k) if isinstance(k, str) else k: _normalise_keys(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_normalise_keys(v) for v in node]
    return node


def _parse_content(text: str) -> Optional[List[Dict]]:
    """Parse + validate a structured-output response body.

    Tolerates the two departures from the schema seen from a provider that
    does not enforce it: a bare array where the `{"frames": [...]}` envelope
    should be, and punctuation inside keys. Returns None on anything else
    malformed or truncated — callers treat that as a batch failure to retry,
    never as partial data to keep.
    """
    try:
        doc = json.loads(text)
    except ValueError:
        return None
    if isinstance(doc, list):
        doc = {"frames": doc}
    try:
        return [f.model_dump() for f in FrameResults.model_validate(_normalise_keys(doc)).frames]
    except Exception:  # noqa: BLE001 — schema mismatch of any kind
        return None


def _all_null(results: List[Dict]) -> bool:
    """Every frame came back with no metric at all. Coverage only sends frames
    it read a metric on, so a whole batch of nulls is a failed answer, not an
    empty screen — it must be retried, not recorded as a completed job."""
    return bool(results) and all(
        not any(v is not None for v in (r.get("metrics") or {}).values()) for r in results
    )


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
    second = _call_api(encoded[mid:], log)
    if second is not None:
        for r in second:
            if isinstance(r.get("frame_index"), int):
                r["frame_index"] += mid
    if first is None and second is None:
        return None
    if first is None or second is None:
        # Keep the half that answered. Discarding it too turned one
        # unreadable frame into a whole batch — and a whole job — lost.
        lost = encoded[:mid] if first is None else encoded[mid:]
        log.error("%d frame(s) dropped after truncation: %s",
                  len(lost), ", ".join(name for name, _ in lost))
    return (first or []) + (second or [])


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
    if settings.openrouter_reasoning_effort:
        body["reasoning"] = {"effort": settings.openrouter_reasoning_effort}

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
                choice = response.json()["choices"][0]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                log.warning(
                    "Attempt %d returned no usable choice (%s): %s",
                    attempt + 1, exc, response.text[:300],
                )
                time.sleep(5)
                continue

            if choice.get("finish_reason") == "length":
                return _split_on_truncation(encoded, log)

            validated = _parse_content((choice.get("message") or {}).get("content") or "")
            if validated is None:
                log.warning("Attempt %d returned unparseable payload; retrying", attempt + 1)
                time.sleep(5)
                continue
            if _all_null(validated):
                log.warning("Attempt %d returned no metric on any of %d frame(s); retrying",
                            attempt + 1, len(validated))
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


def ocr_single_batch(batch: List[Path], job_id: Optional[str] = None) -> Optional[List[Dict]]:
    """OCR one batch of frames. Returns frame results with `actual_frame`
    attached, or None if the batch failed after retries."""
    log = job_logger("processing.ocr_processor", job_id)

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
    answered = {r["actual_frame"] for r in results if r.get("actual_frame")}
    missing = [p.name for p in batch if p.name not in answered]
    if missing:
        log.error("OCR returned nothing for %d of %d frame(s): %s", len(missing), len(batch), ", ".join(missing))
    return results


def frames_missing(batch: List[Path], results: Optional[List[Dict]]) -> int:
    """How many of `batch` have no result — dropped after truncation, or never
    answered. The job is marked incomplete for these, like a failed batch."""
    if results is None:
        return len(batch)
    answered = {r.get("actual_frame") for r in results}
    return sum(1 for p in batch if p.name not in answered)
