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


from . import aggregate, batch, client, prompt, schema  # noqa: F401  (eager: same import footprint)
from .aggregate import _aggregate, assemble_metrics
from .batch import chunk_batches, extract_metrics_from_paths, persist_metrics, process_frames
from .client import (
    _call_api, _encode, _parse_content, _session, _split_on_truncation, _thread_local,
    ocr_single_batch,
)
from .prompt import EXTRACTION_PROMPT
from .schema import (
    RESPONSE_FORMAT, _MAX_PLAUSIBLE_VALUE, _METRIC_FIELDS, _PERCENT_FIELDS,
    _UNSUPPORTED_SCHEMA_KEYS, _strict_schema, FrameResult, FrameResults, Metadata, Metrics,
)

__all__ = [
    "EXTRACTION_PROMPT", "RESPONSE_FORMAT", "Metrics", "Metadata", "FrameResult", "FrameResults",
    "_METRIC_FIELDS", "_PERCENT_FIELDS", "_MAX_PLAUSIBLE_VALUE", "_UNSUPPORTED_SCHEMA_KEYS",
    "_strict_schema",
    "_thread_local", "_session", "_encode", "_parse_content", "_split_on_truncation", "_call_api",
    "ocr_single_batch",
    "_aggregate", "assemble_metrics",
    "persist_metrics", "chunk_batches", "process_frames", "extract_metrics_from_paths",
]
