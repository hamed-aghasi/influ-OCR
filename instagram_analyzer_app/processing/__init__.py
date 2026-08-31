"""Processing pipeline.

Public surface used by the API and the Celery worker:
- frame extraction (`extract_frames_from_video`, `process_campaign_zip`)
- classification (`classify_frames`)
- OCR (`extract_metrics_from_paths`)
- persistence (`create_job`, `update_job_status`, etc.)
- S3 helpers

Names are resolved lazily (PEP 562). Eager re-exports here meant that
`from processing.config import settings` executed this module and pulled in
tensorflow, imagehash, psycopg2 and boto3 — which broke the test suite on any
interpreter missing one of them, and slowed every worker cold start. Submodules
are imported on first attribute access instead, then cached in globals().
"""

import importlib
from typing import Any, Dict, List

_EXPORTS: Dict[str, str] = {
    "settings": "config",
    "APIError": "errors",
    "api_error_handler": "errors",
    "extract_frames_from_video": "frame_extractor",
    "process_campaign_zip": "frame_extractor",
    "classify_frames": "frame_classifier",
    "load_model": "frame_classifier",
    "extract_metrics_from_paths": "gemini_processor",
    "process_frames": "gemini_processor",
    "create_job": "db_client",
    "create_user": "db_client",
    "export_to_excel": "db_client",
    "get_all_jobs": "db_client",
    "get_job_by_id": "db_client",
    "get_user_count": "db_client",
    "save_job_metrics": "db_client",
    "update_job_status": "db_client",
    "verify_user": "db_client",
    "download_json": "s3_storage",
    "get_file_url": "s3_storage",
    "is_s3_configured": "s3_storage",
    "upload_json": "s3_storage",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module_name}", __name__), name)
    globals()[name] = value  # subsequent lookups skip __getattr__ entirely
    return value


def __dir__() -> List[str]:
    return sorted(__all__)
