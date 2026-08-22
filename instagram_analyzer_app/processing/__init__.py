"""Processing pipeline.

Public surface used by the API and the Celery worker:
- frame extraction (`extract_frames_from_video`, `process_campaign_zip`)
- classification (`classify_frames`)
- OCR (`extract_metrics_from_paths`)
- persistence (`create_job`, `update_job_status`, etc.)
- S3 helpers
"""

from .config import settings
from .errors import APIError, api_error_handler
from .frame_extractor import extract_frames_from_video, process_campaign_zip
from .frame_classifier import classify_frames, load_model
from .gemini_processor import extract_metrics_from_paths, process_frames
from .db_client import (
    create_job,
    create_user,
    export_to_excel,
    get_all_jobs,
    get_job_by_id,
    get_user_count,
    save_job_metrics,
    update_job_status,
    verify_user,
)
from .s3_storage import (
    download_json,
    get_file_url,
    is_s3_configured,
    upload_json,
)

__all__ = [
    "settings",
    "APIError",
    "api_error_handler",
    "extract_frames_from_video",
    "process_campaign_zip",
    "classify_frames",
    "load_model",
    "extract_metrics_from_paths",
    "process_frames",
    "create_job",
    "create_user",
    "export_to_excel",
    "get_all_jobs",
    "get_job_by_id",
    "get_user_count",
    "save_job_metrics",
    "update_job_status",
    "verify_user",
    "download_json",
    "get_file_url",
    "is_s3_configured",
    "upload_json",
]
