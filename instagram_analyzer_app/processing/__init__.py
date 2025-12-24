"""
Processing Module

Contains all processing logic:
- frame_extractor: Extract frames from videos
- frame_classifier: ML-based frame classification
- gemini_processor: OCR using Gemini API
- db_client: Database operations and Excel export
"""

from .frame_extractor import process_campaign_zip, extract_frames_from_video
from .frame_classifier import classify_frames, load_model
from .gemini_processor import extract_metrics_from_good_frames, process_frames
from .db_client import (
    init_database,
    create_job,
    update_job_status,
    save_job_metrics,
    get_all_jobs,
    get_job_by_id,
    export_to_excel,
    create_user,
    verify_user,
    get_user_count
)
from .s3_storage import (
    upload_json,
    download_json,
    get_file_url,
    is_s3_configured
)

__all__ = [
    # Frame extraction
    'process_campaign_zip',
    'extract_frames_from_video',
    # Classification
    'classify_frames',
    'load_model',
    # OCR
    'extract_metrics_from_good_frames',
    'process_frames',
    # Database
    'init_database',
    'create_job',
    'update_job_status',
    'save_job_metrics',
    'get_all_jobs',
    'get_job_by_id',
    'export_to_excel',
    # Authentication
    'create_user',
    'verify_user',
    'get_user_count',
    # S3 Storage
    'upload_json',
    'download_json',
    'get_file_url',
    'is_s3_configured',
]
