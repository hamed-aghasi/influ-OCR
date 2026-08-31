"""Centralized application configuration.

All env-driven settings flow through `settings`. Magic numbers that were
previously scattered across modules live here. Import as
`from .config import settings`.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


APP_DIR = Path(__file__).parent.parent.resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ----- HTTP / app -----
    secret_key: str = Field(..., min_length=16)
    admin_username: str = "admin"
    admin_password: str = "admin123"
    port: int = 8000

    # ----- Database -----
    database_url: Optional[str] = None
    db_pool_min: int = 2
    db_pool_max: int = 10

    # ----- Background work -----
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: Optional[str] = None
    celery_result_backend: Optional[str] = None
    job_stale_seconds: int = 2 * 60 * 60  # reaper fails queued/processing jobs older than this
    reaper_interval_seconds: int = 15 * 60

    # ----- OCR -----
    openrouter_api_key: Optional[str] = None
    openrouter_url: str = "https://openrouter.ai/api/v1/chat/completions"
    openrouter_model: str = "qwen/qwen3.8-max"
    ocr_batch_size: int = 12
    ocr_delay_seconds: float = 2.0
    ocr_max_retries: int = 5
    ocr_request_timeout: int = 120  # qwen3.8-max batches can run close to 60s
    ocr_max_tokens: int = 8000

    # ----- Classifier -----
    model_dir: Path = APP_DIR / "models"
    classifier_threshold: float = 0.65
    classifier_image_size: int = 224
    dark_frame_threshold: int = 80

    # ----- Frame extraction -----
    frame_interval: int = 3
    convert_to_720p: bool = True
    ffmpeg_timeout_seconds: int = 600

    # ----- Dedup -----
    perceptual_hash_size: int = 8
    perceptual_hash_distance: int = 4

    # ----- Upload guards -----
    max_upload_bytes: int = 500 * 1024 * 1024  # 500 MB
    max_zip_entries: int = 2_000
    max_zip_uncompressed_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GB

    # ----- Storage -----
    upload_dir: Path = Path("/tmp/uploads")
    processing_dir: Path = Path("/tmp/processing")
    log_dir: Path = APP_DIR / "logs"
    # Job frames are deleted when the job settles, or the volume fills and every
    # later job fails at extraction. Set KEEP_JOB_FRAMES=true to retain them for
    # model evaluation (the 2026-08-22 eval sourced its ground truth this way).
    keep_job_frames: bool = False

    # ----- S3 (Liara) -----
    liara_endpoint: Optional[str] = None
    liara_access_key: Optional[str] = None
    liara_secret_key: Optional[str] = None
    liara_bucket_name: Optional[str] = None

    # ----- MinIO auto-ingest -----
    minio_endpoint: Optional[str] = None  # e.g. https://minio.example.com:9000
    minio_access_key: Optional[str] = None
    minio_secret_key: Optional[str] = None
    minio_bucket: Optional[str] = None
    minio_incoming_prefix: str = "incoming/"
    minio_ingesting_prefix: str = "ingesting/"
    minio_processed_prefix: str = "processed/"
    minio_failed_prefix: str = "failed/"
    minio_poll_interval_seconds: int = 60

    @property
    def broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url

    @property
    def s3_configured(self) -> bool:
        return all(
            [
                self.liara_endpoint,
                self.liara_access_key,
                self.liara_secret_key,
                self.liara_bucket_name,
            ]
        )

    @property
    def database_configured(self) -> bool:
        return bool(self.database_url and self.database_url.strip())

    @property
    def minio_ingest_configured(self) -> bool:
        return all(
            [
                self.minio_endpoint,
                self.minio_access_key,
                self.minio_secret_key,
                self.minio_bucket,
            ]
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
