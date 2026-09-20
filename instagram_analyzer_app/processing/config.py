"""Centralized application configuration.

All env-driven settings flow through `settings`. Magic numbers that were
previously scattered across modules live here. Import as
`from .config import settings`.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

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
    # The reaper fails a *processing* job with no progress for this long
    # (measured from jobs.last_progress_at, migration 0003 — not from
    # created_at, which failed healthy jobs for merely waiting in the queue).
    job_stale_seconds: int = 2 * 60 * 60
    # A *queued* job is only failed after this much longer wait: a backlog of
    # thousands against two worker slots means waiting is normal, so this
    # catches a genuinely lost broker message, nothing else.
    queue_stale_seconds: int = 24 * 60 * 60
    reaper_interval_seconds: int = 15 * 60
    # The sweeper deletes job directories and uploads untouched for this long
    # whose job is no longer queued or processing. Without it, a job killed by
    # the Celery hard time limit (1h) leaves its frames behind for good:
    # finalize_job's cleanup never runs. Comfortably clear of that limit.
    orphan_max_age_seconds: int = 6 * 60 * 60
    sweeper_interval_seconds: int = 60 * 60

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
    # Tied to the deployed model, run 2026-09-19b-sources (good = a schema metric
    # is readable, Top sources of views included; plus round-01 review feedback;
    # training/relabel/2026-09-19b-sources/). Held-out recall is 1.000 from 0.05
    # to 0.85; 0.30 keeps a wide margin for ~6 more frames per 10 videos than
    # 0.85. End to end: ground truth 70 frames, no metric lost; round-01 feedback
    # 21/21 checks pass. Rolling back: backup_* in models/ (previous: run
    # 2026-09-19-relabel, also at 0.30).
    classifier_threshold: float = 0.30
    classifier_image_size: int = 224
    dark_frame_threshold: int = 80

    # ----- Frame extraction -----
    frame_interval: int = 3
    convert_to_720p: bool = True
    ffmpeg_timeout_seconds: int = 600

    # ----- Dedup -----
    # 16/16, not the historical 8/4. An 8x8 hash reduces a frame to 64
    # low-frequency bits, and in Instagram screen recordings the story strip,
    # header and nav bar are pixel-identical throughout while the numbers that
    # differ are small text — exactly the detail that resolution discards.
    # Measured across 58 videos: 59% of multi-frame clusters held screens
    # showing *different* values, so dedup was deleting metrics before OCR ever
    # saw them. On one video a loading skeleton, "Views 22,360", a 24-donut,
    # "Back 2,507" and an Audience split were all merged into one cluster.
    #
    # 16/16 is the same 6.25% bit tolerance as 4/64, just computed at higher
    # resolution. Against pixel-level ground truth it keeps genuinely identical
    # frames collapsed as well as before (93%) while separating clearly
    # different frames far better (56% -> 90%). Expect more frames to survive
    # to OCR, which costs more per video and is the intended trade.
    perceptual_hash_size: int = 16
    perceptual_hash_distance: int = 16

    # ----- Frame selection (which good frames go to OCR) -----
    # "scroll": reconstruct the scroll, propose the fewest frames covering each
    # page, then verify every good frame against the picks row by row
    # (processing/frame_selection.py). "phash": the dedup above only.
    # Measured against the hand-built ground truth (training/ground_truth/, 10
    # videos, 113 metric appearances): scroll sent 192 frames and lost 0
    # metrics; phash sent 217 and lost 2 (a whole-image hash merged two screens
    # differing only in the Profile visits / Follows row). Re-run
    # training/eval_selection.py before changing either.
    frame_selection: Literal["phash", "scroll"] = "scroll"
    # Then keep only the fewest selected frames that together show every
    # metric, read locally with Tesseract (processing/metric_coverage.py).
    # A final pass drops frames whose content is readable in another sent frame
    # (processing/selection_pipeline.py). Ground truth: no metric lost.
    # Needs the tesseract binary with these languages (installed in the
    # Dockerfile); if missing, the step is skipped and logged, never fatal.
    coverage_selection: bool = True
    coverage_ocr_lang: str = "fas+eng"

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
