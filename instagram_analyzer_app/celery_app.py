"""Celery application instance.

Two queues: `celery` (default — run_job, finalize_job, beat tasks) and `ocr`
(ocr_batch only, routed below). Run one worker per queue:
    celery -A celery_app.celery worker --loglevel=info --concurrency=2 -Q celery
    celery -A celery_app.celery worker --loglevel=info --concurrency=8 -Q ocr
"""

from celery import Celery

from processing.config import settings


celery = Celery(
    "instagram_analyzer",
    broker=settings.broker_url,
    backend=settings.result_backend,
    include=["tasks"],
)

celery.conf.update(
    task_track_started=True,
    task_acks_late=True,
    # An OOM-killed worker child must redeliver its task, not ack it into the
    # void — otherwise the job sits in 'processing' until the 2h reaper.
    # Worst case is one duplicate run, which rewrites the same job row.
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_time_limit=60 * 60,        # 1h hard limit
    task_soft_time_limit=55 * 60,   # 55m soft
    result_expires=60 * 60 * 24 * 7,
    broker_connection_retry_on_startup=True,
    # CPU-heavy pipeline tasks and I/O-bound OCR API calls must not compete
    # for the same worker slots; only ocr_batch is routed off the default queue.
    task_routes={"ocr_batch": {"queue": "ocr"}},
    # ponytail: rate_limit is per worker instance — global only while worker-ocr
    # is a single replica; swap for a redis token bucket in _call_api if
    # replicas are ever needed. Scale concurrency, not replicas.
    **(
        {"task_annotations": {"ocr_batch": {"rate_limit": settings.ocr_rate_limit}}}
        if settings.ocr_rate_limit
        else {}
    ),
    beat_schedule={
        "reap-stale-jobs": {
            "task": "reap_stale_jobs",
            "schedule": settings.reaper_interval_seconds,
        },
        **(
            {
                "poll-minio": {
                    "task": "poll_minio",
                    "schedule": settings.minio_poll_interval_seconds,
                }
            }
            if settings.minio_ingest_configured
            else {}
        ),
    },
)
