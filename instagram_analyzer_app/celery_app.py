"""Celery application instance.

Run the worker with:
    celery -A celery_app.celery worker --loglevel=info --concurrency=2
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
    worker_prefetch_multiplier=1,
    task_time_limit=60 * 60,        # 1h hard limit
    task_soft_time_limit=55 * 60,   # 55m soft
    result_expires=60 * 60 * 24 * 7,
    broker_connection_retry_on_startup=True,
)
