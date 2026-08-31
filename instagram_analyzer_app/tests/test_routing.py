"""Queue routing: ocr_batch runs on its own queue; everything else defaults
to `celery` so the chord callback and beat tasks are never orphaned."""


def test_ocr_batch_routed_to_ocr_queue():
    from celery_app import celery

    assert celery.conf.task_routes == {"ocr_batch": {"queue": "ocr"}}


def test_rate_limit_disabled_by_default():
    from celery_app import celery

    assert not (celery.conf.task_annotations or {}).get("ocr_batch")
