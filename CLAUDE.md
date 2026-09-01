# influ-OCR — Instagram Campaign Analyzer

> Session state + open threads: see `HANDOFF.md` (dated 2026-08-22).

FastAPI app that extracts Instagram Insights engagement metrics (views, reach,
likes, shares, etc.) from uploaded screenshots/screen-recordings. Pipeline:
frame extraction (OpenCV) -> TensorFlow MobileNetV2 frame classifier -> OCR via
an OpenRouter vision model (default google/gemini-3.7-flash) -> results in Postgres, optional S3, Excel
export. Supports English and Persian Insights screens. Project is dormant
(last commit 2025-12-28).

## One codebase (verified 2026-08-22)

- `instagram_analyzer_app/` — the app, tracked by git. The 2026-05-05 refactor
  (security hardening, Celery + Redis worker, Alembic, phash dedup,
  docker-compose) was merged to main via PR #1 (commit 2867be4). A leftover
  nested copy (`influ-OCR-refactor/`) was deleted 2026-08-22.

## Running

Original app:
```bash
pip install -r requirements.txt
# .env needs DATABASE_URL, OPENROUTER_API_KEY, SECRET_KEY, ADMIN_PASSWORD (see README.md)
uvicorn instagram_analyzer_app.main:app --host 0.0.0.0 --port 8000
```

Preferred: `docker compose up --build` from repo root
(brings up postgres, redis, alembic migrate, app on :8000, celery worker).

## Where things live

- OCR / OpenRouter calls: `instagram_analyzer_app/processing/ocr_processor.py`
- Frame extraction: `processing/frame_extractor.py`; classifier: `processing/frame_classifier.py`
- S3: `processing/s3_storage.py`; DB: `processing/db_client.py`
- Routes/auth/upload flow: `main.py`; TF model weights: `models/`
- (Same layout inside `influ-OCR-refactor/instagram_analyzer_app/`, plus
  `tasks.py` / `celery_app.py` and `alembic.ini`.)

## Docs status

- `PROXY_SUPPORT_PLAN.md` — proposal only, NEVER implemented (see its status header).
- `CODE_REVIEW.md` — 2024-12-24 review of the original app; the refactor repo
  addressed much of it. Do not treat its findings as current for the refactor.
