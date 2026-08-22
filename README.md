# Instagram Campaign Analyzer

FastAPI app that reads engagement metrics off Instagram Insights screenshots
and screen-recordings using a TF classifier + Gemini-via-OpenRouter for OCR.

## What it does

1. User uploads a video, image, or ZIP of screenshots through the web UI.
2. Frames are extracted (every Nth frame; videos optionally downscaled to 720p).
3. A TensorFlow MobileNetV2 classifier filters out blurred/transition frames.
4. Surviving frames are de-duplicated by perceptual hash.
5. Unique frames are OCR'd in batches by Gemini via OpenRouter.
6. Aggregated metrics are saved to Postgres and (optionally) to Liara S3.
7. User can browse jobs, view per-campaign metrics, and export everything to Excel.

Persian and English Insights screens are both supported.

## Running locally

```bash
cp instagram_analyzer_app/.env.example .env   # edit, especially SECRET_KEY and OPENROUTER_API_KEY
docker compose up --build
```

That brings up:

| Service  | What                                             |
|----------|--------------------------------------------------|
| postgres | Persistent storage                               |
| redis    | Celery broker + result backend                   |
| migrate  | One-shot `alembic upgrade head`                  |
| app      | FastAPI on `http://localhost:8000`               |
| worker   | Celery worker that runs the ingest pipeline      |
| beat     | Celery beat: reaps jobs stuck queued/processing  |

Default login: `admin` / whatever you set as `ADMIN_PASSWORD`.

## Required environment variables

| Var                  | Required | Notes                                             |
|----------------------|----------|---------------------------------------------------|
| `SECRET_KEY`         | yes      | Session signing key, ≥ 16 chars                   |
| `DATABASE_URL`       | yes      | Postgres DSN                                      |
| `REDIS_URL`          | yes      | Defaults to `redis://redis:6379/0`                |
| `OPENROUTER_API_KEY` | yes for OCR | Without it, classification still works         |
| `OPENROUTER_MODEL`   | no       | Default `qwen/qwen3.8-max`                 |
| `ADMIN_USERNAME`     | no       | Default `admin`                                   |
| `ADMIN_PASSWORD`     | no       | Default `admin123` — change before any real use   |
| `MINIO_ENDPOINT`     | optional | All four MINIO_* set → auto-ingest poller active  |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` / `MINIO_BUCKET` | optional | Poller watches `incoming/<company>/<campaign>/<product>/<file>`, claims via `ingesting/`, settles to `processed/` or `failed/` |
| `LIARA_ENDPOINT`     | optional | All four LIARA_* must be set for S3 to be active  |
| `LIARA_ACCESS_KEY`   | optional |                                                   |
| `LIARA_SECRET_KEY`   | optional |                                                   |
| `LIARA_BUCKET_NAME`  | optional |                                                   |

## HTTP surface

| Path                                 | Method   | Auth | Notes                                  |
|--------------------------------------|----------|------|----------------------------------------|
| `/login` / `/logout`                 | GET/POST | —    | Session-cookie auth                    |
| `/`                                  | GET      | yes  | Upload form                            |
| `/upload`                            | POST     | yes  | Validates size+MIME, enqueues Celery job |
| `/status/{job_id}`                   | GET      | yes  | Auto-refreshes while processing        |
| `/jobs`                              | GET      | yes  | List + filter by status                |
| `/export`                            | GET      | yes  | Excel of all jobs                      |
| `/health`                            | GET      | —    | Liveness                               |
| `/api/job/{job_id}`                  | GET      | yes  | Job JSON                               |
| `/api/job/{job_id}/metrics`          | GET      | yes  | Pulls JSON from S3                     |
| `/api/job/{job_id}/metrics/download` | GET      | yes  | Redirects to a presigned S3 URL        |

## Project layout

```
instagram_analyzer_app/
├── main.py                   FastAPI app
├── celery_app.py             Celery instance
├── tasks.py                  Celery tasks: prepare → parallel OCR batches (chord) → finalize
├── alembic.ini, alembic/     Schema migrations
├── processing/
│   ├── config.py             Centralized settings (pydantic-settings)
│   ├── logger.py             Logging with job_id binding
│   ├── errors.py             APIError + handler
│   ├── frame_extractor.py    ffmpeg + OpenCV frame sampling
│   ├── frame_classifier.py   TF SavedModel classifier
│   ├── dedup.py              Perceptual-hash deduplication
│   ├── gemini_processor.py   OCR via OpenRouter
│   ├── db_client.py          Postgres pool + in-memory fallback
│   └── s3_storage.py         Liara S3 wrapper
├── templates/, static/       Jinja2 + plain CSS UI
├── models/                   TF SavedModel + metadata
└── tests/                    Pytest smoke tests
```

## Migrations

Schema is owned by Alembic. The compose file runs `alembic upgrade head`
in a one-shot `migrate` service before the app and worker start. To create
new migrations:

```bash
docker compose run --rm migrate alembic revision -m "your change"
```

## Notes / known limitations

- The classifier `val_accuracy` of 1.0 in `models/model_metadata.json` is a
  signal of an easy validation set, not field accuracy. Re-evaluate before
  trusting it on novel screenshots.
- Redis is not given a persistent volume in this compose file. If you care
  about queue persistence across `docker compose down`, add one.
