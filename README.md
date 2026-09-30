# Instagram Campaign Analyzer (influ-OCR)

FastAPI + Celery app that reads engagement metrics off Instagram Insights screen
recordings and screenshots — English and Persian — using a TensorFlow frame
classifier, local Tesseract for frame selection, and an OpenRouter vision model
for the actual reading. Results land in Postgres and an Excel export.

- **Run it:** `RUNBOOK.md` (install, network requirements, batch runs, exports, failure modes).
- **How it works, stage by stage:** `PIPELINE.md`.
- **Retrain the classifier:** `training/README.md`.
- **Command-line tools** (batch submit, local run, self-check, output verification): `tools/README.md`.
- **History and decisions:** `docs/history/`.

## What it does

1. User uploads a video, image, or ZIP of screenshots through the web UI (or `tools/submit_batch.py`).
2. Frames are extracted (720p, every 3rd frame — `processing/frame_extractor.py`).
3. A MobileNetV2 classifier keeps frames that show a schema metric (`processing/frame_classifier.py`, `models/`).
4. Scroll-aware selection keeps the fewest frames covering each scrolled page (`processing/frame_selection/`).
5. Metric coverage (local Tesseract, Persian + English) keeps the fewest frames that together show
   every metric found, then drops frames whose content another sent frame already shows
   (`processing/metric_coverage/`; steps 4–5 run through `processing/selection_pipeline.choose_ocr_frames`).
6. Those frames go to OpenRouter in batches (`processing/ocr_processor/`); per-frame readings are
   aggregated by consensus (modal value, else low median; disagreements flagged `disputed`).
7. Metrics are saved to Postgres (`job_metrics.metrics_json`) and optionally to Liara S3.
8. Users browse jobs, view per-campaign metrics, and export everything to Excel.

Measured cost with the default model (`qwen/qwen3.8-27b`, reasoning `low`): 19 videos =
51,888 prompt + 38,618 completion tokens ≈ $0.17, i.e. **about $9 per 1,000 videos** at the
2026-09-29 OpenRouter prices. The provider and token usage of every answer is in the worker log.

## Running locally

```bash
cp instagram_analyzer_app/.env.example .env   # set SECRET_KEY, OPENROUTER_API_KEY, ADMIN_PASSWORD
docker compose up --build
```

That brings up:

| Service  | What                                             |
|----------|--------------------------------------------------|
| postgres | Persistent storage                               |
| redis    | Celery broker + result backend                   |
| migrate  | One-shot `alembic upgrade head`                  |
| app      | FastAPI on `http://localhost:8010`               |
| worker   | Celery worker that runs the ingest pipeline (2 slots) |
| beat     | Celery beat: reaps stalled jobs, sweeps orphaned files, polls MinIO if configured |

Login: `admin` / `admin123` **unless `ADMIN_PASSWORD` is set** — set it before anyone else can
reach the port. Every `docker compose` subcommand (even `ps`) needs `SECRET_KEY` in the
environment or in `.env`, because the compose file refuses to interpolate without it.
The worker container is what talks to OpenRouter, so its network route is the one that matters
(see `RUNBOOK.md` — the edge refuses some source addresses with HTTP 403).

Tests: `cd instagram_analyzer_app && python -m pytest tests/` (hermetic: no TensorFlow, Tesseract,
network or Postgres needed). The image ships pytest on purpose, so the same suite runs inside it:
`docker run --rm --entrypoint python influ-ocr:latest -m pytest tests -q`.

## Settings

Every field of `processing/config.py:Settings`, settable as an environment variable of the same
name (upper case) or in `.env`. Defaults are the code's; documenting them here changes nothing.

**Web**

| Var | Default | Notes |
|---|---|---|
| `SECRET_KEY` | *(required)* | Session signing key, ≥ 16 chars |
| `ADMIN_USERNAME` | `admin` | Seeded on first start |
| `ADMIN_PASSWORD` | `admin123` | **Change it.** `.env.example` uses `please-change-me` |
| `PORT` | `8000` | Only `python main.py` honours it; gunicorn binds 8000 and compose maps `127.0.0.1:8010` |

**Database, queue, background jobs**

| Var | Default | Notes |
|---|---|---|
| `DATABASE_URL` | *(none)* | Postgres DSN. Unset → in-memory store (tests only; compose always sets it) |
| `DB_POOL_MIN` / `DB_POOL_MAX` | `2` / `10` | psycopg2 pool size |
| `REDIS_URL` | `redis://redis:6379/0` | Broker + result backend unless the two below are set |
| `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` | *(none)* | Override `REDIS_URL` for either role |
| `JOB_STALE_SECONDS` | `7200` | Reaper fails a *processing* job with no progress for this long |
| `QUEUE_STALE_SECONDS` | `86400` | Reaper fails a *queued* job older than this (lost broker message) |
| `REAPER_INTERVAL_SECONDS` | `900` | Beat schedule for the reaper |
| `ORPHAN_MAX_AGE_SECONDS` | `21600` | Sweeper deletes job dirs/uploads untouched this long whose job is not active |
| `SWEEPER_INTERVAL_SECONDS` | `3600` | Beat schedule for the sweeper |

**OpenRouter OCR**

| Var | Default | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | *(none)* | Required for OCR; without it every job fails at the OCR stage |
| `OPENROUTER_URL` | `https://openrouter.ai/api/v1/chat/completions` | |
| `OPENROUTER_MODEL` | `qwen/qwen3.8-27b` | Always an explicit catalogue id, never an alias (`qwen/qwen3.8-max` once drifted to a different model) |
| `OPENROUTER_REASONING_EFFORT` | `low` | Sent as `reasoning: {"effort"}`; empty string omits it. Off, the model filled absent metrics with 0 |
| `OCR_BATCH_SIZE` | `12` | Frames per request |
| `OCR_MAX_TOKENS` | `8000` | Reply budget; a truncated reply is split into two half-batches |
| `OCR_MAX_RETRIES` | `5` | Per request; 429 backs off 10 s × attempt, 403 drops the pooled connection first |
| `OCR_REQUEST_TIMEOUT` | `120` | Seconds; a 6–8 frame batch with reasoning takes 60–90 s |
| `OCR_DELAY_SECONDS` | `2.0` | Only used by the unused sequential path (see *Unused code*) |

**Frame extraction and classifier**

| Var | Default | Notes |
|---|---|---|
| `FRAME_INTERVAL` | `3` | Keep every Nth frame |
| `CONVERT_TO_720P` | `true` | ffmpeg downscale before sampling |
| `FFMPEG_TIMEOUT_SECONDS` | `600` | |
| `MODEL_DIR` | `instagram_analyzer_app/models` | SavedModel directory |
| `CLASSIFIER_THRESHOLD` | `0.30` | Tied to the deployed model (run `2026-09-26-final-test`); see the comment in `config.py` before changing |
| `CLASSIFIER_IMAGE_SIZE` | `224` | |
| `DARK_FRAME_THRESHOLD` | `80` | Mean luma below which a frame is brightened before classification |

**Frame selection**

| Var | Default | Notes |
|---|---|---|
| `FRAME_SELECTION` | `scroll` | `scroll` (default) or `phash` |
| `COVERAGE_SELECTION` | `true` | Tesseract coverage pass; skipped with a log line if tesseract/`fas`/`eng` are missing |
| `COVERAGE_OCR_LANG` | `fas+eng` | |
| `PERCEPTUAL_HASH_SIZE` / `PERCEPTUAL_HASH_DISTANCE` | `16` / `16` | Used by `phash` selection only; the long comment in `config.py` explains why not 8/4 |

**Uploads and storage**

| Var | Default | Notes |
|---|---|---|
| `MAX_UPLOAD_BYTES` | `524288000` (500 MB) | |
| `MAX_ZIP_ENTRIES` / `MAX_ZIP_UNCOMPRESSED_BYTES` | `2000` / `2147483648` (2 GB) | ZIP bomb guards |
| `UPLOAD_DIR` / `PROCESSING_DIR` / `LOG_DIR` | `/tmp/uploads` / `/tmp/processing` / `instagram_analyzer_app/logs` | The Dockerfile pins all three under `/tmp` and compose mounts volumes there |
| `KEEP_JOB_FRAMES` | `false` | `true` keeps each job's extracted frames and `instagram_metrics.json` (evaluation only; fills the disk) |

**Liara S3 (optional)** — `LIARA_ENDPOINT`, `LIARA_ACCESS_KEY`, `LIARA_SECRET_KEY`,
`LIARA_BUCKET_NAME`: all four set → metrics JSON also uploaded per job and the
`/api/job/{id}/metrics` routes work.

**MinIO auto-ingest (optional)** — `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`,
`MINIO_BUCKET`: all four set → beat polls `incoming/<company>/<campaign>/<product>/<file>` every
`MINIO_POLL_INTERVAL_SECONDS` (`60`), claims via `MINIO_INGESTING_PREFIX` (`ingesting/`), settles to
`MINIO_PROCESSED_PREFIX` (`processed/`) or `MINIO_FAILED_PREFIX` (`failed/`); `MINIO_INCOMING_PREFIX`
defaults to `incoming/`. Enabling it needs a beat restart (the schedule is built at start).

## HTTP surface

| Path                                 | Method   | Auth | Notes                                  |
|--------------------------------------|----------|------|----------------------------------------|
| `/login` / `/logout`                 | GET/POST | —    | Session-cookie auth                    |
| `/`                                  | GET      | yes  | Upload form                            |
| `/upload`                            | POST     | yes  | Validates size+MIME, enqueues Celery job, 303 → `/status/{job_id}` |
| `/status/{job_id}`                   | GET      | yes  | Auto-refreshes while processing        |
| `/jobs`                              | GET      | yes  | List + filter by status                |
| `/export`                            | GET      | yes  | Excel of all jobs (one sheet, 14 job columns + 28 metric columns) |
| `/health`                            | GET      | —    | Liveness only (checks nothing else)    |
| `/api/job/{job_id}`                  | GET      | yes  | Job row + `metrics_json` (the full metrics document) |
| `/api/job/{job_id}/metrics`          | GET      | yes  | Pulls JSON from S3 (503 unless Liara is configured) |
| `/api/job/{job_id}/metrics/download` | GET      | yes  | Redirects to a presigned S3 URL        |

## Project layout

```
instagram_analyzer_app/
├── main.py                   FastAPI app (routes, auth, upload)
├── celery_app.py             Celery instance + beat schedule
├── tasks.py                  Celery tasks: run_job → parallel ocr_batch (chord) → finalize_job;
│                             reap_stale_jobs, sweep_orphans, poll_minio
├── alembic.ini, alembic/     Schema migrations (0001–0003)
├── processing/
│   ├── config.py             Settings (pydantic-settings) — the table above
│   ├── logger.py             Rotating log file + stderr, job_id binding
│   ├── errors.py             APIError + handler
│   ├── filetypes.py          Allowed extensions
│   ├── naming.py             Job id generation
│   ├── upload_guards.py      Content sniff + ZIP limits for the upload route
│   ├── frame_extractor.py    ffmpeg + OpenCV frame sampling (videos, images, ZIPs)
│   ├── frame_classifier.py   TF SavedModel classifier
│   ├── dedup.py              Perceptual-hash dedup (FRAME_SELECTION=phash)
│   ├── frame_selection/      Scroll-aware selection: structs, align, pages, cover, verify, select
│   ├── metric_coverage/      Tesseract metric coverage: text, labels, detect, tesseract, select
│   ├── selection_pipeline.py choose_ocr_frames — the one selection sequence production and tools share
│   ├── ocr_processor/        OpenRouter OCR: schema, prompt, client, aggregate, batch
│   ├── db_client/            Postgres pool + in-memory fallback: memory, pool, jobs, users, export
│   ├── ingest.py             MinIO auto-ingest poller helpers
│   └── s3_storage.py         Liara S3 wrapper
├── templates/, static/       Jinja2 + plain CSS UI
├── models/                   Deployed TF SavedModel + model_metadata.json (backups are local only)
└── tests/                    pytest suite (183 tests)
tools/                        submit_batch.py, run_local.py, doctor.py, check_outputs.py (see tools/README.md)
training/                     Classifier training, ground truth, review rounds, relabel history
docs/history/                 Session logs and superseded documents
```

## Migrations

Schema is owned by Alembic. The compose file runs `alembic upgrade head` in a one-shot
`migrate` service before the app and worker start (it needs `SECRET_KEY` too, since it loads
`Settings`). To create new migrations:

```bash
docker compose run --rm migrate alembic revision -m "your change"
```

## Known issues & next steps

Ordered by impact on results. Items 1–4 change what the model sees or how consensus votes,
so they change outputs; they were deliberately **not** applied in the 2026-09-30 release so that
the shipped code is exactly what produced the reviewed `Final_Test2` results.

1. **`non_followers` can come back as 0.0** (seen on several Story test videos). The Interactions donut
   has its own followers / non-followers split; the model reports it as the Views split, and with
   more Interactions frames than Views frames the majority vote picks the wrong value
   (`summary_quality.<metric>.disputed` is `true` when this happens). Fix: a prompt line excluding
   the Interactions split, or a consensus rule in `ocr_processor/aggregate.py` preferring reads
   from frames that show the Views donut. Evidence: `docs/history/HANDOFF-2026-09-29.md`.
2. **Seven metric labels have no Persian line in the prompt** (تعاملات, پاسخ‌ها, اشتراک‌گذاری‌ها,
   داستان بعدی, فعالیت نمایه, دنبال کردن, دنبال‌کنندگان / غیردنبال‌کنندگان). Tesseract coverage
   knows them (`metric_coverage/labels.py`) and the model usually reads them; adding them to
   `ocr_processor/prompt.py` is a prompt change and needs a re-run to check.
3. **Provider variance.** OpenRouter routes one model id across providers. Most answer per
   frame; Phala and Reka were seen returning one merged dict for a whole batch (with fabricated
   keys) or answering 1 of 6 frames. The provider is logged on every answer
   (`OpenRouter answered … provider=…`); unanswered frames fail the job. Pinning providers is a
   request-body change (`provider` field) and was not done.
4. **Mid-load frames.** Screens where one tile still shows a placeholder `0` while the rest has
   rendered score 0.999 "good" (`55064813`, frame 81: "Views 0"). Consensus outvotes them, but
   they cause `disputed` flags; the fix is more labelled examples of that state and a retrain
   (`training/relabel/`), not a threshold change.
5. **Network route.** OpenRouter's edge answers `403 Access denied by security policy` to some
   source addresses; the worker's pooled connection now drops on a 403 and reconnects. See
   `RUNBOOK.md`; `tools/submit_batch.py --retry-403` handles the Docker case.
6. **UI error surfaces.** `APIError` renders JSON even on HTML routes; `/export` with zero jobs
   answers 500; a database outage reads as "Job not found" / "Invalid credentials". `/health`
   checks nothing but the process.
7. **No CI.** Tests run by hand (`.venv-train` on the host, or inside the image).
8. **Default credentials** `admin` / `admin123` unless `ADMIN_PASSWORD` is set; Postgres
   `analyzer` / `analyzer` is hard-coded in `docker-compose.yml`.
9. Smaller: log level is not configurable and gunicorn workers + Celery processes share one
   `RotatingFileHandler` (rotation can interleave lines); `frame_extractor.process_campaign_zip`
   reads `OUTPUT_BASE` from the environment outside `Settings`; templates say v1.0.0 while the API
   says 2.0.0; `status.html` references an undefined `status-card` class; the Excel export truncates
   nothing but the status page shows percentages as integers (`| int`).

## Unused code (marked, kept)

Reachable only through re-exports; nothing in the app, tests or tools calls them. Left in place
so the public surface does not change; each carries a note in its docstring.

- `processing/ocr_processor/batch.py`: `process_frames`, `extract_metrics_from_paths` (the pre-Celery sequential path; also the only user of `OCR_DELAY_SECONDS`)
- `processing/s3_storage.py`: `upload_file`, `delete_job_files`
- `processing/db_client/jobs.py`: `get_all_jobs(offset=…)` — no caller passes an offset
- `processing/db_client/export.py`: `export_to_excel(output_path=…)` — no caller passes a path
- `Settings.port` — only `python main.py` reads it
