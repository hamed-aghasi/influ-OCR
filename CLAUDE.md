# influ-OCR — Instagram Campaign Analyzer

> Session state + open threads: see `HANDOFF.md` (latest section dated 2026-09-20).
> How the pipeline works stage by stage, its settings and its failure modes:
> **`PIPELINE.md`** — read it before running a large batch.

FastAPI + Celery app that extracts Instagram Insights metrics (views, reach,
likes, shares, gender split, …) from screen recordings and screenshots, English
and Persian. Pipeline, per video:

1. frame extraction: ffmpeg 720p, then every 3rd frame (`processing/frame_extractor.py`)
2. MobileNetV2 good/bad classifier, threshold 0.30 (`processing/frame_classifier.py`, `models/`)
3. scroll-aware selection: fewest frames covering each scrolled page (`processing/frame_selection/` — `align`, `pages`, `cover`, `verify`, `select`)
4. metric coverage: Tesseract `fas+eng` reads which metrics each frame shows, and a set cover keeps the fewest (`processing/metric_coverage/` — `text`, `labels`, `detect`, `tesseract`, `select`); then a pass drops frames whose content is readable in another sent frame. Steps 3 and 4 run through one function, `processing/selection_pipeline.choose_ocr_frames`, which production and every tool call
5. OCR via an OpenRouter vision model (default `qwen/qwen3.8-max`, `processing/ocr_processor/` — `schema`, `prompt`, `client`, `aggregate`, `batch`)
6. consensus aggregation, then Postgres, optional S3, Excel export

Steps 3 and 4 are switched by `FRAME_SELECTION` (`scroll` or `phash`) and
`COVERAGE_SELECTION`. The classifier threshold is tied to the deployed model;
see the comments in `processing/config.py`.

## Running

`docker compose up --build` from the repo root brings up postgres, redis,
alembic migrate, the app on **:8010**, the celery worker and beat. It needs
`SECRET_KEY` (and `OPENROUTER_API_KEY` for OCR) in a root `.env` (see
`instagram_analyzer_app/.env.example`). Beat's reaper fails jobs stuck
queued/processing.

Tests: `cd instagram_analyzer_app && python -m pytest tests/` (no TensorFlow
or tesseract needed for most of them).

## Training and evaluation (`training/`)

- `.venv-train/` (repo root, untracked): Python 3.9 + TF 2.20 environment for everything below.
- `extract_frames.py` → sort into `train/<video>/good|bad` → `train_frame_classifier.py` (runs in `training/runs/`; `--deploy` backs up and swaps into `models/`).
- `eval_selection.py`: frames sent and **metrics lost** per strategy, against the hand-built `training/ground_truth/`. Run it before changing selection, coverage or the threshold.
- `trace_frames.py`: which frames one video would send to OpenRouter (no network).
- `review_videos.py VIDEOS --out review-rounds/<round>`: a human review round (per-video `review.html`, `final/`, and a `feedback.json` to fill in).
- `eval_feedback.py`: replays every `review-rounds/*/*/feedback.json` marked done as regression checks. Run it with `eval_selection.py` before changing selection, coverage or the model.
- `relabel/`: label changes, each with an exact undo. 2026-09-19 narrowed `good` to schema metrics; `2026-09-19b-sources/` added Top sources of views and the round-01 feedback.
- `train/`, `test-frame/`, `training/runs/`, the model backups and the review outputs are local only (see `.gitignore`).

## Docs status

- `PROXY_SUPPORT_PLAN.md`: proposal only, never implemented (see its status header).
- `SESSION-2026-09-12.md`: the dark-mode classifier diagnosis and the start of retraining.
- `test-frame-output*/`: v1/v2/v3 reviews of the two test videos (each README says what it shows).
