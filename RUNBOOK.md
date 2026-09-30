# RUNBOOK — operating influ-OCR

For the people who run it. What the pipeline does is in `README.md` and `PIPELINE.md`.

## 1. Install

Docker is the supported way to run the app. The host needs nothing but Docker and, for the
command-line tools and the training scripts, the Python 3.9 environment described in
`training/README.md` (`.venv-train/` at the repo root; `pip install -r training/requirements.txt`).

```bash
git clone <repo> && cd ocr
cp instagram_analyzer_app/.env.example .env
# edit .env: SECRET_KEY (>= 16 random chars), OPENROUTER_API_KEY, ADMIN_PASSWORD
docker compose up --build -d
docker compose ps            # app healthy on http://localhost:8010, worker healthy, beat running
```

Notes that bite:

- **Every `docker compose` command needs `SECRET_KEY`** (in `.env` or the environment), even
  `ps` or `logs` — the compose file refuses to interpolate without it. If you would rather not
  keep secrets in a file, pass them on the command line each time:
  `SECRET_KEY=… OPENROUTER_API_KEY=… docker compose up -d`.
- The app listens on `127.0.0.1:8010` only. Put a reverse proxy in front for anything else.
- The classifier is inside the image (`instagram_analyzer_app/models/`, ~20 MB); nothing is
  downloaded at runtime except OpenRouter answers.
- Rebuild after any code change: `docker compose build && docker compose up -d`. Tag the old
  image first (`docker tag influ-ocr:latest influ-ocr:before-<date>`) so you can roll back with
  `docker tag influ-ocr:before-<date> influ-ocr:latest && docker compose up -d`.

## 2. Network: the route to OpenRouter

The **worker container** makes the OpenRouter calls, so its route is what matters, not the
browser's. OpenRouter sits behind Cloudflare and answers
`403 {"success": false, "error": "Access denied by security policy."}` to some source
addresses (seen from an Iranian ISP range). Symptoms: every job fails with
`1/1 OCR batches failed; extracted metrics are incomplete`, and the worker log shows
`API error 403` five times per batch. No tokens are billed for these.

- Docker Desktop shares the host's network path, so a system-level VPN carries the containers
  too. Check from inside the worker, not from the host:
  `docker exec ocr-worker-1 python -c "import requests;print(requests.get('https://openrouter.ai/api/v1/models',timeout=20).status_code)"` → `200`.
- `tools/doctor.py` runs this and every other prerequisite check in one go.
- A connection opened while the route was blocked used to stay blocked (keep-alive); since the
  2026-09-30 fix the worker drops its pooled connection on a 403 and reconnects. If you run an
  older image, `docker restart ocr-worker-1` clears it.
- VPN clients often install a system proxy (e.g. `127.0.0.1:3067`). Python `requests` on the
  host then routes `localhost` through it and gets 502 — the tools default to
  `trust_env=False`; pass `--trust-env` only if you need the proxy.

## 3. Running videos

**One at a time:** log in at `http://localhost:8010`, upload a video (`.mp4/.mov/.mkv/.avi`),
an image, or a ZIP of screenshots. The status page refreshes until the job settles; `/jobs`
lists everything.

**A folder of videos:** `tools/submit_batch.py` submits through the same web route, waits for
each job, and writes one folder per video with the job's full metrics document plus the
provider/token line from the worker log:

```bash
ANALYZER_PASSWORD=… .venv-train/bin/python tools/submit_batch.py --input videos/ --out results/ \
    --campaign-name "Spring" --company acme --worker-container ocr-worker-1
```

It never overwrites an existing result, writes `_run_report.json`, and with
`--worker-container` + `--retry-403` waits out a dropped route and resubmits.

**Without the web app** (no Postgres, no Celery — the same pipeline in-process):
`OPENROUTER_API_KEY=… .venv-train/bin/python tools/run_local.py --input videos/ --out results/`.
Needs ffmpeg, Tesseract with `fas`+`eng` and TensorFlow on the host. Frame selection can differ
slightly from the Docker run when the host Tesseract version differs from the image's (5.5.0).

**Cost:** about $0.01 per video with the default model; `OpenRouter answered … prompt_tokens=…
completion_tokens=…` in the worker log is the per-request usage. Reasoning tokens are part of
completion. See `README.md` for the measured figures.

## 4. Where the results are

| Where | What |
|---|---|
| `/status/<job_id>` | summary metrics, frame counts, status, error message |
| `/api/job/<job_id>` | the job row plus `metrics_json`: the complete document (`summary`, `summary_quality` with `disputed` flags, `all_frames_data` per frame, batch/missing counters) |
| `/export` | Excel: one row per job, 14 job columns + 28 metric columns (empty cell = metric not seen) |
| Postgres `job_metrics.metrics_json` | the same document (JSONB; key order not preserved) |
| worker log (`docker logs ocr-worker-1`, or the `logs` volume `/tmp/logs/instagram_analyzer.log`) | per-stage counts, coverage decisions, provider + token usage, errors |
| `/tmp/processing/<job_id>/` (volume `processing`) | frames and `instagram_metrics.json` **only while the job runs**; deleted when it settles unless `KEEP_JOB_FRAMES=true` |
| Liara S3 (optional) | `<job_id>/instagram_metrics.json` |

Job statuses: `queued` → `processing` → `completed` | `failed`. A job is **failed**, with the
metrics it did get kept, when an OCR batch failed after retries, when frames went unanswered by
the model, when the model returned no metric at all, or when the classifier found no Insights
frames. `error_message` says which.

## 5. Failure modes, short list

Full catalogue: `PIPELINE.md` §8.

| You see | Cause | Do |
|---|---|---|
| every job: `OCR batches failed`, worker log `API error 403` | route to OpenRouter blocked | §2 |
| worker log `API error 401` | bad `OPENROUTER_API_KEY` | fix `.env`, `docker compose up -d` |
| `Frame classifier unavailable` | model dir missing/corrupt in the image | rebuild; `tools/check_outputs.py static` shows the model digests |
| `Coverage skipped` in the log, more frames sent than usual | Tesseract or `fas`/`eng` missing | the image installs them; on the host install `tesseract-ocr-fas` |
| `OCR returned nothing for N of M frame(s)` → job failed | the provider answered only some frames | resubmit; another provider usually serves it (`README.md` known issue 3) |
| `summary_quality.<metric>.disputed = true` | frames disagreed; consensus took the majority | check the value; `non_followers` is the usual one (`README.md` known issue 1) |
| jobs stuck in `processing` for hours | worker died mid-job | beat's reaper fails them after `JOB_STALE_SECONDS` (2 h); `docker compose restart worker` |
| "Invalid credentials" for the right password | Postgres unreachable at start | `docker compose ps`, then restart `app` |
| worker or beat shown `unhealthy` on an old image | inherited HTTP healthcheck | fixed 2026-09-30 (compose per-service checks); harmless |

## 6. Checking the installation

```bash
.venv-train/bin/python tools/doctor.py            # host: tesseract, ffmpeg, model, key, route, model in catalogue
docker compose run --rm -v ./tools:/tools worker python /tools/doctor.py   # the same inside the worker image
```

`tools/check_outputs.py` re-verifies that the code still produces the reviewed outputs
(offline; see `tools/README.md`). Run it before and after any change and `compare` the two.

## 7. Maintenance

- **Retrain the classifier:** `training/README.md`. Deploying a model changes selection; run
  `training/eval_selection.py` and `training/eval_feedback.py` first.
- **Change the OCR model:** set `OPENROUTER_MODEL` to an explicit catalogue id; run one video and
  compare `all_frames_data` against a known-good result before a batch.
- **Disk:** frames are deleted per job; the sweeper removes orphans older than 6 h. Redis has no
  persistent volume (queued jobs are lost on `docker compose down`; the reaper fails them later).
- **Backups:** Postgres volume `postgres_data`; the Excel export is a complete extract of the metrics.
