# influ-OCR — user guide

How to run the Instagram Insights analyzer, what it needs, what it costs, and where the
results go. For deeper detail: `RUNBOOK.md` (operations), `PIPELINE.md` (how it works),
`README.md` (every setting, known issues).

## What it does

You give it screen recordings (or screenshots) of Instagram Insights screens — Reels, Posts,
Stories, English or Persian. It picks the frames that show numbers, sends them to a vision
model through OpenRouter, and returns one set of metrics per upload: views, viewers
(accounts reached), followers / non-followers split, interactions, likes, comments (`replies`),
shares, navigation (forward / next story / back / exited), profile activity / visits, follows,
link and sticker taps, gender split, and the top sources of views. Results are stored in
Postgres, shown in the web UI, and exported to Excel.

## Requirements

- **Docker** with Docker Compose (Docker Desktop on macOS/Windows, or Docker Engine on Linux).
  Everything the pipeline needs — ffmpeg, Tesseract with English and Persian, TensorFlow, the
  trained classifier — is inside the image. ~2 GB of disk for the image.
- **An OpenRouter account and API key** (`https://openrouter.ai/keys`) with credit on it. This
  is the only paid component.
- **A network route that OpenRouter accepts.** OpenRouter's edge refuses some source
  addresses (seen from Iranian ISPs) with `403 Access denied by security policy`. If that is
  your situation, a VPN must be on *before* starting the containers and stay on while jobs
  run; the worker container makes the calls, and it uses the machine's route. `RUNBOOK.md` §2.
- Optional, only for the command-line tools and for retraining: Python 3.9–3.12 on the host
  with `training/requirements.txt` installed into `.venv-train/` (`training/README.md`).

## Keys and settings

Copy `instagram_analyzer_app/.env.example` to `.env` in the repo root and set:

| Variable | What |
|---|---|
| `SECRET_KEY` | any random string, at least 16 characters; signs the login session. **Required by every `docker compose` command**, even `ps`. |
| `OPENROUTER_API_KEY` | your OpenRouter key (`sk-or-v1-…`). Without it every job fails at the OCR step. |
| `ADMIN_PASSWORD` | the web login password. The default is `admin123` — change it. Username is `admin` (`ADMIN_USERNAME`). |

Everything else has a working default. The ones you might touch: `OPENROUTER_MODEL`
(default `qwen/qwen3.8-27b` — always an explicit OpenRouter catalogue id, never an alias),
`OPENROUTER_REASONING_EFFORT` (`low`), `KEEP_JOB_FRAMES` (`true` keeps each job's extracted
frames on disk for inspection; fills the disk). The full table is in `README.md`.

Never commit `.env`; it is ignored by git. If you prefer not to keep secrets in a file, pass
them on the command line instead: `SECRET_KEY=… OPENROUTER_API_KEY=… docker compose up -d`.

## Setup

```bash
git clone https://github.com/hamed-aghasi/influ-OCR.git
cd influ-OCR
cp instagram_analyzer_app/.env.example .env      # then edit the three keys above
docker compose up --build -d                     # first build takes a few minutes
docker compose ps                                # app and worker "healthy"; beat running
```

Then open **http://localhost:8010** and log in. To check the installation end to end
(keys, ffmpeg, Tesseract, classifier, database, the route to OpenRouter, the model in the
catalogue):

```bash
docker compose run --rm -v ./tools:/tools worker python /tools/doctor.py
```

Stop with `docker compose down` (the database volume persists); update with
`git pull && docker compose build && docker compose up -d`.

## Uploading videos

**Web UI (one at a time).** On the front page fill in campaign date, campaign name, product
and company, choose a file and upload. Accepted: `.mp4 .mov .mkv .avi`, `.jpg .jpeg .png`, or
a `.zip` of screenshots; up to 500 MB. You land on the job's status page, which refreshes
until the job is `completed` or `failed`; **Jobs** lists everything with filters.

A typical 10–20 s recording takes 30–150 s end to end (extraction, classification, frame
selection, then one OpenRouter request of 2–10 frames). Two jobs run in parallel.

**Many videos at once.** From the repo root, with the host Python environment:

```bash
ANALYZER_PASSWORD=<your ADMIN_PASSWORD> .venv-train/bin/python tools/submit_batch.py \
    --input /path/to/videos --out results/ --campaign-name "Spring 2026" --company acme \
    --worker-container ocr-worker-1 --retry-403 2
```

It submits each video through the same upload route, waits for it, and writes
`results/<video>/<video>.json` (the complete result) plus the worker's log lines for that job.
It never overwrites an existing result, and `--retry-403` waits out a dropped network route.

**Without the web app** (no database; one JSON per video straight from the pipeline):
`OPENROUTER_API_KEY=… .venv-train/bin/python tools/run_local.py --input videos/ --out results/`
— needs ffmpeg, Tesseract (`fas`+`eng`) and TensorFlow installed on the host.

## Where the results are saved

| Where | What you get |
|---|---|
| **Status page** `/status/<job_id>` | the metrics summary, frame counts, status, error message |
| **Excel** — the *Export* button, or `GET /export` | one row per job: campaign fields, status, frame counts, then the 28 metric columns (empty cell = not seen in that video) |
| **JSON** — `GET /api/job/<job_id>` (log in first) | the job row plus `metrics_json`: `summary` (the values), `summary_quality` (how many frames read each value, min/max, `disputed` when frames disagreed), `all_frames_data` (every frame's raw reading), and counters |
| **Postgres** (volume `postgres_data`) | table `jobs` + `job_metrics.metrics_json` — the same document |
| `tools/submit_batch.py` / `run_local.py` | `<out>/<video>/<video>.json` per video and `_run_report.json` |
| Liara S3 (optional, when the four `LIARA_*` settings are set) | `<job_id>/instagram_metrics.json` per job |

Extracted frames are deleted when a job finishes (unless `KEEP_JOB_FRAMES=true`). Logs:
`docker logs ocr-worker-1` — each OpenRouter answer is logged with its provider and token
counts.

A `disputed: true` in `summary_quality` means the frames disagreed and the majority value
was taken; check it against the video. The known case is `non_followers` on Story videos
(`README.md`, known issue 1).

A job is marked **failed** — with whatever metrics it did get — when the OpenRouter call
failed after retries, when the model left frames unanswered, when it returned no metric at
all, or when no Insights screen was found in the file. The status page shows which.

## Cost

Measured on the reviewed test batch (19 videos, `qwen/qwen3.8-27b`, reasoning `low`,
2026-09-29): **51,888 prompt tokens + 38,618 completion tokens ≈ $0.17**, i.e. about
**$0.009 per video → ≈ $9 per 1,000 videos** at OpenRouter's prices that day ($0.063 per
million prompt tokens, $4.40 per million completion tokens; completion tokens are 98 % of the
bill). Add a margin: prices on OpenRouter change (this model's completion price went from
$3.00 to $4.40 per million within three days), long videos send more frames, and a failed
job that is resubmitted pays twice. Budget **$10–15 per 1,000 videos** and check the
current price at `https://openrouter.ai/models` before a large batch. HTTP 403 refusals
cost nothing; only answered requests are billed.

## If something goes wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| every job fails with `OCR batches failed`, worker log shows `API error 403` | route to OpenRouter blocked | turn the VPN on; `docker restart ocr-worker-1` on an older image; see `RUNBOOK.md` §2 |
| `API error 401` | wrong `OPENROUTER_API_KEY` | fix `.env`, `docker compose up -d` |
| `docker compose` says `SECRET_KEY` is missing | no `.env` / not exported | create `.env` or pass it on the command line |
| "Invalid credentials" with the right password | database not reachable at startup | `docker compose ps`, then `docker compose restart app` |
| job stuck in `processing` for hours | worker died mid-job | it is auto-failed after 2 h; `docker compose restart worker` |
| `OCR returned nothing for N of M frame(s)` | the provider answered only some frames | resubmit the video; another provider usually serves it |

More in `RUNBOOK.md` §5 and `PIPELINE.md` §8.

## Verifying an installation still produces the reviewed results

`tools/check_outputs.py all --out verify/x` (offline) and `tools/replay_in_docker.sh
influ-ocr:latest <videos> verify/replay` compare the running code against the fingerprints
and frame selections recorded in `tools/baselines/`; see `tools/README.md`.
