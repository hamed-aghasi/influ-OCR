# tools/ — command-line tools for operators

All run from the repo root with the training environment
(`.venv-train/bin/python`, see `training/README.md`), or inside the worker image where noted.
None of them is imported by the app; the app never depends on this directory (it is outside
the Docker build context).

| Tool | What it does | Talks to OpenRouter? |
|---|---|---|
| `submit_batch.py` | submits a folder of videos through the running web app, one job at a time, and collects each job's full metrics document + provider/token line | via the worker (billed) |
| `run_local.py` | the same pipeline in-process, no web app / Postgres / Celery; one JSON per video | yes (billed) |
| `doctor.py` | read-only self-check: keys, ffmpeg, Tesseract, classifier, DB, Redis, the route to OpenRouter, the model in the catalogue | one catalogue GET (free) |
| `check_outputs.py` | offline proof that the code still produces the reviewed outputs | never |
| `replay_in_docker.sh` | runs `check_outputs.py replay` inside an image | never |

## submit_batch.py

```bash
ANALYZER_PASSWORD=… .venv-train/bin/python tools/submit_batch.py \
    --input videos/ --out results/ --campaign-name "Spring 2026" --company acme \
    --worker-container ocr-worker-1 --retry-403 2
```

Writes `results/<stem>/<stem>.json` (the job's document as `/api/job/<id>` returns it, plus a
`_job` block), `results/<stem>/worker.log` (with `--worker-container`) and `results/_run_report.json`.
Never overwrites; skips videos that already have a result. `--trust-env` re-enables proxy
environment variables (off by default because VPN clients' local proxies answer 502 for
`localhost`). `--retry-403` needs `--worker-container`: it waits for the OpenRouter route to
come back, restarts the worker if its pooled connection is stale, and resubmits.

## run_local.py

```bash
OPENROUTER_API_KEY=… .venv-train/bin/python tools/run_local.py --input videos/ --out results/ --only <stem>
```

Same stages and same document as the app (`tasks.run_job → ocr_batch → finalize_job`), plus a
`_run` block listing the frames sent. Needs ffmpeg, Tesseract (`fas`, `eng`) and TensorFlow on
the host; the host's Tesseract version can make frame selection differ slightly from the image's
(5.5.0). Stops after two consecutive failures.

## doctor.py

```bash
.venv-train/bin/python tools/doctor.py                    # host
docker compose run --rm -v ./tools:/tools worker python /tools/doctor.py    # inside the worker image
```

Exit code 1 if anything fails. `--offline` skips the two OpenRouter checks. A 403 on the route
check means this address is refused by OpenRouter's edge (`RUNBOOK.md` §2).

## check_outputs.py — output-identity harness

Every check writes a file into `--out`; `compare A B` diffs two such directories. Empty diff =
identical. Checks that need data outside the repo record `{"skipped": …}` instead of failing.

```bash
.venv-train/bin/python tools/check_outputs.py all --out verify/before      # before a change
# ... change code ...
.venv-train/bin/python tools/check_outputs.py all --out verify/after
.venv-train/bin/python tools/check_outputs.py compare verify/before verify/after
```

`all` = `static` (prompt and schema hashes, settings, export headers, package exports, Celery
tasks, tracked-file digests, logger names, log literals) + `wire` (the exact OpenRouter request
body for three synthetic frames, captured with a stubbed session — no network) + `aggregate`
(re-runs consensus over every stored `all_frames_data` under `Final_Test*_output/` and
`docs/eval-2026-08-22/` and checks it equals the stored summary) + `export-offline` (Excel export
from the stored documents through the in-memory store, normalised cell by cell) + `docs-links`.

Two more, run separately:

- `replay` — extraction → classifier → selection for every video in a folder, compared with
  `tools/baselines/final_test2_frames_sent.json` (the frames the reviewed 2026-09-29 run sent).
  Run it **inside the image** so the image's Tesseract decides:
  `tools/replay_in_docker.sh influ-ocr:latest Final_Test2 verify/replay` (add `--mount-tree` to
  run the working tree's code in that image). ~1 min per video, no network.
- `export-live` — `GET /export`, `/jobs`, `/status/<id>`, `/api/job/<id>` from a running stack:
  `ANALYZER_PASSWORD=… tools/check_outputs.py export-live --out verify/live`.

`expect` rebuilds the frames-sent baseline from `Final_Test2_output/` (only when a new reviewed
run replaces it). `tools/baselines/fingerprints.json` records the prompt/schema/model/wire
fingerprints of the released code.
