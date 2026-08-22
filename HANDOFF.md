# HANDOFF — session state for the next context

STATUS (2026-08-22): ✅ pipeline live and verified end-to-end on real campaign
videos; model = qwen/qwen3.8-max; all work committed through `d6fe3f5`.
Read CLAUDE.md first for the project map; this file is session state + open threads.

## What works right now (all verified live today)

- `docker compose up --build -d` from repo root brings up postgres, redis,
  migrate (alembic 0001+0002), app on **http://localhost:8010** (NOT 8000 —
  host 8000/5432/6379 are taken by the `mrn-*` stack; DB/redis are
  internal-network only now), worker (concurrency 2), beat (reaper + minio poll).
- `.env` at repo root exists with a **valid** OpenRouter key (ends `02408b`;
  verified against /api/v1/auth/key) and admin credentials for the web UI.
- Full pipeline proven on 10 real campaign videos (CDN sample set, manifest in
  docs/eval-2026-08-22/sample-videos-manifest.tsv): upload → queued →
  chord fan-out (parallel OCR batches) → ffmpeg 720p → frame extract →
  TF classify → phash dedup (~70-90% reduction) → structured-output OCR →
  consensus aggregation → honest terminal status. 9 clean, 1 honest reject.
- Tests: **35 passing** — `cd instagram_analyzer_app && python -m pytest tests/`
  (needs venv with requirements minus TF; opencv-headless suffices).

## Today's commits (all pushed to github.com/hamed-aghasi/influ-OCR main)

- `13d3bdd` structured outputs, batch 50→12, job tracking (queued status,
  celery_task_id, progress, reaper via beat), chord fan-out, UI fixes
- `6cab3bf` MinIO auto-ingest poller (incoming/→ingesting/→processed|failed),
  compose ports change
- `6595da3` consensus aggregation (replaces max()), plausibility validator,
  label-mapped prompt, zero-Insights jobs fail honestly
- `d6fe3f5` default model → qwen/qwen3.8-max, ocr_request_timeout 120s

## Model decision (evidence in docs/eval-2026-08-22/)

Benchmarked on 50 human-verified frames (truth.json = adjudicated ground truth):
qwen3.8-max 99.3% value-acc / 70.4% recall (~96% on returned frames) at
~$0.12/video; gemini-3.7-flash 100%/57.6% at ~$0.036/video (misses icon-strip
metrics, prone to digit-repetition hallucination — reproduced twice);
qwen3-vl-8b 41.9% recall; qwen3-vl-235b returned nothing schema-valid.
Fallback anytime: `OPENROUTER_MODEL=google/gemini-3.7-flash` env var.
Known qwen3.8-max quirk: occasionally returns empty content for a batch —
covered by retry + "N/M batches failed" honest job failure.

## Gotchas learned the hard way

- **Uploads >~1MB through localhost:8010 hang intermittently** (Docker Desktop
  proxy). Reliable path: `docker cp` file into app container, then POST to
  localhost:8000 from inside (python requests; session login first). See the
  upload loop pattern in git history / docs/eval-2026-08-22/bench.py.
- The user's IDE .env edits repeatedly failed to reach disk — always verify
  with grep before concluding a key is set.
- WebSearch tool errors in this project's sessions ("long context beta") —
  use OpenRouter/models API via curl, or a subagent.
- Frames persist in the `processing` volume (cleanup never implemented) —
  that's how the eval got its ground-truth images: /tmp/processing/<job_id>/frames/.

## Known data issue

Job `trend_rcmpbsseni_20260822143046_b850` + `..._150938_e248` (dailymetanat):
stored summary says follows=3154; the frame actually shows **31** (verified
visually). A re-run under qwen would fix it. Sample videos are re-downloadable
via the manifest CDN URLs; also still in the app container at /tmp/samples/.

## Open threads, in priority order

1. **Re-run the 10 samples under qwen3.8-max** (~$1.20) — confirms recall gain
   end-to-end (reach/accounts_reached should now appear), fixes dailymetanat.
2. **Production integration should be REPORT-DRIVEN, not folder-watching.**
   Discovery from ~/Desktop/api-gateway-schema.md (schema of hamgit.ir/trend/api-gateway):
   the platform stores influencer evidence in `campaign_item_reports`
   (type start_screenshot/end_screenshot/video, FK file_manager_id → `files`
   with disk='s3', path like `campaigns/shots/instagram/video/<uuid>.mp4`,
   served at cdn.trendmedia.ai) and metrics belong in `campaign_item_insights`
   keyed by campaign_item_id with source in ('telegram','instagram','audit') —
   columns map ~1:1 to our extracted metrics. Right shape: poll/subscribe to
   new reports → fetch object by files.path → OCR → write insights row
   (probably source='audit'). The current incoming/-prefix poller (still
   useful standalone) awaits MINIO_* creds the user never provided.
3. **hamgit repos are private** — https://hamgit.ir/trend/api-gateway and
   /trend/infra. Clone attempts failed (auth); user said to use their token but
   credential probing is permission-blocked. Ask user to `! git clone` them or
   paste a read-only token. Needed to confirm files.path generation + whether
   'audit' is the right insights source + MinIO deploy details (infra repo).
4. Remaining review items from the 2026-08-22 code review (never done):
   video magic-byte sniff is a no-op for .mkv/.avi/.mov (main.py `_sniff_ok`);
   no CSRF token; no login rate-limit; processing-dir cleanup (disk leak —
   but see gotcha above, eval depends on frames persisting); no CI workflow;
   TF→ONNX classifier shrink; sweep for objects stuck in ingesting/ when the
   reaper kills a job.
5. kala.check sample video: completed with empty metrics (6 frames passed
   classifier, no numbers read) — eyeball the video to see what those frames are.

## Eval assets (docs/eval-2026-08-22/, committed)

truth.json (adjudicated per-frame ground truth for 50 frames), sample.json
(which frames + what the prod model returned), bench_*.json (per-model raw
results), bench.py + score.py (rerunnable harness — point MODELS at any
OpenRouter slug to benchmark a new model against the same truth for pennies).
The 50 frame JPEGs themselves are NOT in the repo — regenerate by re-running
sample videos and pulling from the processing volume per sample.json's
job/frame names.
