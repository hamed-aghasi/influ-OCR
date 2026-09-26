# HANDOFF — session state for the next context

## 2026-09-26 — behaviour-preserving modularisation (branch `feat/scroll-coverage-selection`)

- **Layout changed, behaviour did not.** The four 500–650-line modules are packages now:
  `processing/frame_selection/` (structs, align, pages, cover, verify, select),
  `processing/metric_coverage/` (text, labels, detect, tesseract, select),
  `processing/ocr_processor/` (schema, prompt, client, aggregate, batch),
  `processing/db_client/` (memory, pool, jobs, users, export). Each package `__init__`
  re-exports the full surface, so every previous import path still works. The upload sniff and
  ZIP guards moved from `main.py` to `processing/upload_guards.py`. The six live training scripts
  share `training/_bootstrap.py`. Function bodies were copied by exact line range, not retyped.
- **Proof, not assertion.** A scratch checker captured fingerprints at the pre-refactor commit
  `1066e89` and after every commit: `EXTRACTION_PROMPT` and `RESPONSE_FORMAT` hashes (4,150
  bytes), the exact OpenRouter request body for a 3-frame batch, detector outputs / coverage
  selections / scroll selections / final frame sets on 17 replayed videos (ground truth,
  test-frame, Final_Test), re-aggregation of every stored result, the lazy `processing.__all__`
  map, and Celery task registration — **all byte-identical**. Ground truth 47 frames / 0 lost
  before and after. 172 tests pass on the host and inside the rebuilt image.
- **Test edits: 5 lines**, all patch-target aliases (`test_ocr_transport.py` ×4 → `ocr_processor.client`,
  `test_metric_coverage.py` ×1 → `metric_coverage.select`). No assertions changed.
- **Postgres path smoked live** (it has no test coverage): alembic 0001→0003 and every SQL branch
  against a throwaway `postgres:16` with this tree — all pass.
- **Docker:** previous image kept as `influ-ocr:pre-refactor`; rebuilt image verified
  (model digest `f9516a57…`, 172 tests). Gotcha found on the way: `.dockerignore`'s `.DS_Store`
  rule only matches the context root, so a Finder file inside `models/frame_classifier_savedmodel/`
  was copied into the image and changed `verify_model`'s digest. Deleted; consider `**/.DS_Store`.
- **Deliberately left:** the frozen review recipes (`test-frame-output*/_generate_review.py`) and
  the relabel `apply.py` scripts; the `db_client` in-memory branches (not behaviourally parallel to
  SQL, so not collapsible); dead code `s3_storage.upload_file` / `delete_job_files` and the
  `ocr_processor.process_frames` / `extract_metrics_from_paths` chain (only reachable via
  `processing._EXPORTS`) — a separate decision. `PIPELINE.md` references are now `file:name`.

## 2026-09-22 — round 03, the player-UI bug, and a verified Docker deploy

- **Deployed model: run `2026-09-22-round03`** (sha256 `f9516a57e609f2da…`), threshold 0.30.
  Held-out recall **1.000**, precision 0.965, val accuracy 0.991. Previous model backed up to
  `models/backup_20260922T131815Z/`, verified byte-identical to what it replaced before the swap.
- **Two pipeline fixes**, each measured against the ground truth on its own:
  - the `unrecognised` net now requires the page title or a panel section header, so the
    Reel/Story player is no longer treated as an Insights screen (49 → 47 frames, 0 lost);
  - scroll selection's tie-break takes the classifier score before stillness, so a placeholder
    panel no longer beats the frame where the numbers have rendered. Costs 3 frames on
    rounds 01-02 (152 → 155) and fixes the round-03 drops.
- **Docker verified, not assumed.** Image rebuilt; the model inside it has the *same* sha256 as
  the host, loads through the app's own path, 172 tests pass in the image, and the ground truth
  inside the container is **47 frames / 0 lost — identical to the host**. Earlier rounds differed
  here (71 vs 68), so this is new. Previous image kept as `influ-ocr:pre-round03`.
  `training/verify_model.py` is the check; run it on both sides of any deploy.
- **Feedback corpus**: rounds 01-02 are 67 checks (65 pass), round 03 adds **51 more across 49
  videos**. `eval_feedback.py` now scores 73 videos and takes hours — budget for it.
- **Open**: (1) `0dabe58a` frame 051 was reported good but is the Stories archive with a spinner;
  351 labelled good instead, awaiting the reviewer's word. (2) Tesseract cannot read the Persian
  gender donut, so coverage cannot target gender on that layout. (3) The two remaining rounds-01/02
  failures are `78a59e11` 462 and `a54e4eda` 561, both dark-mode fade-ins.

## 2026-09-20 — Docker rebuilt at round-01c; PIPELINE.md written

- **Image rebuilt** at commit `04e4254`. The previous image is kept as
  `influ-ocr:pre-round-01c`. Model is byte-identical in the image
  (`d4a4e0494b7efc75703f8c43c73fed4b`, run `2026-09-19b-sources`).
- **Container vs local, four checks.** Ground truth: **60 frames, 0 metrics
  lost — identical to local.** `run_job` on both `test-frame/` videos: 3 and 4
  frames on both sides. Round-01 feedback: container **25 frames, 19/21**;
  local 24 frames, 21/21.
- **The two failures are cost, not loss.** Debian Tesseract 5.5.0 reads
  `1f2bdcac` 183 as `115 0` where 5.5.3 reads `11s 0` (three digits ⇒ kept as
  "unrecognised"), and hallucinates the Persian word `تال` on `bf96a4ca` 033's
  blurred status bar (number + 3-letter word ⇒ kept). Both frames were marked
  "not needed" by the reviewer. **Metrics delivered are identical on both sides
  for all eight videos.**
- **Evaluate against the container, not the Mac**, when numbers matter.
- **`PIPELINE.md`** is new: every stage, its settings, its failure modes, and a
  scale-readiness section. Its §8 lists the risks for a thousands-of-videos run;
  the top three are a corrupt video recorded as success (0 frames, no error),
  frames never swept after a hard kill, and the reaper failing jobs that are
  merely queued. **None of these were fixed** — the round-01c state was verified
  and documented, not changed.
- Tests: 145 pass.

## 2026-09-20 (later) — the top three §8 risks fixed

- **Unreadable video is now an error** (`VideoUnreadable` in `frame_extractor.py`), not
  0 frames reported as success. A ZIP member that fails is recorded as an error and counted
  into `videos_failed`; the job then fails as incomplete.
- **`sweep_orphans`**, hourly on beat: deletes job directories and uploads untouched for 6h
  whose job is not queued/processing. Covers the case `finalize_job`'s cleanup misses
  (hard time limit, OOM, lost chord message). `KEEP_JOB_FRAMES` disables it.
- **The reaper keys off `jobs.last_progress_at`** (**migration 0003** — run
  `alembic upgrade head`). Processing jobs fail after `JOB_STALE_SECONDS` without progress;
  queued jobs only after `QUEUE_STALE_SECONDS` (24h), so a backlog is no longer failed.
- Tests **165** pass, locally and in the container. Migration verified against a throwaway
  Postgres (upgrade, downgrade, re-upgrade; reaper scenarios). `run_job` on the two test
  videos sends the same frames as before. `PIPELINE.md` §8.1 has the detail; its line
  numbers were re-verified after the changes.
- **Still open from §8**, highest first: pin Tesseract/ffmpeg versions; fail loudly when
  Tesseract is missing instead of silently skipping coverage; give the MinIO poller a
  per-tick budget and the web route's ZIP validation; act on `create_job` returning False.

## 2026-09-19 (later) — human feedback loop, round 01

- **Rounds:** `training/review_videos.py` produces rounds in `review-rounds/`; you fill in `feedback.json`,
  and `training/eval_feedback.py` replays them as checks.
- **Round 01** (6 unseen videos): the feedback exposed five causes, all fixed and tested. See
  `review-rounds/round-01c/README.md`. Round-01 frames sent went from 39 to 24; feedback checks pass 21/21.
- **Schema:** Top sources of views is now metrics `source_*`.
- **Classifier:** retrained as run `2026-09-19b-sources` (131 frames relabelled, plus the round-01 verdicts),
  deployed at 0.30. The previous model is in `models/backup_20260919T162324Z/`.
- **Ground truth** (Top sources added for 5 Reels): 60 frames sent, **0 metrics lost**; the floor is 30.
- **Docker:** the image has **not** been rebuilt since; rebuild and re-run the container check before relying on it.

## 2026-09-19 — frame selection overhaul (branch `feat/scroll-coverage-selection`, local only, not pushed)

**What changed:** the OCR input per video went from a median of ~27 frames to ~6, with no ground-truth
metric lost:
- `processing/frame_selection.py`: scroll-aware selection (default; `FRAME_SELECTION=phash` restores the old dedup);
- `processing/metric_coverage.py`: Tesseract `fas+eng` metric-coverage pass (`COVERAGE_SELECTION`);
- classifier retrained after narrowing `good` to "a schema metric is readable" (`training/relabel/`), deployed as
  run `2026-09-19-relabel` at threshold **0.30** (the previous model is in `models/backup_20260919T100558Z/`,
  which ran at 0.95);
- new schema fields `gender_men` / `gender_women` (Audience > Gender %), in the prompt, Excel and coverage;
- the Dockerfile installs `tesseract-ocr` + `eng` + `fas`; `requirements.txt` adds `pytesseract`.

**Evidence:** `training/eval_selection.py` against `training/ground_truth/` (10 videos, 21 metrics). The truth
files were labelled by Claude, not a human (`labelled_by`). A human spot-check is still the most useful
next step.

**Known limits:**
- a gender split can only be captured if it is on screen. Reel Audience defaults to the Age view, so
  recorders must tap Gender;
- gender-only screens have never been seen by the classifier;
- `c70ca2f3` is the only unseen test video.

**Docker check (2026-09-19):** image `influ-ocr:latest` (Debian: Python 3.11, TF 2.21, Tesseract 5.5.0).
- Inside the worker, `eval_selection.py --strategies scroll,coverage` (repo mounted read-only) sends **71**
  frames with **0 metrics lost**; locally it is 68.
- The classifier and scroll selection are identical; only coverage differs.
- The Tesseract model files are byte-identical, but Tesseract 5.5.0 reads some text worse than 5.5.3
  (e.g. "Follows A" for "Follows 0"), so coverage keeps a few more frames. It never drops needed ones.
- The real `tasks.run_job` in the container sends c70ca2f3 069+111 (same as local) and siah tulani 102+162
  (local: 102+168; 162 and 168 are the same Summary screen). The JPEG bytes differ because ffmpeg is 7.1
  vs 8.1.
- Evaluate against the container, not the Mac, when numbers matter.

**Review outputs:** `test-frame-output/` (v1 baseline), `-v2/` (regenerated in place, see its README), `-v3/`
(current). All are local only; their generators and READMEs are tracked.

---

## 2026-08-22 (earlier state, unchanged)

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
- Tests: **102 passing** (verified 2026-08-31, branch fix/ocr-call-hardening) —
  `cd instagram_analyzer_app && python -m pytest tests/`
  Runs without TensorFlow installed since `processing/__init__.py` became lazy;
  needs fastapi, opencv, imagehash, bcrypt, aiofiles, itsdangerous, celery, psycopg2.
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
- Job frames are now deleted when a job settles (2026-08-31, `tasks._cleanup_job_dir`).
  The eval used to source ground truth from /tmp/processing/<job_id>/frames/ —
  to do that again, set `KEEP_JOB_FRAMES=true` and remember to unset it.

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
