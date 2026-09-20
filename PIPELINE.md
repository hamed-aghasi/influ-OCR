# The pipeline, end to end

What happens to a video between upload and a row of metrics, stage by stage:
what each stage decides, on what setting, and how it breaks. Written
2026-09-20 against the round-01c state on `feat/scroll-coverage-selection`,
for a run of thousands of videos. Updated the same day with the three fixes in
§8.1; line numbers were re-verified against the code after them.

Every claim carries a `file:line` so it can be re-checked rather than trusted.
Line numbers are accurate as of that commit and will drift; the function names
will not.

> **Read §8 before the large run.** It is the collected failure modes. §10 says
> what to do about them.

---

## 0. The map

```
upload (web) ──┐
               ├─→ jobs row ─→ run_job ─→ frame extraction ─→ classifier ─→ scroll selection ─→ metric coverage ─→ OCR ─→ consensus ─→ Postgres / S3 / Excel
MinIO poller ──┘                          ffmpeg + OpenCV      MobileNetV2     block alignment     Tesseract        OpenRouter   per metric
                                          every 3rd frame      keep > 0.30     fewest per page     fewest covering  1 request    mode, else median_low
```

Per video, measured on the round-01 set: **50–225 frames extracted →
47–205 pass the classifier → 15–81 after scroll selection → 2–7 sent to OCR**,
one OpenRouter request.

The three switches that change the shape of this, all in
`processing/config.py`:

| setting | default | effect |
|---|---|---|
| `classifier_threshold` | `0.30` (`config.py:77`) | tied to the deployed model; see §3 |
| `frame_selection` | `"scroll"` (`config.py:113`) | `"phash"` restores the old whole-frame dedup |
| `coverage_selection` | `True` (`config.py:120`) | off ⇒ every scroll-selected frame goes to OCR |

Stages 4 and 5 run through one function, `selection_pipeline.choose_ocr_frames`
(`selection_pipeline.py:31`), which production (`tasks.py:129`) and every
evaluation tool call. They cannot diverge.

---

## 1. Ingest

### Web upload — `main.handle_upload` (`main.py:136`)

In order: session auth (`main.py:144`) → extension allow-list
(`.mp4 .avi .mov .mkv .jpg .jpeg .png .zip`, `filetypes.py:6`) → date parse →
**magic-byte sniff of the first 64 bytes** (`main.py:155`) → job id → streamed
write in 1 MiB chunks with a running size check against `max_upload_bytes`
(500 MB, `config.py:124`) → ZIP validation → `jobs` row → `celery.send_task`.

ZIP validation (`main._validate_zip:223`) reads the central directory and
rejects more than `max_zip_entries` (2 000) or more than
`max_zip_uncompressed_bytes` (2 GiB).

Job id (`naming.py:12`) is `slug(company)_slug(campaign)_YYYYMMDDHHMMSS_xxxx`
with 4 hex characters of randomness.

### MinIO auto-ingest — `tasks.poll_minio` (`tasks.py:375`)

Runs on beat every `minio_poll_interval_seconds` (60 s, `config.py:152`) when
all four MinIO settings are present. It lists `incoming/`, requires the key to
be exactly `company/campaign/product/file.ext` (`ingest.parse_object_key:56`),
claims each object by copying it to `ingesting/` and deleting the original,
downloads it, creates the job row and enqueues `run_job`.

**The poller is not the web route.** It performs **no magic-byte sniff and no
ZIP validation at all** — `max_zip_entries` and `max_zip_uncompressed_bytes`
are enforced only in `main._validate_zip`. It also **discards the return value
of `create_job`** (`tasks.py:418`), and `db_client.create_job` returns `False`
on any failure rather than raising (`db_client.py:120`).

### What breaks

- **`.mkv`, `.avi` and `.mov` skip content sniffing entirely** (`main.py:60`:
  `or ext in {".mkv", ".avi", ".mov"}`). Any blob with that extension is
  accepted and only discovered as garbage inside the worker.
- **Non-ASCII names slug to the empty string** (`naming.py:9` strips everything
  outside `[a-z0-9]`). Persian company and campaign names — likely, given the
  `fas+eng` OCR — produce ids like `__20260920131500_a1b2`, so all collision
  resistance rests on the timestamp plus 4 hex characters.
- **Job-id collision on the poller path is silent and destructive.** The insert
  fails, the return is ignored, and both jobs then use the same
  `processing_dir/{job_id}` — where extraction deletes existing frames on
  startup (`frame_extractor.py:106`).
- **A failed `create_job` still pays for the whole job.** Every later status and
  metric write no-ops against a row that does not exist, `/status/{job_id}`
  returns 404, and the OCR spend produces nothing persisted.
- **A ZIP containing no video passes every guard** and fails with the wrong
  message, "No Instagram Insights screens detected in this upload"
  (`tasks.py:254`).
- `handle_upload` is `async`, but `_validate_zip`, the DB calls and
  `celery.send_task` are blocking calls made on the event loop.
- The 500 MB limit is enforced *after* streaming; there is no `Content-Length`
  pre-check, and no global cap or free-space check on `upload_dir`.

---

## 2. Frame extraction — `processing/frame_extractor.py`

`extract_frames_from_video` (`:94`), called at `tasks.py:104`.

1. Delete any existing `frame_*.jpg` in the output folder (`:88`).
2. **Transcode to 720p** if `convert_to_720p` (default `True`, `config.py:83`)
   and OpenCV reports a height above 720 (`:84`):
   `ffmpeg -y -i IN -vf scale=-2:720 -c:v libx264 -preset fast -crf 23 -c:a copy OUT`
   (`:70-75`), with `ffmpeg_timeout_seconds` = 600 (`config.py:84`).
3. Decode **every** frame, save **every third** (`frame_interval` = 3,
   `config.py:82`) as `frame_NNNNNN.jpg` at OpenCV's default JPEG quality
   (`:148-158`).
4. Write `extraction_metadata.json` (never read downstream).

Raises `VideoUnreadable` if the file cannot be opened or yields no frames; the
ZIP path catches it per video and records that member as an error.

The ZIP path (`process_campaign_zip:226`) loops videos **serially**, extracting
each to a `TemporaryDirectory` and re-opening the ZIP once per video.

### What breaks

- ~~`cap.isOpened()` is never checked.~~ **Fixed 2026-09-20.** A file OpenCV
  cannot open, or that decodes to nothing, now raises `VideoUnreadable`
  (`:133`, `:169`) instead of returning 0 frames silently. See §8.1.
- **`scale=-2:720` pins the *long* edge for portrait video.** Instagram
  recordings are portrait, so 1080×1920 becomes ~**406×720**; the frames in this
  repo measure 324–334 × 720. That is a ~2.7× downscale of exactly the small
  numeric text everything downstream must read. It is the current operating
  point, not a regression — every measured result was produced at this size —
  but it is the single biggest lever on legibility.
- **Sampling is by frame index, not timestamp** (`:148`). Screen recorders
  routinely produce variable frame rate, so the real time between sampled frames
  varies. Stage 4 assumes consecutive samples are consecutive in time.
- **`FRAME_INTERVAL=0` is a `ZeroDivisionError`.** No numeric setting in
  `config.py` has bounds; only `secret_key` has a validator (`config.py:28`).
- **No cap on frames or duration.** A 10-minute 60 fps video at interval 3 is
  ~12 000 JPEGs, 2–3 GB for one job.
- ffmpeg timeout or non-zero exit is logged and **the original file is used
  instead** (`:84`, `:88`) — up to 10 minutes of CPU spent, then silently
  ignored.
- `cv2.imencode` returning false drops the frame with **no log line at all**
  (`:150`); a write `OSError` logs and continues (`:158`), so a full disk
  produces a partially extracted video rather than a failure.
- ~~One bad video in a ZIP is skipped and the job completes green.~~ **Fixed
  2026-09-20**: it is recorded as `"status": "error"`, counted into
  `videos_failed`, logged, and the job is failed as incomplete. See §8.1.

---

## 3. Classifier — `processing/frame_classifier.py`

MobileNetV2 transfer-learning SavedModel, loaded once per process as a Keras
`TFSMLayer` with `call_endpoint="serving_default"` (`:38`), cached in a module
global behind a `threading.Lock`.

Per frame (`_preprocess:47`): BGR read → if mean grey < `dark_frame_threshold`
(80, `config.py:79`) apply `convertScaleAbs(alpha=1.5, beta=40)` → resize to
224×224, aspect ratio deliberately discarded → `/255.0`. This matches training
exactly (`training/train_frame_classifier.py:105`).

Decision (`:87`): `GOOD if confidence > classifier_threshold else BAD`, with the
threshold at **0.30** (`config.py:77`).

**What `good` means matters more than the number.** The label was narrowed this
month to *"at least one schema metric is readable, with its label and its
value"*, Top sources of views included (`training/relabel/2026-09-19b-sources/`).
The deployed model is run `2026-09-19b-sources`; held-out recall is 1.000 from
0.05 to 0.85, so 0.30 sits in a wide flat region. Rollback copies are in
`models/backup_*`.

### What breaks

- **Inference runs one frame at a time** (`:119`), so 12 000 frames is 12 000
  eager `TFSMLayer` calls. There is no batching parameter.
- **Only the first unclassifiable frame is ever logged** (`:131`), and the
  all-failed guard only fires at 100 % (`:167`). A systematic 40 % decode
  failure is invisible; `failed_frames` is written into the job directory, which
  is then deleted, and never reaches the database.
- **Two Celery prefork children each hold their own TensorFlow runtime and
  model.** There is no `worker_max_tasks_per_child`, so per-call eager overhead
  accumulates for the container's lifetime. With `task_acks_late=True` and no
  `task_reject_on_worker_lost` (`celery_app.py:19`), an OOM kill re-delivers the
  same message to the next child — a poison-message loop.
- The confidence unwrap (`:86`) assumes a single sigmoid output; a two-class
  softmax export would silently read the wrong class. Nothing checks the output
  shape against `models/model_metadata.json`.
- `classifier_threshold` is unbounded: `1.5` classifies everything BAD and
  surfaces only as "No Instagram Insights screens detected".

---

## 4. Scroll-aware selection — `processing/frame_selection.py`

The question this stage answers: *given frames of a page being scrolled, which
smallest subset shows every row at least once?* It replaced perceptual-hash
dedup, which compared whole frames and therefore merged two screens differing
only in one row of digits while splitting one screen photographed at two scroll
positions.

Everything works on frames downscaled to **160 px wide** (`ALIGN_WIDTH`,
`:39`), so an "align row" is about 2 source rows at 720p.

### 4.1 The aligner — `vertical_shift` (`:112`)

Rows are reduced to 20 column-block means (`BLOCK` = 8, `:90`). A row *matches*
when **every** block differs by less than `BLOCK_MATCH` = 20.0 grey levels
(`:91`). Rows flatter than `TEXTURE_STD` = 4.0 (`:93`) carry no evidence.

- **Stillness short-circuit** (`:132`): if at least `STILL_FRAC` = 0.85 (`:97`)
  of textured rows match in place, return 0 — "not scrolled", even if something
  local changed.
- **Exhaustive search** (`:147`) over every integer offset in **±75 % of frame
  height** — 535 candidates at h = 356 — scoring matched textured rows.
- **Acceptance** (`:161`): the winner needs `MIN_EVIDENCE // 2` = 12 matched
  rows *and* `MIN_MATCH_FRAC` = 0.6 of the textured rows in the overlap, and the
  best runner-up more than 3 px away must score below `PEAK_RATIO` = 0.8 of it
  (`:163`). Otherwise `None` — "no offset clearly supported".

Frames are chained into pages by summing these offsets (`_build_pages:197`);
when a link fails, `_link_to_earlier` (`:181`) tries up to `LINK_SAMPLES` = 8
frames of each earlier page, newest first, before starting a new page.

### 4.2 Pinned bands, coverage, verification

`_fixed_bands` (`:222`) compares the lowest and highest frame of a page and
declares rows that barely differ to be fixed UI (header, nav bar), capped at
**45 % of the height each** (`:237`). Default when a page barely scrolled:
10 % top, 8 % bottom (`:229`).

`_cover_page` (`:241`) is a greedy interval cover over page coordinates,
requiring a full `ROW_HEIGHT` = 0.10 (`:42`) of overlap — the assumed tallest
label-plus-number row — and breaking ties by `(stillness, -score, order)`.

`_verify` (`:391`) then checks **every** good frame row by row against the
picks, and adds frames back until no frame has more than `ROW_SLACK` = 2 (`:43`)
uncovered rows. Rows that change between neighbouring frames are excluded from
what must be covered (`_stable_rows:350`).

### What breaks

- **Horizontal scrolling is unmodelled.** A carousel swipe changes every row at
  every offset ⇒ every link fails ⇒ one page per frame, every frame kept, and
  the quadratic re-link path engages.
- **Partial horizontal motion is worse than that.** If a strip moves sideways
  while ≥ 85 % of textured rows hold still, `:134` returns 0, both frames sit at
  the same offset, and the moving rows then match *neither* neighbour, so
  `_stable_rows` drops them from what must be covered. **Nothing is then
  required to show them.** The same reasoning applies whenever `STILL_FRAC`
  denies a genuine scroll of a small region.
- **A screen that exists for exactly one sampled frame can be dropped.** Its
  rows differ from both neighbours, so they are treated as animation rather than
  content. At `frame_interval` 3 that is anything visible for under ~200 ms — a
  fast tab flick, a tooltip, a value that flashes.
- **Animated content is excluded by design** — donut charts drawing in, loading
  shimmer, count-up animations. A count-up's *final* value is real content that
  changed relative to the previous frame.
- **`_fixed_bands` can declare up to 90 % of the screen pinned.** When a page
  barely scrolled, genuine content rows near both edges look identical in the
  two compared frames, and the inward scans stop only at the 45 % cap. Anything
  inside the bands is never required to be covered.
- **A fling beyond ±75 % of the screen fails to align** and starts a new page —
  fail-safe for content, expensive in frames.
- **Repeated identical rows defeat the peak test.** A list of identical values,
  a column of zeros or a grid of thumbnails makes the runner-up score ≥ 0.8 of
  the best, so the frame is refused and a page is cut. The test fixture is
  explicitly built so that every offset is unique, which is precisely the
  condition real footage violates.
- **A global brightness change breaks alignment**, because `BLOCK_MATCH` is an
  absolute grey level: auto-brightness, a notification shade, a fade, a
  dark-mode toggle mid-recording. Dark mode itself is fine; *changing* is not.
- **`BLOCK_MATCH` = 20 may not see a single changed digit.** The unit of
  comparison is the mean of ~16×2 source pixels; the comment at `:91` asserts a
  changed digit breaks it, and no test or measurement backs that. When it does
  not, `content_covered` declares two frames with different numbers identical
  and `drop_duplicate_content` deletes one. Digit changes that alter a number's
  *width* (999 → 1,004) are detected reliably.
- **`MIN_EVIDENCE` does not mean what it says.** The comment documents 25
  (`:94`); both uses are `MIN_EVIDENCE // 2` = 12 (`:153`, `:161`).
- **Cost.** `_link_to_earlier` is the quadratic term: every frame whose forward
  link fails is compared against 8 samples of *every* page so far, each a ~4 ms
  search. If alignment fails systematically, 200 frames is on the order of
  minutes and 500 frames tens of minutes. `_verify`'s add-back loop is cubic in
  the number of initially-uncovered frames. There is **no cap, no timeout and no
  logging anywhere in this module** — the logger at `:37` is never used.
- **Memory.** Every readable frame's downscaled float32 array (~222 KB) is
  retained in `Selection.frame_info` and held through OCR dispatch. 224 frames
  ≈ 50 MB; a ZIP of ten 500-frame videos ≈ 1.1 GB resident.

---

## 5. Metric coverage — `processing/metric_coverage.py`

Reads each surviving frame locally with Tesseract, works out which metrics it
shows, and keeps the fewest frames covering all of them.

### 5.1 Reading

`read_lines` (`:420`): greyscale → **invert if mean < 100** (a different
threshold from the classifier's 80) → **2× bicubic upscale** →
`image_to_string(lang="fas+eng", config="--psm 6")` (`:430`). At 324×720 input
that is a ~648×1440 OCR, and it is the dominant CPU cost of the stage.

`_norm` (`:100`) applies NFKC, maps Persian **and** Arabic-Indic digits to ASCII,
lowercases, unifies Arabic ي/ك to Persian ی/ک, strips ZWNJ/RLM/LRM, and removes
all whitespace and punctuation. Everything downstream is substring matching on
that gap-free string.

### 5.2 Deciding what a frame shows — `metrics_in_text` (`:226`)

Labels are matched longest-first with consumption (`:216`), so "profile visits"
wins over "views". A label takes a number from its own line, or from the next
line (or the one after, if the next is OCR noise), and only if the line has at
least as many numbers as it has labels — "Average watch time / Follows" over a
lone "4s" credits nobody.

Guards, each added after a real misread:

| guard | what it stops |
|---|---|
| `_is_filter_row` (`:88`) | "All · Followers · Non-followers" over the chart, whose next line is an axis tick |
| `_is_section_header` (`:185`) | "Interactions ⓘ" at a screen edge taking the donut's arc as its value |
| `_NOT_A_VALUE` (`:82`) | chart headers and tab bars containing a metric word |
| `_CLOCK` (`:135`) | the status-bar clock |
| `_gender` (`:208`) | "men" inside "comments", Persian "زن" inside "بازنشر" |
| `_split_kind` (`:169`) | the Followers/Non-followers split under *Interactions*, which is not the schema's split |

### 5.3 Choosing frames

1. **Cross-frame value evidence** (`:490`): a value of 3+ digits paired with a
   label on one frame counts as that metric on any frame showing the same value.
   Added because "Likes 124" read as "'3 124" made two identical frames both
   look necessary.
2. **Greedy set cover** (`:502`), tie-broken by new metrics → total metrics →
   values read → **latest frame**, because a page still loading reads fewer
   values and loads top to bottom.
3. **Two safety nets.** `unrecognised` (`:501`) keeps a frame with no recognised
   metric but a plausible metric number on it (`has_real_number:359`).
   `backstop` (`:515`) keeps a frame showing a **count** no kept frame shows;
   percentages are excluded (`:525`) because OCR reads small ones inconsistently
   and each misread was costing a frame.
4. `drop_duplicate_content` (`selection_pipeline.py:48`) then drops any frame
   whose content is readable in another frame already being sent.

`Coverage.reasons` records why each frame was kept or dropped: `cover`,
`unrecognised`, `backstop`, `covered`, `blank`, `duplicate`.

### What breaks

- **Tesseract missing ⇒ the whole stage is silently skipped.** `tesseract_available`
  (`:405`) requires the binary, the package and *every* language in
  `coverage_ocr_lang`; on failure `select_by_coverage` returns its input
  unchanged with one warning line (`:463`). Every scroll-selected frame then
  goes to OpenRouter: **3–8× the LLM spend**, with nothing in the database
  recording that it happened. The probe is cached for the life of the process,
  so a worker that starts before the filesystem is ready never recovers.
- **Version sensitivity is real and measured.** See §9 — Debian's Tesseract
  5.5.0 and Homebrew's 5.5.3 disagree on noisy frames, which changes which
  frames are sent. Nothing pins the version: the Dockerfile installs whatever
  bookworm serves that day (`Dockerfile:5`) and `requirements.txt` is entirely
  `>=`.
- **`cv2.imread` returning `None` drops the frame with no log at all**
  (`:425`): empty lines ⇒ no metrics ⇒ `has_real_number` false ⇒ reason `blank`
  ⇒ never reaches the vision model. A raised exception, by contrast, is caught
  and the frame is **kept** (`:481`). The safe path and the silent path differ
  by which failure mode the corrupt file triggers.
- **No timeout on `image_to_string`** (`:430`). A pathological image wedges a
  worker slot until the reaper's 2-hour no-progress limit.
- **`_is_filter_row` is a literal prefix test.** It requires the line to start
  with "all" or "همه". Tesseract misreads "All" often — I found `الم Followers
  Non-followers` on a real frame during this verification — and the guard then
  does not fire, so the chart axis below is credited as the followers split.
  Confirmed live: it creates phantom `followers`/`non_followers` on `1f2bdcac`
  frame 171.
- **The icon strip credits three metrics from no labels** (`:256`): any
  "mostly numeric" line above a tab bar becomes likes/replies/shares, without
  checking that there are exactly three numbers or what they are.
- **`_NOT_A_VALUE` contains the bare token `"rate"`** (`:83`), so *any* line
  containing those four letters anywhere is discarded everywhere.
- **Cross-frame value evidence can delete a real frame.** If two different
  metrics on different screens happen to share a 3+ digit value, a frame is
  credited with a metric it does not show, and the frame that really shows it
  can then be dropped as redundant. No test covers a value collision.
- **`_canonical` corrupts some tokens** (`:148`): `1,234.5` becomes `34.5%`,
  because the loop stripping the bar-glyph digit keeps eating while the value
  exceeds 100. Any one-decimal non-percentage becomes `%`-suffixed and is
  therefore excluded from the backstop.
- **K/M defeats value matching.** `8.6K` and `8600` are different tokens, so one
  frame showing each survives as two backstop frames. Instagram abbreviates
  above 10K, so this is systematic for Reels.
- **A metric whose label is never recognised** is invisible here and survives
  only by accident, through the backstop or the unrecognised net.
  `links_clicks` and `external_link_taps` have the thinnest label coverage and
  appear in no test.
- **The selection is unstable under small OCR changes.** One character read
  differently moves a metric between frames and can re-cut the whole set. The
  same six videos went 39 → 34 → 24 frames across three rounds this month.

---

## 6. OCR — `processing/ocr_processor.py`

One request per batch of `ocr_batch_size` = 12 frames (`config.py:62`), each
frame inlined as base64 in a single user message (`:277`), with a strict
JSON schema derived from the pydantic model (`_strict_schema:125`) and
`temperature = 0.1` (`:285`). Model: `qwen/qwen3.8-max` (`config.py:61`).
Timeout 120 s, `max_tokens` 8000.

In production the batches are dispatched as a Celery **chord**
(`tasks.py:160`), all at once, with no inter-batch delay; parallelism is the
worker concurrency of 2. `ocr_batch` never raises — a raised exception in a
chord header would abort the callback — so failures come back as data.

Retries (`_call_api:290`): **5 attempts shared across all failure kinds**, with
`sleep(5)` between most, and `(attempt+1) * 10` seconds for HTTP 429 (`:329`).
`Retry-After` is ignored and there is no jitter. A `finish_reason == "length"`
triggers `_split_on_truncation` (`:237`), which halves the batch and gives each
half a **fresh** 5-attempt budget, recursively.

### What breaks

- **One bad field discards twelve frames.** `_parse_content` (`:225`) is
  all-or-nothing: invalid JSON and a schema mismatch are the same `except`.
  A model that emits `"views": 8.6` instead of expanding `8.6K` loses the whole
  batch, retries 5× at temperature 0.1 — likely reproducing the same output, at
  1.5–4 MB uploaded per attempt — then fails the batch, which marks the **whole
  job** failed.
- **429 handling is lockstep.** Linear back-off, no jitter, ~100 s of total
  budget, with every parallel worker retrying on the same schedule.
- **Truncation recursion is unbounded in cost**; a pathological job can issue
  dozens of multi-megabyte vision requests.
- **Values that fail the plausibility bounds become `None` silently**
  (`:82-89`): percentages outside 0–100, counts above 10 000 000 000 (a guard
  against digit-repetition hallucinations). The aggregate is then computed over
  fewer reads with no signal that this happened.
- `_encode` skips unreadable files with a warning but results are mapped by the
  **unfiltered** batch index (`:405`), so one skipped file shifts every
  subsequent `actual_frame` label. Provenance only; aggregation is
  frame-agnostic.
- Nothing validates that the configured model exists or supports vision and
  strict structured output. The module's own docstring notes the schema
  strictness "happens to pass with the current model and would fail on the
  next one".

---

## 7. Aggregation and persistence

`_aggregate` (`:340`) collects every read of each metric across **all** frames
and batches of the job, then: **the modal value wins if any value repeats;
otherwise `median_low`** (`:371`). `median_low` rather than the max, because
misreads inflate far more often than they deflate — this replaced a `max()`
after a production case where `follows` read as 3154 against a true 31.

It also records `reads`, `min`, `max` and `disputed` (spread above 5 %) per
metric in `summary_quality` — which **is persisted but never acted on**: not in
the job status, not in the Excel export.

**The schema is 28 metrics** (`ocr_processor.py:31`, `db_client.py:410`), of
which 11 are percentages (`_PERCENT_FIELDS:43`): `followers`, `non_followers`,
`gender_men`, `gender_women`, and the seven `source_*` fields (feed, reels_tab,
profile, explore, stories, search, other). The other 17 are counts: `views`,
`accounts_reached`, `interactions`, `likes`, `replies`, `shares`,
`links_clicks`, `sticker_taps`, `navigation`, `forward`, `next_story`, `back`,
`exited`, `profile_activity`, `profile_visits`, `external_link_taps`,
`follows`.

### What breaks

- **Consensus has almost nothing to work with.** Coverage's whole purpose is to
  send the fewest frames, so most metrics now arrive with exactly **one** read,
  accepted unconditionally. The mechanism that was built to catch misreads has
  been designed out of its own inputs.
- With two disagreeing reads, the **smaller** wins by design. With two equally
  frequent modes, the value seen **first** wins silently.
- **Every database call swallows its exception** and returns `False`/`None`/`[]`
  (`db_client.py:118, 135, 152, 190, 218, 261, 292, …`). `finalize_job` ignores
  the return of `save_job_metrics` and then marks the job completed: a DB hiccup
  loses the metrics while the job reports success.
- **No `DATABASE_URL` silently means an in-memory store** (`db_client.py:27`).
  Compose always sets it; a bare `docker run` does not.
- **A failed job still has a metrics row.** `finalize_job` persists before it
  decides the job failed (`tasks.py:216`), so any consumer reading `job_metrics`
  without joining `jobs.status` ingests partial data.
- The Excel export (`db_client.py:442`) caps at 10 000 jobs and writes only
  `summary`; `disputed`, `reads` and `ocr_batches_failed` are not exported, so a
  value from one shaky read looks exactly like one confirmed five times.

---

## 8. Failure modes, collected

**Loses a metric** = silent, and the expensive kind: you find out when someone
checks a number by hand. **Costs frames** = visible on the OpenRouter bill.

| # | Failure | Stage | Likely at 1,000+ | Effect |
|---|---|---|---|---|
| ~~1~~ | ~~Corrupt/unreadable video → 0 frames, recorded as success~~ | 2 | — | **FIXED 2026-09-20**, see §8.1 |
| 2 | Tesseract missing or a language pack absent | 5 | medium | costs 3–8× frames |
| 3 | Tesseract version drift changes which frames are sent | 5 | **certain** | costs frames; §9 |
| 4 | `create_job` fails, poller ignores it | 1 | medium | full OCR spend, nothing persisted |
| 5 | DB write fails, job still reports completed | 7 | medium | loses metrics |
| ~~6~~ | ~~Reaper marks a queued job failed, it then completes~~ | 0 | — | **FIXED 2026-09-20**, see §8.1 |
| ~~7~~ | ~~Frames never swept after a hard kill~~ | 0/2 | — | **FIXED 2026-09-20**, see §8.1 |
| 8 | One bad field discards a 12-frame batch → job failed | 6 | medium | loses a whole job's metrics |
| 9 | Partial horizontal motion drops the moving rows | 4 | medium | loses a metric |
| 10 | `_fixed_bands` declares up to 90 % of the screen pinned | 4 | medium | loses a metric |
| 11 | A screen visible for one sampled frame is dropped | 4 | medium | loses a metric |
| 12 | `BLOCK_MATCH` misses a single changed digit → duplicate drop | 4/5 | unknown | loses a metric |
| 13 | Misread "All" defeats `_is_filter_row` | 5 | **confirmed live** | phantom metrics, costs frames |
| 14 | Value collision in cross-frame evidence | 5 | low | loses a metric |
| 15 | K/M vs expanded number keeps two frames | 5 | **high for Reels** | costs frames |
| 16 | Horizontal scroll / carousel | 4 | medium | costs frames + minutes of CPU |
| 17 | `_link_to_earlier` quadratic blow-up | 4 | medium | tens of minutes per video |
| 18 | Worker RSS creep, OOM kill, poison-message loop | 3 | medium | job never finishes, retries forever |
| 19 | Chord message lost → `finalize_job` never fires | 0 | low | job stuck (files now swept, see §8.1) |
| 20 | Redis redelivery of a ≥1 h task → double execution | 0 | low | double OCR spend, racing cleanup |
| 21 | 429 lockstep retries across parallel workers | 6 | medium | batches lost |
| 22 | Poller backlog downloads everything in one task | 1 | **high** | fills the uploads volume, hard kill |
| 23 | Non-ASCII names slug to empty → id collisions | 1 | medium | two jobs share a directory |
| ~~24~~ | ~~ZIP: one bad video skipped, job completes green~~ | 2 | — | **FIXED 2026-09-20**, see §8.1 |
| 25 | Brightness change / dark-mode toggle mid-recording | 4 | low | costs frames |

**Everything that swallows an exception**, since at scale these are invisible:
`frame_extractor.py` (`:45, :83, :87, :110, :150, :158, :184, :214, :280`),
`frame_classifier.py` (`:42, :58, :69, :78, :131, :152`), `metric_coverage.py`
(`:415, :425, :481`), `ocr_processor.py` (`:88, :221, :233, :310, :438`),
`s3_storage.py` (`:62`), `db_client.py` (twelve sites), `tasks.py`
(`:54, :169, :187, :264, :329, :390, :414, :418`), `ingest.py` (`:110, :121, :134`).

### 8.1 Fixed on 2026-09-20

Three of the rows above are resolved. Each has regression tests built from the
failure it describes, and the suite is at **165 passing** (was 145).

**Failures 1 and 24 — an unreadable video is an error now, never a silent zero.**
`processing/frame_extractor.py` gained `VideoUnreadable`. It is raised when
OpenCV cannot open the file, and when a file opens but yields no frames (frame 0
is always sampled, so zero saved frames means every write failed). The ZIP path
catches it per member and records `"status": "error"`, so a bad video among good
ones is counted into `videos_failed` rather than reported as a success; the job
then fails as incomplete, the same rule already used for a dropped OCR batch.
A ZIP with no video members, and one whose members all fail, each fail with a
message that says so instead of the misleading "No Instagram Insights screens
detected". The 720p probe also checks `isOpened()` first, so an undecodable file
no longer burns up to ten minutes of ffmpeg timeout.
*Tests:* `tests/test_frame_extractor.py` — empty, junk and truncated files, a
missing file, a readable video that must still succeed, every frame failing to
write, the ffmpeg skip, and the three ZIP cases. Two older tests that asserted
the silent-zero behaviour were rewritten; they had pinned the bug.

**Failure 7 — orphaned frames and uploads are swept.**
New beat task `tasks.sweep_orphans`, hourly (`sweeper_interval_seconds`).
It deletes a job directory or an upload only when all three hold: the path sits
directly inside `processing_dir` / `upload_dir`; nothing in it has been touched
for `orphan_max_age_seconds` (6 h, comfortably past the 1 h Celery hard limit);
and its job is not queued or processing. It walks the tree rather than trusting
a directory mtime, because writing frames into `<job>/frames/` leaves `<job>/`
untouched, and it stops at the first recent file so a live job costs one stat.
Anything it cannot determine — an unreadable mtime, a database it cannot reach —
it keeps: deleting a live job's frames is unrecoverable, keeping an orphan for
another hour costs disk. `KEEP_JOB_FRAMES` disables it entirely.
*Tests:* `tests/test_cleanup.py` — an orphan removed, a live job's directory
kept, a recent directory kept, fresh frames inside an old directory kept, an
orphaned upload removed while a live job's upload survives, the
`KEEP_JOB_FRAMES` no-op, everything kept when the active-job list is
unavailable, and a symlink that must not lead it out of `processing_dir`.

**Failure 6 — the reaper measures activity, not age.**
`jobs` gained `last_progress_at` (**migration 0003**, nullable, backfilled from
`created_at`, indexed with `status`), stamped by `update_job_progress` and
`update_job_status`. `fail_stale_jobs` now uses two clocks: a **processing** job
fails after `job_stale_seconds` (2 h) with no progress, measured from
`COALESCE(last_progress_at, created_at)`; a **queued** job fails only after
`queue_stale_seconds` (24 h), because with two worker slots and a backlog of
thousands, waiting is the normal state and not a symptom. The safety net for a
genuinely lost broker message survives, with its own message.
*Tests:* `tests/test_tracking.py` — a stalled job failed, a nine-hour job still
reporting progress spared, a five-hour queued job spared, a thirty-hour queued
job failed, both update paths stamping liveness, and the NULL fallback for rows
predating the migration.

**Deployment note:** migration 0003 must run (`alembic upgrade head`, which
compose already does) before the new reaper behaves as described. Until then
`last_progress_at` does not exist; the COALESCE keeps the old
created_at behaviour rather than erroring.

---

## 9. Where the numbers come from

**Ground truth** — `training/ground_truth/`, 10 videos, each metric mapped to
the frame ranges where it is readable. `training/eval_selection.py` scores any
strategy against it. Current state: **60 frames sent, 0 metrics lost**, against
a floor of 30.

**Human feedback** — `review-rounds/round-01/`, six videos reviewed frame by
frame; `training/eval_feedback.py` replays every verdict as a regression check.
Current state: **21/21 pass locally, 24 frames** (down from 39 before the
fixes).

**Tests** — 145 pass. The coverage fixtures are the exact Tesseract output of
the frames involved.

### Container vs local, measured 2026-09-20

The image was rebuilt at this commit and checked four ways. The model is
byte-identical (`d4a4e0494b7efc75703f8c43c73fed4b` both sides, run
`2026-09-19b-sources`).

| | container | local |
|---|---|---|
| Tesseract | **5.5.0** (Debian) | **5.5.3** (Homebrew) |
| ffmpeg | 7.1.5 | 8.1.1 |
| TensorFlow / Python | 2.21.0 / 3.11.16 | — / 3.9 |
| ground truth | **60 frames, 0 lost** | 60 frames, 0 lost |
| round-01 feedback | **25 frames, 19/21** | 24 frames, 21/21 |
| `run_job` on the two test videos | 3 and 4 frames | 3 and 4 frames |

**Metrics delivered are identical on both sides for all eight videos.** The
container sends different *frames*, never fewer *metrics*.

The two failing checks are both frames the reviewer marked "not needed", which
the container still sends:

- `1f2bdcac` frame 183: 5.5.3 reads `11s 0`, 5.5.0 reads `115 0`. "115" is three
  digits, so `has_real_number` accepts it and the frame is kept as
  "unrecognised".
- `bf96a4ca` frame 033: on a blurred status bar, 5.5.0 hallucinates the Persian
  word `تال` where 5.5.3 reads punctuation. A number plus a 3-letter word is the
  rule for "this could be a metric".

Both are cost, not loss. **Evaluate against the container, not the Mac, when
numbers matter** — production runs in the container.

### Honest caveats

- **The ground truth is my labelling, not an independent human's**
  (`labelled_by` in each `truth.json`).
- **Round-01's six videos are now in the training set**, so their feedback
  checks are no longer unseen tests. `bf96a4ca` and `c70ca2f3` are the genuinely
  held-out ones.
- The gender split can only be captured if the recorder tapped the Gender tab;
  Reel Audience defaults to Age.
- The scroll selection module has no test for horizontal motion, rotation,
  brightness change, animation, a non-portrait aspect ratio, or any input larger
  than 106 frames.

---

## 10. Scale readiness

Nothing below is implemented. It is what I would do before a run of thousands,
in the order I would do it.

### Cost and throughput, per 1,000 videos

Measured: classification runs ~224 frames in ~2 s; a video yields 50–225 frames
and 2–8 MB of JPEGs; coverage OCR runs on 15–81 frames per video at roughly
1–3 s each; one OpenRouter request per video.

- **CPU:** coverage OCR dominates at roughly **7–20 CPU-hours per 1,000
  videos**, plus classification on every extracted frame. At `--concurrency=2`
  that is days of wall-clock on one box. Tesseract and TensorFlow contend for
  the same cores.
- **Disk:** ~5 GB of frames transient for 1,000 videos, and **unbounded if jobs
  die hard** (failure 7).
- **OpenRouter:** ~1,000 requests, each 1.5–4 MB of images. If Tesseract is
  missing it becomes 2,000–7,000 requests with far more image tokens, with no
  signal that it happened.

### Before the run

1. **Pin Tesseract and ffmpeg versions in the Dockerfile**, and pin
   `requirements.txt`. Today two builds a month apart can behave differently,
   and §9 shows the effect is real.
2. **Fail loudly when Tesseract is unavailable** rather than skipping the stage
   — or at least record `Coverage.applied` on the job, so the bill has an
   explanation.
3. ~~Check `cap.isOpened()`.~~ ~~Add a disk sweeper.~~ ~~Make the reaper key on
   last progress.~~ **All three done 2026-09-20 — see §8.1.** The remaining
   deployment step is `alembic upgrade head` for migration 0003.
4. **Give the poller a per-tick budget** (N objects or a time box) and an
   overlap lock, and apply the web route's ZIP validation to it.
5. **Act on `create_job` returning `False`** in the poller: do not enqueue.
6. **Set `worker_max_tasks_per_child`** so a leaking child is recycled, and
   `task_reject_on_worker_lost=False` so an OOM does not loop forever.
7. **Consider raising `frame_interval` or capping frames per video.** A cap is
   the cheapest protection against both the quadratic selection path and the
   disk path.
8. **Reconsider `scale=-2:720` for portrait**, or at least measure OCR accuracy
   at 720 wide against 720 tall on a sample. It is the largest single lever on
   every downstream stage.

### While it runs

Watch, in order of how much they cost if missed:

- jobs failing with "not a decodable video" or "could not be read" — these are
  now reported rather than silent (§8.1), so the count is a real signal about
  the uploads themselves;
- free space on the `processing` and `uploads` volumes, plus the sweeper's
  hourly line `Sweeper removed N job dir(s)…`: a rising count means jobs are
  dying without settling (§8.1);
- the log line `Coverage: N -> M frames`; if `M ≈ N` across many videos,
  coverage is not working — failure 2;
- worker RSS over time — failure 18;
- OpenRouter 429 rate and `batches_failed` — failures 21 and 8;
- objects sitting in `ingesting/` — a job died without settling;
- jobs stuck `queued` far longer than the backlog explains — the reaper no
  longer masks this by failing them at 2 h (§8.1), so it is visible now.

### Keep the feedback loop running

The cheapest insurance is the loop already built: run
`training/review_videos.py` on a sample of each new batch, review the final
frames, and let `training/eval_feedback.py` turn each verdict into a permanent
regression check. Every failure mode in §8 marked "confirmed live" was found
that way, not by reading the code.
