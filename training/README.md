# Training the frame classifier

Rebuilds the TensorFlow model that decides whether an extracted frame is worth
sending to the OCR model. **OpenRouter is billed per image**, so this classifier
is a cost control: keep the frames that carry the numbers, drop the rest.

The scripts:

| | |
|---|---|
| `extract_frames.py` | videos → raw frames, for you to sort by hand |
| `train_frame_classifier.py` | sorted frames → a model, plus reports saying what to fix next |
| `trace_frames.py` | a model + a video → exactly which frames would reach OpenRouter |
| `eval_selection.py` | frames sent and **metrics lost** per selection strategy, against `ground_truth/` |
| `eval_feedback.py` | replays every review round's `feedback.json` marked done as regression checks |
| `review_videos.py` | a human review round: per-video `review.html`, `final/`, a `feedback.json` to fill in |
| `verify_model.py` | inventory + sha256 digest of the deployed model, optional live prediction |
| `relabel/apply_relabel.py`, `relabel/<date>/apply.py` | label changes to `train/`, each with an exact undo |
| `_bootstrap.py` | shared setup the scripts call before importing the app (not a CLI) |

Run `eval_selection.py` **and** `eval_feedback.py` before changing selection, coverage, the
threshold or the model; both are offline (no OpenRouter). `eval_feedback.py` takes about an hour.

and the folders:

| | |
|---|---|
| `runs/` | **every model ever trained**, with the dataset it was trained on — see `runs/README.md` |
| `ground_truth/` | hand-labelled metric positions for 10 videos — see `ground_truth/README.md` |
| `relabel/` | every label change, dated, with its undo — see `relabel/README.md` |
| `logs/`, `_trace_work/` | scratch output; safe to delete |

`instagram_analyzer_app/models/` holds only the *currently deployed* model.
Training history lives in `runs/` and is never overwritten.

## Environment

A prepared virtualenv lives at `.venv-train/` in the repo root (Python 3.9,
TensorFlow 2.20, Keras 3.10). Use it directly — no activation needed:

```bash
.venv-train/bin/python training/extract_frames.py
```

To rebuild it from scratch: `python -m venv .venv-train &&
.venv-train/bin/pip install -r training/requirements.txt` (Python ≤ 3.12).

The original training script was never committed — to this repo or to
`hamed-aghasi/OCR`. `train_frame_classifier.py` reconstructs it from the shipped
`frame_classifier_v1.h5`, whose stored config gives the exact architecture,
optimizer and loss, so a retrain is a drop-in replacement rather than a new
contract.

## 1. Extract

```bash
mkdir -p train && cp /path/to/*.mp4 train/
python training/extract_frames.py
```

Each video gets its own folder:

```
train/
  my-video.mp4
  my-video/
    frames/          every extracted frame
    good/            <- you move frames here
    bad/             <- you move frames here
    _shortlist/      symlinks: one representative per near-duplicate cluster
    clusters.json    representative -> the frames it stands in for
    video_info.json  resolution, fps, theme, extraction settings
```

Extraction calls the app's own `extract_frames_from_video`, so it is identical to
production: ffmpeg 720p downscale then every 3rd frame. **No classifier runs at
this stage** — the point is to see every frame the app would have to judge.

`_shortlist/` is optional. A 19-second clip yields ~155 frames of which only ~60
are visually distinct, because the screen barely changes during a slow scroll.
Sorting the shortlist first is a shortcut; the full set stays in `frames/` either
way. It uses the same perceptual-hash settings as `processing/dedup.py`, so a
cluster here is a cluster in production.

Fill in `ui_version` and `language` in `video_info.json` if you know them — the
training reports slice their metrics by those fields, which is how you find out
whether one Instagram UI version is dragging the model down.

## 2. Sort

Move each frame in `frames/` into `good/` or `bad/`.

**`good` means the frame carries information we need to extract** — a metric
rendered as a readable number. Everything else is `bad`: loading skeletons where
the values haven't appeared yet, mid-scroll blur, the feed, the reel playing, and
screens that are genuinely Instagram Insights but show no number we want.

That last category is the trap that broke v1. Its labels meant "clean,
fully-rendered Insights panel", which is not the same question. On a dark-mode
test video it kept exactly one frame — the "rates" tab, sharp and fully rendered
and containing none of the 19 metrics in the schema — while ~60 frames showing
`Views 24,774 / Accounts reached 17,088` were all rejected at confidence 0.0000.
The job reported success with an empty summary.

You don't need a written rule for the edge cases. Sort by judgement; the
`errors.md` report after each round lists the frames where the model disagreed
with you most confidently, which is where an inconsistent label shows up.

## 3. Train

```bash
python -m venv .venv-train && source .venv-train/bin/activate   # Python <= 3.12
pip install -r training/requirements.txt

python training/train_frame_classifier.py --data-dir train --dedup
```

Writes to `training/runs/<UTC timestamp>/` — **not** into the app. Installing a
model is a separate, explicit step:

```bash
python training/train_frame_classifier.py --data-dir train --deploy
```

`--deploy` backs up the current model to `models/backup_<timestamp>/` first,
stages the new one beside the live directory and swaps it in atomically, and
refuses to run at all if verification fails.

Both dataset layouts are accepted: `train/<video>/{good,bad}/` (preferred) and
the legacy `{good_frame,bad_frame}/` at the top level. Under the legacy layout a
subfolder inside the class directory names the video.

Useful flags: `--epochs`, `--batch-size`, `--val-split`, `--holdout-videos`,
`--lr`, `--seed`, `--dedup`, `--threshold`, `--min-recall`, `--no-augment`,
`--cache-file`, `--run-name`.

## What a run gives you

| file | what it's for |
|---|---|
| `frame_classifier_savedmodel/` | what the app loads |
| `frame_classifier.keras` | native Keras copy, reloadable for later rounds |
| `frame_classifier_v1.h5` | legacy copy |
| `model_metadata.json` | architecture, dataset, per-video counts, split, seed, git SHA, threshold sweep |
| `predictions.csv` | every frame, train and val: score, your label, prediction, brightness |
| `errors.md` | disagreements, ranked by how confident the model was when it disagreed |
| `per_video.md` | which video is dragging the model down |
| `threshold_sweep.json` / `history.json` | the sweep table and the training curves |

`errors.md` is the one to read between rounds. Validation errors are the model's
blind spots. **Training errors matter just as much**: a frame the model refuses
to fit after 20 epochs on data it has already seen is usually a mislabel.

## Things worth knowing

**The split is by video, not by frame.** Frames sampled every 3rd frame from one
recording are near-identical, so a random split puts `frame_000117` in train and
`frame_000120` in validation and the model scores itself on what it memorised.
That is how v1 shipped `val_accuracy: 1.0` and was still blind in practice.
Whole videos go to one side or the other, bucketed on a stable hash of the video
name so adding a video next round doesn't reshuffle the previous one and make the
numbers incomparable. Pin it explicitly with `--holdout-videos`.

With only one video there is nothing to hold out and the script says so. Two
videos of the *same* screens are also weakly independent — the axis that matters
is UI version, theme and language, not video identity.

**`--dedup` is a volume reducer, not the leakage control.** It drops
near-duplicates *within* each video and runs *after* the split, so it can never
move a frame's twin across the boundary.

**The decision threshold is swept and reported.** The model emits P(good); the
app decides at `classifier_threshold` in `processing/config.py` (0.30 for the
model deployed 2026-09-26, run `2026-09-26-final-test`). Those are
separate things and they drift apart on every retrain, so each run prints
precision, recall, F1 and **frames kept** across the whole threshold range, plus
the cheapest threshold still holding recall above `--min-recall` (default 0.90).
Recall is the constraint because a dropped good frame is a metric nobody can
recover, while a kept bad frame only costs tokens. If a better cutoff exists the
run prints the literal `CLASSIFIER_THRESHOLD=...` line to set — pydantic
`BaseSettings` picks it up with no code change.

**Dark mode has to come from real data.** The augmentation is photometric
(±15% brightness, ±15% contrast) plus a small crop. That cannot turn a light-mode
screenshot into a dark-mode one — that is a polarity inversion, not a brightness
nudge. The run warns if fewer than 10% of your `good` frames are dark.

**Preprocessing is shared with inference, on purpose.** `preprocess_frame()` is a
copy of `frame_classifier._preprocess`: OpenCV BGR, brighten if mean luma < 80,
resize to 224, divide by 255. No `mobilenet.preprocess_input`, no `Rescaling`
layer. If you change one side, change the other, or the frozen backbone sees
inputs at training time it never sees in production.

**The images stay BGR.** OpenCV decodes BGR and the app never converts, so
training matches rather than "fixing" it. Switching both sides to RGB is a
legitimate improvement — ImageNet weights expect RGB — but it is a change to both
files and needs a retrain, not a one-line edit.

**`--no-dark-boost` is a footgun.** The app applies the boost unconditionally and
`config.py` has no switch, so a model trained without it cannot be served
correctly. The script warns loudly.

**No horizontal flip in the augmentation.** These are screenshots of text; a
mirrored Insights panel is not an input that exists.

**Fine-tuning is not enabled.** v1 froze MobileNetV2 end to end and trained only
the 64→32→1 head (84,097 parameters). Unfreezing the top backbone blocks at a low
learning rate is the usual next lever once head-only training plateaus — keep
every BatchNorm layer frozen if you do.

## 4. Measure what the model does to a real video

```bash
MODEL_DIR=training/runs/<run> .venv-train/bin/python training/trace_frames.py video.mp4 out/
```

Runs extraction → classification → scroll-aware selection → Tesseract metric
coverage → batching (the production sequence, `selection_pipeline.choose_ocr_frames`)
and **stops at the OpenRouter boundary**: no network call, no API key, no cost.
Prints the frames each request would carry, and writes `ocr_input/` (the surviving
frames), a labelled `contact_sheet.jpg`, `frame_trace.json`, and — with `--all` —
the frames the classifier rejected and the ones selection dropped, so a missing
metric can be traced to the stage that lost it.

With `MODEL_DIR` unset it traces the currently deployed model. It refuses to
overwrite an existing trace, since a trace is evidence about one (model, video)
pair and silently replacing it destroys the comparison.

## Verification

The script reloads its own export through
`keras.layers.TFSMLayer(..., call_endpoint="serving_default")` — the same call
`frame_classifier.load_model()` makes — prints predictions for a few frames of
each class, and **fails if good frames score below bad ones**, which catches an
inverted sigmoid before it reaches the app rather than after.

Only `model.export()` produces the `serving_default` endpoint; `model.save()` on a
directory does not.
