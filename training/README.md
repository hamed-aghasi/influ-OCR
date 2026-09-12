# Training the frame classifier

Rebuilds `instagram_analyzer_app/models/frame_classifier_savedmodel/`, the
TensorFlow model that decides whether an extracted frame is a readable
Insights screen (`good_frame`) or a blurred/transition one (`bad_frame`).

The original training script was never committed — to this repo or to
`hamed-aghasi/OCR`. `train_frame_classifier.py` reconstructs it from the
shipped `frame_classifier_v1.h5`, whose stored config gives the exact
architecture, optimizer and loss. A retrain is a drop-in replacement, not a
new contract.

## Dataset layout

```
data/frames/
  bad_frame/    blurred, mid-scroll, transition, partially-rendered frames
  good_frame/   clean, fully-rendered Insights panels
```

Folder names set the labels: **alphabetical order**, so `bad_frame` = 0 and
`good_frame` = 1, matching `model_metadata.json`. `bad/` and `good/` also work.
Subfolders are scanned recursively, so you can keep per-video folders inside.

## Run

```bash
python -m venv .venv-train && source .venv-train/bin/activate   # Python <= 3.12
pip install -r training/requirements.txt

python training/train_frame_classifier.py --data-dir data/frames --dedup
```

Writes into `instagram_analyzer_app/models/` by default (use `--output-dir` to
stage elsewhere first):

- `frame_classifier_savedmodel/` — what the app loads
- `frame_classifier_v1.h5` — Keras copy, useful for inspecting the model later
- `model_metadata.json` — architecture, epochs, val metrics, label map

Useful flags: `--epochs`, `--batch-size`, `--val-split`, `--lr`, `--seed`,
`--dedup`, `--no-augment`, `--patience`.

## Things worth knowing

**Preprocessing is shared with inference, on purpose.** `preprocess_frame()`
is a copy of `frame_classifier._preprocess`: OpenCV BGR, brighten if mean
luma < 80, resize to 224, divide by 255. No `mobilenet.preprocess_input`, no
`Rescaling` layer. If you change one side, change the other, or the frozen
backbone sees inputs at training time it never sees in production.

**The images stay BGR.** OpenCV decodes BGR and the app never converts, so
training matches rather than "fixing" it. Switching both sides to RGB is a
legitimate improvement — ImageNet weights expect RGB — but it is a change to
both files and needs a retrain, not a one-line edit.

**No horizontal flip in the augmentation.** These are screenshots of text; a
mirrored Insights panel is not an input that exists.

**v1 reported `val_accuracy: 1.0`.** On frames sampled from a handful of
videos that is near-certainly near-duplicates spanning the train/val split,
not a perfect model. `--dedup` runs the app's own perceptual-hash helper
(`processing/dedup.py`) over each class before splitting. The script prints a
warning if it sees ~1.0 without `--dedup`.

**The decision threshold lives in the app**, not the model:
`classifier_threshold = 0.65` in `processing/config.py`. The model emits
P(good_frame); retraining does not change where the cutoff sits, so re-check
it against a held-out set if the score distribution shifts.

**Fine-tuning is not enabled.** v1 froze MobileNetV2 end to end
(`trainable=False`) and trained only the 64→32→1 head. Unfreezing the top
backbone blocks at a low LR is the usual next lever if head-only training
plateaus.

## Verification

The script reloads its own export through
`keras.layers.TFSMLayer(..., call_endpoint="serving_default")` — the same call
`frame_classifier.load_model()` makes — and prints predictions for a few
frames, so a broken export fails at train time rather than at upload time.

Only `model.export()` produces the `serving_default` endpoint; `model.save()`
on a directory does not. Newer Keras omits `keras_metadata.pb` from the
export, which the shipped v1 has. It is not needed by `TFSMLayer`.
