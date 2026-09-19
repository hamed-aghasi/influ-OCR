# Training runs

Every model ever trained, one folder per run. **Nothing here is ever
overwritten or deleted** — this is the history of the classifier, and a run you
throw away is a result you cannot reproduce once its dataset has moved on.

`train_frame_classifier.py` writes new runs here by default
(`training/runs/<UTC timestamp>/`). Deploying a model into the app is a separate
step (`--deploy`) — `instagram_analyzer_app/models/` holds only the *currently
deployed* model, and its `model_metadata.json` names the run it came from.

## What a run folder holds

| | |
|---|---|
| `frame_classifier_savedmodel/` | the model, in the format the app loads |
| `frame_classifier.keras` | native Keras copy — reloadable for warm-starting a later round |
| `frame_classifier_v1.h5` | legacy copy |
| `model_metadata.json` | architecture, dataset, per-video counts, split, seed, git SHA |
| `dataset/` | **the labelled frames this model was trained on** |
| `predictions.csv`, `errors.md`, `per_video.md` | what it got right and wrong |
| `threshold_sweep.json`, `history.json` | the threshold table and training curves |

`dataset/` is copied in rather than referenced. It costs a few MB and means a run
stays reproducible even after `train/` has been re-sorted, relabelled or cleared.

## Runs

| run | what it is | deployable? |
|---|---|---|
| `2026-09-12-dark-mode-retrain` | first proof that dark-mode screens are learnable | **no** — trained and tested on one video |
| `2026-09-12-split-mechanism-test` | verification of the rewritten training script | **no** — synthetic split, 2 halves of one video |
| `2026-09-14-dedup` | first 58-video run, within-video dedup | superseded |
| `2026-09-14-nodedup` | first 58-video run, no dedup; deployed 2026-09-14 at threshold 0.95 | superseded — backed up in `models/backup_20260919T100558Z/` |
| `2026-09-19-relabel` | same data after `good` was narrowed to schema metrics (`training/relabel/`) | **deployed 2026-09-19 at threshold 0.30** |

The 2026-09-12 runs are not production models; they are kept as the evidence
behind the current design. The 2026-09-14 and 2026-09-19 runs were trained on
`train/` directly and carry no `dataset/` copy — `training/relabel/labels_before_2026-09-19.json`
records the labels the 09-14 runs saw.

## Restoring an older model

```bash
cp -R training/runs/<run>/frame_classifier_savedmodel \
      instagram_analyzer_app/models/frame_classifier_savedmodel
cp training/runs/<run>/model_metadata.json \
   instagram_analyzer_app/models/model_metadata.json
```

Or re-run the trainer with `--deploy`, which backs up the live model first.
