# Relabel of 2026-09-19: `good` narrowed to schema metrics

The first labelling round used a broad rule, roughly "any number visible". It
taught the classifier to keep charts, the rates list and top-sources screens.
Those carry no metric the OCR step extracts, but they were still sent and paid
for. This relabel narrows `good` to:

> at least one of the schema metrics readable with its label (or icon) and
> its complete number.

Everything else is `bad`: views-over-time and retention charts, the rates list,
top sources, the audience age/country split, skeletons, blur, and
non-Insights screens. The edge case: one metric cut off but another fully
readable stays `good`.

**Amended the same day:** the Audience **gender** split (Men % / Women %) became
schema metrics `gender_men` / `gender_women`, so a readable gender split is
`good` from now on. No existing label had to change: an OCR scan of all 1,541
`bad/` frames found none showing it, and all 188 frames in `train/` that show
it (22 videos) already score above the 0.30 threshold. In every one of them it
shares the screen with another metric; a gender-only screen has never been
labelled, so label one `good` if you meet it.

## What was done

- **Only good → bad moves.** `bad/` was not touched. 179 of 903 good frames
  moved, leaving 724 good and 1,541 bad (2,265 labels, unchanged).
- **49 videos** were labelled by visual inspection (Claude, via four parallel
  agents working from the rule above, plus one video done directly). A random
  sample of 24 decisions (12 each way) was checked by eye and all 24 held.
- **9 videos** that also have ground truth (`training/ground_truth/`) were
  derived from it. A frame stays good if and only if it falls inside some
  metric's readable range, so labels and ground truth agree by construction.
- 7 frames in those 9 videos that you had labelled `bad` fall inside a
  ground-truth readable range: fade-in edges, listed in `gt_bad_to_good.json`.
  They were left as you labelled them.

**None of this was human-reviewed.** Spot-checking `decisions/*.json` (every
entry has a `why`), or opening `relabel.html` against the new folders, is the
cheapest way to catch a systematic mistake.

## Files

| | |
|---|---|
| `decisions/<video>.json` | per-frame decision and reason, for every frame that was in `good/` |
| `labels_before_2026-09-19.json` | full snapshot of every label before the move |
| `apply_relabel.py` | dry run / `--apply` / `--restore` (exact undo from the snapshot) |
| `gt_bad_to_good.json` | the 7 edge frames above, not applied |
| `relabel.html` | the click-to-mark page (superseded; kept for a human re-review) |
