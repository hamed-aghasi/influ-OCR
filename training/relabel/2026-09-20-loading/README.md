# Relabel 2026-09-20: the loading state

Triggered by the round-02 review. The reviewer found, across 16 of 20 videos,
that frames whose on-screen numbers had not finished loading were reaching the
final set — and suspected the round-01 fixes had regressed.

**They had not.** The deployed model was the retrained one (`2026-09-19b-sources`,
hash matching its run folder) and all 21 round-01 checks still passed. What round
02 exposed is that the round-01 fix for this symptom was narrow: two `bad` labels
from one video, plus a tie-break inside metric coverage. Neither teaches a
classifier what a half-loaded Insights screen looks like.

Tracing every frame the reviewer flagged:

| frames rejected by the reviewer | kept by |
|---|---|
| 25 of 29 | the `unrecognised` net in metric coverage |
| 3 | the value backstop |
| 1 | the set cover |

and of the frames they said were missing, 11 of 13 were dropped by scroll
selection before coverage saw them — though in every case but one, the same
numbers were delivered by another frame that *was* sent. So the symptom was
wasted OCR input, not lost data.

## What is labelled here

**76 frames: 52 `bad`, 24 `good`, across 18 videos** (`labels.json`).

- The reviewer's own verdicts on 16 videos.
- Skeleton frames from `8ba8ce38` and `e6c89684` that I confirmed by eye on a
  contact sheet: the whole panel is grey placeholder bars.
- Loaded counterparts the reviewer named, plus Interactions donuts from
  `5198f946`, `a54e4eda`, `c20db917` and `d7750aef` that are plainly loaded and
  that Tesseract fails to read — labelling them `good` states the case
  explicitly.

**No perceptual-hash shortlist.** Frames are taken straight from
`review-rounds/round-02/<video>/frames`, so a loading frame and its loaded twin
— which differ only in the digit regions, exactly what a phash collapses — both
reach the label set. This is why the earlier corpus had almost no examples of
the loading state: on `4fc5cd74`, 129 of 159 frames had never been labelled at
all.

## An automatic pass that was built and discarded

I first generated candidates mechanically: a classifier-good frame with no
recognised metric *and* no candidate value is a skeleton, so label it `bad`.
It produced 59 candidates. A contact sheet showed **about half were fully
loaded screens** — Interactions donuts reading 255, 395 and 2,517 — that
Tesseract had simply failed to read. `5198f946` frame 738 is the clearest: a
complete screen with Interactions 395, Followers 98.3%, Non-followers 1.7%.

Labelling those `bad` would have repeated the 2026-09-19 mistake, where marking
Top-sources screens `bad` taught the classifier to reject a metric the schema
needed. **OCR failure is not the loading state.** The heuristic is recorded here
so it is not tried again without that check.

## Undo

```
python training/relabel/2026-09-20-loading/apply.py --restore
```

removes exactly the frames `added.json` records. The round-02 videos copied into
`train/` are left; delete `train/<video>/` and `train/<video>.mp4` to drop those
too.
