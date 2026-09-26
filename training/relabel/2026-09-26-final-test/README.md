# Relabel 2026-09-26: a mid-load frame from the Final_Test run

Triggered by the reviewer inspecting the frames the 2026-09-26 Final_Test run
sent to OpenRouter (`Final_Test_output/<video>/frames_sent/`). Frame 123 of
`02d4f7e5-efc6-46e7-8665-50befdc4028f` was sent although the screen had not
finished loading: Overview, Views, Interactions and Audience still show
placeholder bars and the Interactions donut is fading in, while Profile activity
had already rendered (54 / 53 / 1 / 0). That rendered section is what got the
frame past the classifier and kept it in the coverage set.

**Rule** (unchanged from `2026-09-20-loading` and `2026-09-22-loading-2`): an
Insights screen is `bad` while any visible metric section shows placeholder
bars or values still fading in, even when another section is loaded.

Labels come from the reviewer looking at the image, never from OCR output
(`2026-09-20-loading/README.md` records why). Candidate frames were laid out
between their time neighbours (`Final_Test_output/_mid_load_review/`) and
confirmed or rejected one by one before being added here.

## What was labelled

The reviewer went through `Final_Test_output/_mid_load_review/CANDIDATES.jpg`
(each candidate between its time neighbours) and decided:

| frame | label | why |
|---|---|---|
| `02d4f7e5` 120, 123, 126, 129 | bad | placeholder bars / donut fading in while Profile activity had rendered |
| `02d4f7e5` 165 | good | the loaded twin: every section settled |
| `2298108b` 186 | bad | Overview values grey, Views donut dim — fading in |
| `2298108b` 189, 192 | good | settled (189 judged settled by the reviewer: 52 / 21 still slightly grey but readable) |
| `57e39f36` 306 | bad | Overview values grey; 309 (already good) is settled |

Rejected, not labelled: `05ea28da` 663 (carousel switching story, not a load
state) and `c70ca2f3` 069 (only the Views-over-time chart is a placeholder; every
schema metric is rendered). The loaded twins are labelled `good` as the earlier
loading rounds did, so the classifier learns the contrast, not just the negative.

## Source

Frames: `Final_Test_output/<video>/frames/` — the exact frames the run
extracted (and, for the sent ones, encoded into the request). Videos:
`Final_Test/`. Both stay out of git, like `review-rounds/`. `57e39f36` is a
ground-truth video already in `train/`, so `apply.py` falls back to
`train/<video>/frames/` when the round folder has no frames for a video — the
one line this script adds to the round-03 one.

Two Finder-created duplicate symlinks were found in the training set and
**left alone** so this round's before/after compares like with like:
`train/7cb649a2…/good/frame_000087 2.jpg` and `train/3c50a8ce…/good/frame_000288 2.jpg`
each point at the same frame as the real entry, so the trainer counts that
`good` frame twice. Remove them in a separate pass.

## Apply / undo

```
python training/relabel/2026-09-26-final-test/apply.py --apply
python training/relabel/2026-09-26-final-test/apply.py --restore
```

`--apply` copies each labelled frame into `train/<video>/{good,bad}/` and the
video into `train/`, skipping frames already labelled, and records every added
path in `added.json`. `--restore` removes exactly those paths; the copied video
and the now-empty folder are left, as in the earlier rounds.
