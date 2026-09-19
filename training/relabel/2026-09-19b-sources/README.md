# Relabel 2026-09-19b: Top sources of views, and round-01 feedback

This round was triggered by the review of `review-rounds/round-01/`. The reviewer
marked `5361dd11` frame 237 ("Top sources of views": Feed / Reels tab / Profile /
Explore %) as *should have been sent*. The 2026-09-19 relabel had made those
screens `bad`, so the classifier scored it 0.007. The reviewer confirmed that
Top sources is needed. It became schema metrics `source_*` (see
`processing/ocr_processor.py`), and the rule is now:

> `good` = at least one schema metric readable with its label and value,
> **including a Top sources of views row**.

## What changed

- **131 frames `bad` → `good`** (`moves_sources.json`). Candidates came from
  two places: an OCR scan of every `bad/` frame for readable source rows (128),
  and frames the earlier relabel had moved to bad for showing top sources (4
  not found by OCR). All 132 candidates were checked by eye; 131 show a readable
  source row. One was excluded: `1eed0fb5` 135, a mid-transition frame with the
  percentages cut off.
- **Five new training videos** from round 01 (`round01_labels.json`), labelled
  **only** on the frames the reviewer judged. Frames and videos are copied from
  `review-rounds/round-01/` and `test-frame-output-v3/`; the originals are not
  touched.
  - `good`: 5361dd11 237; 1f2bdcac 108/168/210/216/219/225; bf96a4ca 060;
    edc93edd 231/264/297/318/327/348.
  - `bad`: 5361dd11 189, 1f2bdcac 183, 4985b8cc 276 (rates only); bf96a4ca
    033/054 (numbers not loaded yet).
  - `8102e03d` is not added: its feedback (duplicates) is about selection, not
    the classifier.

Labels afterwards: 878 good, 1,417 bad, 63 videos.

## Undo

`python training/relabel/2026-09-19b-sources/apply.py --restore` puts every
moved file back, using `labels_before.json`. To drop the round-01 videos,
delete `train/<video>/` and `train/<video>.mp4` for the five videos listed in
`round01_labels.json`.
