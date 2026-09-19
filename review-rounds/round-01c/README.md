# Review round 01c: round 01 re-run after acting on its feedback

The same six videos as `../round-01/`, re-run with every fix made in response
to that review, and with the retrained classifier (`2026-09-19b-sources` at
0.30). **No OpenRouter call was made.** `../round-01b/` is an intermediate run,
made before the last two fixes; it is kept, not current.

| video | round 01 sent | **01c sent** | your round-01 verdicts now |
|---|---|---|---|
| `1f2bdcac` | 7 | **3** | 183 (rates) no longer sent; 108 and 210–225 content still sent |
| `4985b8cc` | 7 | **4** | 276 (rates) no longer sent |
| `5361dd11` | 2 | **2** | 189 (rates) no longer sent; **237's Top sources now sent** (as 234, the same screen) |
| `8102e03d` | 12 | **5** | only one of 585/603, and one of 471/480 |
| `bf96a4ca` | 5 | **3** | 033 and 054 (not loaded) no longer sent; 060's loaded numbers sent via 087 |
| `edc93edd` | 6 | **7** | everything you marked needed is still sent |
| **total** | **39** | **24** | `training/eval_feedback.py`: 21/21 checks pass |

## What changed, and why

- **Rates, axis ticks, status-bar noise and a loading skeleton** were kept by
  the "unrecognised but has a number" rule, which counted any 2-digit number.
  Now only numbers that could be a metric count.
- **Top sources of views** was missing from the schema, and the classifier had
  been taught those screens were bad. It is now metrics `source_*`: 131
  training frames were relabelled, and the classifier was retrained.
- **Duplicates:** OCR misread "Likes 124" as "'3 124", so two identical frames
  both looked necessary. A value paired with a label on one frame now counts as
  that metric on any frame showing the same value. There is also a pass that
  drops frames whose content is readable in another sent frame.
- **Half-loaded frames:** ties now go to the most complete frame (then the
  later one), and the value backstop guards counts only. Small percentages OCR
  inconsistently ("1.1%" as "11%"), and each misread had kept another frame.
- **A section header** ("Interactions ⓘ") at a screen edge had borrowed the
  donut's arc, read as digits, as its value. Headers no longer take the next
  line's number.

`edc93edd` is the one video that went up (6 → 7): OCR read a count slightly
differently on two near-identical frames (243, 327), and 201 has an unpaired
number. Nothing is lost; OCR variance on counts is the next thing to tighten.

## Your next review

Fill in each video's `feedback.json` here, as in round 01, if you want to
review this round too. `training/eval_feedback.py` picks up every
`review-rounds/*/*/feedback.json` marked `"done"`.
