# docs/history — how the project got here

Session logs and superseded documents, kept for the record. Nothing in this
directory is needed to run or operate the app; the current documentation is
`README.md`, `RUNBOOK.md` and `PIPELINE.md` at the repository root.

| File | What it is | Status |
|---|---|---|
| `HANDOFF-2026-09-29.md` | Dated session log, 2026-08-22 → 2026-09-29: every change, its evidence, and the open threads at the time. Newest section first. | Superseded by `README.md` (known issues) and `RUNBOOK.md`; the evidence trail is still valid. |
| `SESSION-2026-09-12.md` | The dark-mode classifier diagnosis and the start of retraining. | Historical. |
| `SESSION-2026-09-19.md` | The frame-selection overhaul (scroll-aware selection, Tesseract coverage) and review round 01. Refers to the pre-2026-09-26 single-file modules. | Historical; module names are now packages (`processing/frame_selection/` etc.). |
| `PROXY_SUPPORT_PLAN.md` | Proposal for HTTP-proxy support in the OpenRouter client. | **Never implemented.** Its file/function citations no longer resolve. |
| `2026-09-30-release-cleanup.md` | The release cleanup: commits, verification evidence, image id, model digest. | Current record of the release state. |

Related material elsewhere:

- `docs/eval-2026-08-22/` — the model benchmark that chose the OCR model (`bench.py` has a
  hard-coded path from the original author's machine and is kept as a frozen artefact).
- `test-frame-output/`, `-v2/`, `-v3/` — three frame-review rounds on the two test videos;
  their READMEs say what each shows. The frames are local only; `training/eval_selection.py`
  reads them from there.
- `review-rounds/` — human review rounds; the tracked `feedback.json` files are the
  regression corpus for `training/eval_feedback.py`.
- `training/relabel/` — every label change to the training set, each with an exact undo.
