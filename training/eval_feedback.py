#!/usr/bin/env python3
"""Replay human review feedback as regression checks.

For every review-rounds/<round>/<video>/feedback.json marked "done", re-runs the
current pipeline (the deployed model and config, or --model-dir/--threshold) on
that video's extracted frames and checks each verdict:

- needed: false            the frame, or a frame with the same content, is NOT sent
- needed: true             the frame, its exact content, or every metric it shows is sent
- should_have_been_sent    the same (so a later frame with the identical screen counts)
- duplicate_groups         at most one frame of each group is sent

"Same content" is the pipeline's own row-by-row check (frame_selection.
content_covered), so a later frame showing the identical screen counts. No
OpenRouter call.

    .venv-train/bin/python training/eval_feedback.py
    .venv-train/bin/python training/eval_feedback.py --model-dir training/runs/<run> --threshold 0.5
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
from _bootstrap import bootstrap, scrape_model_overrides  # both must run before processing.* is imported

scrape_model_overrides()   # --model-dir / --threshold into the environment Settings reads
bootstrap(secret="training-only-not-a-real-secret", log_dir=REPO / "training" / "logs")

from processing.config import settings  # noqa: E402
from processing.frame_classifier import classify_frames  # noqa: E402
from processing.frame_selection import content_covered  # noqa: E402
from processing.metric_coverage import metrics_in_text, read_lines  # noqa: E402
from processing.selection_pipeline import choose_ocr_frames  # noqa: E402


def check_video(folder: Path, fb: dict):
    frames = sorted((folder / "frames").glob("frame_*.jpg"))
    by_name = {p.name: p for p in frames}
    cls = classify_frames(frames, job_id=f"fb-{folder.name[:8]}")
    if cls.get("error"):
        sys.exit(f"classifier error: {cls['error']}")
    scores = {Path(g["path"]): g["confidence"] for g in cls["good_frames"]}
    choice = choose_ocr_frames([frames], cls["good_paths"], scores, job_id=f"fb-{folder.name[:8]}")
    final, sel = choice.final, choice.selection
    sent = {p.name for p in final}

    def shown(name):          # the frame itself, or its exact content, is sent
        return name in sent or content_covered(sel, by_name[name], final)

    cov = choice.coverage
    sent_metrics = set().union(*(cov.metrics.get(p, set()) for p in final)) if cov else set()

    def delivered(name):
        """(ok, how): the frame, its exact content, or every metric it shows is sent."""
        if name in sent:
            return True, "sent"
        if content_covered(sel, by_name[name], final):
            return True, "same content sent"
        m = cov.metrics.get(by_name[name]) if cov and by_name[name] in cov.metrics else \
            metrics_in_text(read_lines(by_name[name]))
        if m and m <= sent_metrics:
            return True, f"its metrics sent elsewhere ({len(m)})"
        missing = sorted(m - sent_metrics) if m else []
        return False, f"LOST {missing}" if missing else "LOST (no metric recognised on it)"

    results = []
    for f in fb.get("sent_frames", []):
        if f.get("needed") is False:
            ok = not shown(f["frame"]) if f["frame"] not in sent else False
            results.append((ok, f"not needed    {f['frame']}", "sent" if not ok else "not sent"))
        elif f.get("needed") is True:
            ok, how = delivered(f["frame"])
            results.append((ok, f"needed        {f['frame']}", how))
    for f in fb.get("should_have_been_sent", []):
        ok, how = delivered(f["frame"])
        results.append((ok, f"should send   {f['frame']}", how))
    for group in fb.get("duplicate_groups", []):
        n = sum(1 for g in group if g in sent)
        results.append((n <= 1, f"duplicates    {', '.join(g[6:12] for g in group)}", f"{n} sent"))
    return results, final


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", default=str(REPO / "review-rounds"))
    ap.add_argument("--model-dir")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    items = []
    for fb_path in sorted(Path(args.rounds).glob("*/*/feedback.json")):
        fb = json.loads(fb_path.read_text(encoding="utf-8"))
        if fb.get("status") == "done":
            items.append((fb_path.parent, fb))
    if not items:
        sys.exit("no feedback marked done")

    print(f"model {settings.model_dir}  threshold {settings.classifier_threshold}\n")
    passed = total = sent_total = 0
    report = []
    for folder, fb in items:
        results, final = check_video(folder, fb)
        sent_total += len(final)
        print(f"{folder.parent.name}/{folder.name[:8]}  sends {len(final)}: {[p.name[6:12] for p in final]}")
        for ok, what, detail in results:
            print(f"    {'ok  ' if ok else 'FAIL'} {what:<40} {detail}")
        passed += sum(ok for ok, _, _ in results)
        total += len(results)
        report.append({"video": folder.name, "round": folder.parent.name, "sent": [p.name for p in final],
                       "checks": [{"ok": ok, "check": w, "detail": d} for ok, w, d in results]})
    print(f"\n{passed}/{total} feedback checks pass; {sent_total} frames sent across {len(items)} video(s)")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
