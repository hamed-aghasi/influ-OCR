#!/usr/bin/env python3
"""Score frame-selection strategies against the hand-built ground truth.

For every ``training/ground_truth/<video>/truth.json`` that is filled in, runs
the deployed classifier (current config) and each selection strategy, then
reports how many frames each would send to OCR and which metrics it would
LOSE — a metric is lost when none of the frames sent falls inside the frame
ranges where the ground truth says that metric is readable.

Also reports the floor: the fewest frames that could carry every metric
(greedy set cover over the ground truth), and metrics already lost at the
classifier, before any selection runs.

No OpenRouter call, no OCR.

    .venv-train/bin/python training/eval_selection.py
    .venv-train/bin/python training/eval_selection.py --strategies phash,scroll --include-drafts
    .venv-train/bin/python training/eval_selection.py --model-dir training/runs/<run> --threshold 0.9
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GT_DIR = REPO / "training" / "ground_truth"
from _bootstrap import bootstrap, scrape_model_overrides  # both must run before processing.* is imported

scrape_model_overrides()   # --model-dir / --threshold into the environment Settings reads
bootstrap(secret="training-only-not-a-real-secret", log_dir=REPO / "training" / "logs")

from processing.config import settings  # noqa: E402
from processing.frame_classifier import classify_frames  # noqa: E402
from processing.frame_selection import select_frames  # noqa: E402
from processing.metric_coverage import tesseract_available  # noqa: E402
from processing.selection_pipeline import choose_ocr_frames  # noqa: E402


def parse_ranges(text: str) -> set:
    """'066-105, 213' -> {66, 69, ..., 105, 213} (frame numbers as in filenames)."""
    out = set()
    for part in (text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        lo, hi = int(lo), int(hi or lo)
        out.update(range(min(lo, hi), max(lo, hi) + 1))
    return out


def frames_dir(video: str) -> Path:
    for cand in (REPO / "train" / video / "frames", REPO / "test-frame-output" / video / "frames"):
        if cand.is_dir():
            return cand
    sys.exit(f"no frames found for {video}")


def fnum(p: Path) -> int:
    return int(p.stem.split("_")[1])


def floor_cover(metric_frames: dict, available: set) -> int:
    """Fewest frames covering every metric (greedy set cover; exact enough here)."""
    todo = {m: fr & available for m, fr in metric_frames.items() if fr & available}
    n = 0
    while todo:
        counts = {}
        for fr in todo.values():
            for f in fr:
                counts[f] = counts.get(f, 0) + 1
        best = max(counts, key=counts.get)
        todo = {m: fr for m, fr in todo.items() if best not in fr}
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategies", default="phash,scroll",
                    help="comma-separated: phash, scroll, coverage (= the production pipeline: scroll, "
                         "metric coverage, duplicate pass)")
    ap.add_argument("--include-drafts", action="store_true", help="also score truth files not yet marked done")
    ap.add_argument("--json", type=Path, help="write per-video results here")
    ap.add_argument("--model-dir", help="evaluate a run's model instead of the deployed one "
                                        "(the folder holding frame_classifier_savedmodel/)")
    ap.add_argument("--threshold", type=float, help="classifier threshold override")
    args = ap.parse_args()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    truths = []
    for f in sorted(GT_DIR.glob("*/truth.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        if doc.get("status") == "done" or (args.include_drafts and doc.get("status") != "todo"):
            truths.append(doc)
    if not truths:
        sys.exit("no filled-in ground truth yet (status must be \"done\"; --include-drafts for drafts)")

    print(f"model {settings.model_dir}")
    print(f"classifier threshold {settings.classifier_threshold}, phash "
          f"{settings.perceptual_hash_size}/{settings.perceptual_hash_distance}, "
          f"{len(truths)} video(s)\n")
    rows, totals = [], {s: [0, 0] for s in strategies}
    for doc in truths:
        video = doc["video"]
        paths = sorted(frames_dir(video).glob("frame_*.jpg"))
        metric_frames = {m: parse_ranges(v.get("frames", "")) for m, v in doc["metrics"].items()
                         if v.get("frames")}
        cls = classify_frames(paths, job_id=f"eval-{video[:8]}")
        if cls.get("error"):
            sys.exit(f"classifier error: {cls['error']}")
        good = cls["good_paths"]
        scores = {Path(g["path"]): g["confidence"] for g in cls["good_frames"]}
        good_nums = {fnum(p) for p in good}
        lost_at_classifier = sorted(m for m, fr in metric_frames.items() if not fr & good_nums)
        row = {"video": video, "metrics": len(metric_frames), "frames": len(paths), "good": len(good),
               "floor": floor_cover(metric_frames, {fnum(p) for p in paths}),
               "lost_at_classifier": lost_at_classifier, "strategies": {}}
        for s in strategies:
            if s == "coverage":           # scroll selection, then metric coverage (Tesseract)
                if not tesseract_available():
                    sys.exit("coverage strategy needs tesseract with " + settings.coverage_ocr_lang)
                chosen = choose_ocr_frames([paths], good, scores, job_id=f"eval-{video[:8]}").final
            else:
                chosen = select_frames([paths], good, scores, strategy=s).selected
            sent = {fnum(p) for p in chosen}
            lost = sorted(m for m, fr in metric_frames.items() if not fr & sent)
            row["strategies"][s] = {"sent": len(chosen), "lost": lost,
                                    "frames": sorted(sent)}
            totals[s][0] += len(chosen)
            totals[s][1] += len(lost)
        rows.append(row)

        print(f"{video[:12]}  metrics {row['metrics']:>2}  floor {row['floor']}  good {row['good']:>3}"
              + (f"  LOST AT CLASSIFIER: {lost_at_classifier}" if lost_at_classifier else ""))
        for s in strategies:
            r = row["strategies"][s]
            print(f"    {s:<7} sent {r['sent']:>3}  lost {len(r['lost'])}" + (f"  {r['lost']}" if r["lost"] else ""))

    print("\n" + "=" * 60)
    print(f"floor (fewest frames that could carry every metric): {sum(r['floor'] for r in rows)}")
    for s in strategies:
        print(f"{s:<7} frames sent {totals[s][0]:>4}   metrics lost {totals[s][1]}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
