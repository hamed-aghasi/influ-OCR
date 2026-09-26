"""Run a video through the pre-OCR pipeline and report exactly which frames
would be sent to OpenRouter.

Runs the same stages as tasks.run_job, using the project's own modules:
extraction -> classification -> frame selection (config.frame_selection,
"scroll" by default) -> metric coverage via Tesseract (when
config.coverage_selection is on) -> batching. Then it STOPS. No network call,
no API key, no cost.

This is how you measure what a model actually does to a real video — the frames
it keeps, the frames it drops, and how many OpenRouter requests that implies.
Point it at any trained model with MODEL_DIR::

    MODEL_DIR=training/runs/<run> python training/trace_frames.py video.mp4

With MODEL_DIR unset it uses the model currently deployed in
instagram_analyzer_app/models/.

Results are written to <project>/test-frame/<video-stem>/ unless an outdir is
given. Pass --all to also copy the frames the classifier rejected and the ones
dedup dropped, so a missing metric can be traced to the stage that dropped it.

Usage:  python trace_frames.py <video> [outdir] [--all]
"""

import json
import os
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
APP = PROJECT / "instagram_analyzer_app"
RESULTS_ROOT = PROJECT / "test-frame"

from _bootstrap import bootstrap  # placeholder SECRET_KEY, LOG_DIR, PROCESSING_DIR, app on sys.path — before processing.*

bootstrap(secret="training-only-not-a-real-secret", log_dir=PROJECT / "training" / "logs",
          processing_dir=PROJECT / "training" / "_trace_work")

from processing.config import settings                                  # noqa: E402
from processing.selection_pipeline import choose_ocr_frames             # noqa: E402
from processing.frame_classifier import classify_frames                 # noqa: E402
from processing.frame_extractor import extract_frames_from_video        # noqa: E402
from processing.ocr_processor import chunk_batches                      # noqa: E402


def contact_sheet(paths, dest, cols=4, thumb_w=360):
    """Montage of the OCR-input frames, labelled with batch + frame name."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    if not paths:
        return None
    thumbs = []
    for label, p in paths:
        im = Image.open(p).convert("RGB")
        h = int(im.height * (thumb_w / im.width))
        im = im.resize((thumb_w, h))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, thumb_w, 22], fill=(0, 0, 0))
        d.text((5, 5), label, fill=(255, 255, 255))
        thumbs.append(im)
    rows = (len(thumbs) + cols - 1) // cols
    cell_h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (cols * thumb_w, rows * cell_h), (24, 24, 24))
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * thumb_w, (i // cols) * cell_h))
    sheet.save(dest, quality=88)
    return dest


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    copy_all = "--all" in sys.argv

    video = Path(args[0]).expanduser().resolve()
    if not video.exists():
        sys.exit(f"No such video: {video}")
    out = Path(args[1]).expanduser().resolve() if len(args) > 1 else RESULTS_ROOT / video.stem
    # A trace is evidence about one (model, video) pair. Silently overwriting an
    # earlier one — which is what happens if you re-trace with a different model —
    # destroys the comparison you were trying to make.
    if out.exists() and any(out.iterdir()) and "--force" not in sys.argv:
        model_tag = Path(os.environ.get("MODEL_DIR", "deployed")).name or "deployed"
        sys.exit(
            f"{out} already holds a trace.\n"
            f"  Pass an explicit output folder to keep both, e.g.\n"
            f"    python {Path(__file__).name} {video.name!r} "
            f"'{RESULTS_ROOT / f'{video.stem}__{model_tag}'}'\n"
            f"  or re-run with --force to overwrite it."
        )
    out.mkdir(parents=True, exist_ok=True)

    work = settings.processing_dir / f"trace_{video.stem}"
    frames_dir = work / "frames"

    print(f"\n=== 1. EXTRACT  (every {settings.frame_interval}rd frame, 720p={settings.convert_to_720p})")
    count, frames = extract_frames_from_video(video, frames_dir, job_id="trace")
    print(f"    extracted {count} frames -> {frames_dir}")
    if not frames:
        sys.exit("No frames extracted; is this a readable video?")

    print(f"\n=== 2. CLASSIFY (MobileNetV2, threshold={settings.classifier_threshold})")
    cls = classify_frames(frames, output_dir=work, job_id="trace")
    if cls.get("error"):
        sys.exit(f"Classifier unavailable: {cls['error']}")
    good = cls["good_paths"]
    conf = {Path(f["path"]).name: f["confidence"] for f in cls["good_frames"] + cls["bad_frames"]}
    print(f"    good={len(good)}  bad={len(cls['bad_frames'])}  failed={len(cls['failed_frames'])}")

    print(f"\n=== 3. SELECT   (frame_selection={settings.frame_selection})")
    scores = {Path(f["path"]): f["confidence"] for f in cls["good_frames"]}
    choice = choose_ocr_frames([frames], good, scores, job_id="trace")   # exactly as tasks.run_job
    sel, coverage = choice.selection, choice.coverage
    print(f"    frame selection kept={len(sel.selected)}  dropped={len(sel.duplicates)}")
    if coverage is not None and coverage.applied:
        reasons = {}
        for p in sel.selected:
            reasons[coverage.reasons[p]] = reasons.get(coverage.reasons[p], 0) + 1
        print(f"\n=== 3b. COVERAGE (Tesseract {settings.coverage_ocr_lang}) + duplicate pass")
        print(f"    kept={len(choice.final)}  reasons={reasons}")
    elif coverage is not None:
        print(f"    coverage SKIPPED: {coverage.reason} — the pipeline would send the frames above")
    unique, dupes = choice.final, choice.not_sent

    print(f"\n=== 4. BATCH    (ocr_batch_size={settings.ocr_batch_size})")
    batches = chunk_batches(unique)
    print(f"    {len(unique)} frames -> {len(batches)} OpenRouter request(s)")

    # ---- report ----
    ocr_dir = out / "ocr_input"
    if ocr_dir.exists():
        shutil.rmtree(ocr_dir)
    ocr_dir.mkdir(parents=True)

    labelled, manifest = [], []
    print(f"\n{'='*64}\nFRAMES THAT WOULD REACH OPENROUTER\n{'='*64}")
    for bi, batch in enumerate(batches):
        print(f"\n--- request {bi}  ({len(batch)} images in one call) ---")
        entry = {"batch_index": bi, "frames": []}
        for pos, p in enumerate(batch):
            label = f"b{bi:02d}.{pos:02d}  {p.name}"
            print(f"    [{pos:2d}] {p.name}   classifier_confidence={conf.get(p.name, 0):.4f}")
            dest = ocr_dir / f"b{bi:02d}_{pos:02d}_{p.name}"
            shutil.copy2(p, dest)
            labelled.append((label, dest))
            entry["frames"].append(
                {"batch_position": pos, "frame": p.name,
                 "classifier_confidence": round(float(conf.get(p.name, 0)), 6),
                 "saved_as": dest.name}
            )
        manifest.append(entry)

    if copy_all:
        for name, group in (("rejected_by_classifier", cls["bad_paths"]), ("dropped_by_selection", dupes)):
            d = out / name
            d.mkdir(exist_ok=True)
            for p in group:
                shutil.copy2(p, d / p.name)
            print(f"    copied {len(group):4d} frames -> {d}")

    # Frame geometry: the pipeline's scale=-2:720 caps HEIGHT, so a portrait
    # phone recording loses width. Record both so the shrink is visible.
    def _dims(p):
        try:
            from PIL import Image
            with Image.open(p) as im:
                return list(im.size)
        except Exception:
            return None

    report = {
        "video": str(video),
        "frame_size_after_extraction": _dims(frames[0]),
        "settings": {
            "frame_interval": settings.frame_interval,
            "convert_to_720p": settings.convert_to_720p,
            "classifier_threshold": settings.classifier_threshold,
            "frame_selection": settings.frame_selection,
            "coverage_selection": settings.coverage_selection,
            "coverage_applied": bool(coverage and coverage.applied),
            "phash_distance": settings.perceptual_hash_distance,
            "ocr_batch_size": settings.ocr_batch_size,
            "model_that_would_be_called": settings.openrouter_model,
        },
        "counts": {
            "extracted": len(frames),
            "classified_good": len(good),
            "classified_bad": len(cls["bad_frames"]),
            "classifier_failed": len(cls["failed_frames"]),
            "sent_after_selection": len(unique),
            "dropped_by_selection": len(dupes),
            "openrouter_requests": len(batches),
        },
        "ocr_input_batches": manifest,
        "dropped_by_selection": [p.name for p in dupes],
        "coverage_reasons": ({p.name: r for p, r in coverage.reasons.items()}
                             if coverage and coverage.applied else None),
        "rejected_by_classifier": [
            {"frame": Path(f["path"]).name, "confidence": round(float(f["confidence"]), 6)}
            for f in cls["bad_frames"]
        ],
    }
    (out / "frame_trace.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    sheet = contact_sheet(labelled, out / "contact_sheet.jpg")

    print(f"\n{'='*64}")
    print(f"extracted {len(frames)} -> classifier kept {len(good)} -> selection kept {len(unique)}"
          f" -> {len(batches)} OpenRouter request(s)")
    print(f"  frames   : {ocr_dir}")
    print(f"  manifest : {out / 'frame_trace.json'}")
    if sheet:
        print(f"  montage  : {sheet}")
    print(f"  raw frames kept at: {frames_dir}")


if __name__ == "__main__":
    main()
