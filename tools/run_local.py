#!/usr/bin/env python
"""Run the pipeline on a folder of videos in-process: no web app, no Postgres,
no Celery. One metrics JSON per video, the same document the app stores.

    OPENROUTER_API_KEY=... tools/run_local.py --input videos/ --out results/ [--only STEM ...]

Mirrors tasks.run_job -> ocr_batch -> finalize_job exactly, sequentially:
extract -> classify -> choose_ocr_frames -> chunk_batches -> ocr_single_batch
-> assemble_metrics -> <out>/<stem>.json (+ a `_run` block: frame counts,
frames sent, model, elapsed). Needs ffmpeg, Tesseract with fas+eng and
TensorFlow on this machine; frame selection can differ slightly from the
Docker run when the host Tesseract version differs from the image's.

Never overwrites an existing <stem>.json; stops after --stop-after-failures
consecutive failures so a dead route does not burn through a whole folder.
This DOES call OpenRouter (about $0.01 per video with the default model).
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
APP = REPO / "instagram_analyzer_app"
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}


def bootstrap(work: Path) -> None:
    """The app's Settings are built at import; set the environment first."""
    os.environ.setdefault("SECRET_KEY", "local-run-not-a-real-secret-key")
    os.environ.pop("DATABASE_URL", None)
    os.environ["LOG_DIR"] = str(work / "logs")
    os.environ["PROCESSING_DIR"] = str(work / "processing")
    os.environ["UPLOAD_DIR"] = str(work / "uploads")
    for d in ("logs", "processing", "uploads"):
        (work / d).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(APP))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="folder of videos, or one video")
    ap.add_argument("--out", required=True, type=Path, help="one <stem>.json per video is written here")
    ap.add_argument("--only", action="append", default=[], metavar="STEM", help="run only these videos (repeatable)")
    ap.add_argument("--work", type=Path, default=None, help="scratch dir for frames/logs (default: <out>/_work)")
    ap.add_argument("--stop-after-failures", type=int, default=2)
    a = ap.parse_args(argv)
    if not os.environ.get("OPENROUTER_API_KEY"):
        ap.error("OPENROUTER_API_KEY is not set (this tool sends frames to OpenRouter)")

    work = a.work or (a.out / "_work")
    a.out.mkdir(parents=True, exist_ok=True)
    bootstrap(work)

    from processing.config import settings
    from processing.frame_extractor import extract_frames_from_video, VideoUnreadable
    from processing.frame_classifier import classify_frames
    from processing.selection_pipeline import choose_ocr_frames
    from processing.metric_coverage import tesseract_available
    from processing.ocr_processor import assemble_metrics, chunk_batches, frames_missing, ocr_single_batch

    meta = settings.model_dir / "model_metadata.json"
    print(f"model        : {settings.openrouter_model} (reasoning effort: {settings.openrouter_reasoning_effort or 'not sent'})")
    print(f"classifier   : run {json.loads(meta.read_text()).get('run_id') if meta.exists() else '?'} at threshold {settings.classifier_threshold}")
    print(f"selection    : {settings.frame_selection}, coverage={settings.coverage_selection}, tesseract={tesseract_available()}")
    print(f"output       : {a.out}", flush=True)

    src = a.input
    videos = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if a.only:
        videos = [v for v in videos if v.stem in a.only]
    if not videos:
        raise SystemExit("no videos to run")

    def run_one(video: Path) -> dict:
        stem, started = video.stem, time.time()
        job_dir = work / "processing" / stem
        job_dir.mkdir(parents=True, exist_ok=True)
        count, frame_paths = extract_frames_from_video(video, job_dir / "frames", job_id=stem)
        classification = classify_frames(frame_paths, output_dir=job_dir, job_id=stem)
        if classification.get("error"):
            raise RuntimeError(f"Frame classifier unavailable: {classification['error']}")
        good_paths = classification.get("good_paths", [])
        scores = {Path(f["path"]): f["confidence"] for f in classification.get("good_frames", [])}
        choice = choose_ocr_frames([frame_paths], good_paths, scores, job_id=stem)
        final = choice.final
        print(f"  extracted {count}, good {len(good_paths)}, scroll {len(choice.selection.selected)} -> sent {len(final)}", flush=True)

        batches = chunk_batches(final)
        all_results, batches_failed = [], 0
        for i, batch in enumerate(batches):
            print(f"  OCR batch {i + 1}/{len(batches)} ({len(batch)} frames)...", flush=True)
            res = ocr_single_batch(batch, stem)
            if res is None:
                batches_failed += 1
                continue
            all_results.extend(res)
        missing = 0 if batches_failed else sum(frames_missing(b, all_results) for b in batches)
        metrics = assemble_metrics(all_results, total_frames=len(good_paths), unique_frames=len(final),
                                   duplicate_frames=len(choice.not_sent), batches_total=len(batches),
                                   batches_failed=batches_failed, frames_missing=missing)
        metrics["_run"] = {
            "video": video.name, "extracted_frames": count, "classifier_good_frames": len(good_paths),
            "scroll_selected_frames": len(choice.selection.selected), "frames_sent_to_ocr": len(final),
            "frames_sent": [p.name for p in final],
            "coverage_applied": bool(choice.coverage and choice.coverage.applied),
            "elapsed_seconds": round(time.time() - started, 1), "model": settings.openrouter_model,
            "reasoning_effort": settings.openrouter_reasoning_effort, "frames_missing": missing,
        }
        out_path = a.out / f"{stem}.json"
        out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"video": video.name, "status": "failed" if (batches_failed or missing) else "ok",
                "batches_failed": batches_failed, "frames_missing": missing, "batches_total": len(batches),
                "sent": len(final), "summary_fields": len(metrics.get("summary", {})),
                "output": str(out_path), "elapsed": round(time.time() - started, 1)}

    report, consecutive = [], 0
    for n, video in enumerate(videos, 1):
        if (a.out / f"{video.stem}.json").exists():
            print(f"[{n}/{len(videos)}] {video.name}  -> skipped, output exists", flush=True)
            continue
        if consecutive >= a.stop_after_failures:
            print(f"[{n}/{len(videos)}] {video.name}  -> NOT RUN: {consecutive} consecutive failures", flush=True)
            report.append({"video": video.name, "status": "not_run"})
            continue
        print(f"[{n}/{len(videos)}] {video.name}", flush=True)
        try:
            row = run_one(video)
        except VideoUnreadable as exc:
            row = {"video": video.name, "status": "error", "error": f"VideoUnreadable: {exc}"}
        except Exception as exc:  # noqa: BLE001 — one bad video must not stop the run
            row = {"video": video.name, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()[-1500:]}
        consecutive = 0 if row["status"] == "ok" else consecutive + 1
        report.append(row)
        print(f"  -> {row['status']}" + (f"  ({row.get('summary_fields')} metrics, {row.get('elapsed')}s)"
                                         if row["status"] != "error" else f"  {row.get('error')}"), flush=True)
    (a.out / "_run_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    ok = sum(1 for r in report if r["status"] == "ok")
    print(f"DONE ok={ok}  partial={sum(1 for r in report if r['status'] == 'failed')}  error={sum(1 for r in report if r['status'] == 'error')}")
    return 0 if ok == len(report) else 1


if __name__ == "__main__":
    sys.exit(main())
