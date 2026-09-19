#!/usr/bin/env python3
"""Review round: run videos through the local pre-OCR pipeline, as deployed, and
lay out per video what would be sent to OpenRouter, for human feedback.

    extract -> classify (deployed model, config threshold)
            -> scroll-aware selection (frame_selection.py)
            -> metric-coverage selection via Tesseract (metric_coverage.py)
            -> batch

NO OpenRouter call, no vision-model OCR, no API key. Uses the app's own modules
and the effective settings from processing/config.py; refuses to run if anything
overrides them or if Tesseract with the configured languages is missing, so a
silently skipped step cannot pass for a result.

Never overwrites: every round goes to a new --out folder, and a video whose
output folder already exists is refused.

Per video, <out>/<video stem>/:
    review.html           start here: timeline, sent frames + metrics read, what coverage dropped and why
    final/                frames that would be sent to OCR, named b<request>_<pos>__<frame>
    scroll_selected/      the coverage step's input
    dropped_by_coverage/  named <reason>__<frame>
    frames/               every extracted frame
    frames.csv            one row per frame: score, label, stage, coverage reason, metrics read
    summary.json          settings, model md5, video md5, counts
    feedback.json         YOUR review of this video (template; see the round README)
and <out>/index.html + index.json for the whole round.

Usage:
    .venv-train/bin/python training/review_videos.py VIDEO_OR_DIR [...] --out review-rounds/<round>
"""

import argparse
import csv
import hashlib
import html
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "instagram_analyzer_app"
WORK = Path(os.environ.get("REVIEW_WORK_DIR", "/tmp/frame-review-work"))
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}

os.environ.setdefault("SECRET_KEY", "review-only-not-a-real-secret")
os.environ.setdefault("LOG_DIR", str(WORK / "logs"))
os.environ.setdefault("PROCESSING_DIR", str(WORK / "processing"))
sys.path.insert(0, str(APP))

import cv2  # noqa: E402

from processing.config import Settings, settings  # noqa: E402
from processing.frame_classifier import classify_frames  # noqa: E402
from processing.frame_extractor import extract_frames_from_video  # noqa: E402
from processing.metric_coverage import read_lines, tesseract_available  # noqa: E402
from processing.selection_pipeline import choose_ocr_frames  # noqa: E402
from processing.ocr_processor import chunk_batches  # noqa: E402


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


CHECKED = (
    "classifier_threshold", "classifier_image_size", "dark_frame_threshold",
    "frame_interval", "convert_to_720p", "perceptual_hash_size", "perceptual_hash_distance",
    "frame_selection", "coverage_selection", "coverage_ocr_lang", "ocr_batch_size",
)
REASON_TEXT = {
    "cover": "needed to cover a metric",
    "unrecognised": "kept: no metric read, but a number on screen",
    "backstop": "kept: shows a labelled value no other kept frame shows",
    "covered": "dropped: its metrics are in kept frames",
    "duplicate": "dropped: all its content is readable in another sent frame",
    "blank": "dropped: no metric and no number",
}


def effective_settings():
    eff = {k: getattr(settings, k) for k in CHECKED}
    defaults = {k: Settings.model_fields[k].default for k in CHECKED}
    drift = {k: (eff[k], defaults[k]) for k in CHECKED if eff[k] != defaults[k]}
    if drift:
        sys.exit(f"refusing: settings differ from config.py defaults (env/.env override?): {drift}")
    if not (settings.frame_selection == "scroll" and settings.coverage_selection):
        sys.exit("refusing: this review is of the scroll + coverage pipeline")
    if not tesseract_available():
        sys.exit(f"refusing: tesseract with '{settings.coverage_ocr_lang}' is not available")
    return eff


def model_identity():
    mdir = settings.model_dir / "frame_classifier_savedmodel"
    meta = json.loads((settings.model_dir / "model_metadata.json").read_text())
    h = hashlib.md5()
    for f in sorted(mdir.rglob("*")):
        if f.is_file():
            h.update(f.relative_to(mdir).as_posix().encode())
            h.update(f.read_bytes())
    ds = meta.get("dataset", {})
    return {"run_id": meta.get("run_id"), "savedmodel_md5": h.hexdigest(),
            "train_videos": set(ds.get("train_videos", [])), "val_videos": set(ds.get("val_videos", []))}


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def review_video(video, out, eff, model):
    out.mkdir(parents=True)          # main() has checked it does not exist; never overwrite
    print(f"\n=== {_rel(video)}")

    # 1. extract
    count, frames = extract_frames_from_video(video, out / "frames", job_id=video.stem)
    fps = json.loads((out / "frames" / "extraction_metadata.json").read_text()).get("fps") or 0
    print(f"    extracted {count}")

    # 2. classify
    cls = classify_frames(frames, output_dir=WORK / "cls" / video.stem, job_id=video.stem)
    if cls.get("error"):
        sys.exit(f"classifier error: {cls['error']}")
    score, label = {}, {}
    for key, lab in (("good_frames", "GOOD"), ("bad_frames", "BAD"), ("failed_frames", "FAILED")):
        for f in cls[key]:
            score[Path(f["path"]).name] = f["confidence"]
            label[Path(f["path"]).name] = lab
    good = cls["good_paths"]
    print(f"    classifier kept {len(good)} / rejected {len(cls['bad_frames'])}")

    # 3-4. frame selection, metric coverage, duplicate pass: the production call
    scores = {Path(f["path"]): f["confidence"] for f in cls["good_frames"]}
    choice = choose_ocr_frames([frames], good, scores, job_id=video.stem)
    sel, cov, final = choice.selection, choice.coverage, choice.final
    if cov is None or not cov.applied:
        sys.exit(f"coverage step did not run: {cov.reason if cov else 'no frames'}")
    cov.dropped = [p for p in sel.selected if p not in set(final)]
    stands_for = {m.name: rep.name for rep, members in sel.groups.items() for m in members}
    print(f"    scroll selection kept {len(sel.selected)} ({sel.pages} page(s))")
    print(f"    coverage + duplicate pass kept {len(final)} -> these would be sent to OCR "
          f"({len(choice.dropped_as_duplicate)} dropped as duplicate content)")

    # OCR lines again, for display only (the selection above already happened)
    lines = {p.name: read_lines(p) for p in sel.selected}

    # 5. batch
    batches = chunk_batches(final)
    batch_of = {p.name: (bi, pos) for bi, b in enumerate(batches) for pos, p in enumerate(b)}
    print(f"    -> {len(batches)} OpenRouter request(s) would be made (none were)")

    selected_names = {p.name for p in sel.selected}
    rows = []
    for p in frames:
        n = p.name
        if label[n] != "GOOD":
            stage = "rejected_by_classifier" if label[n] == "BAD" else "classifier_failed"
        elif n in batch_of:
            stage = "sent"
        elif n in selected_names:
            stage = "dropped_by_coverage"
        else:
            stage = "merged_by_scroll_selection"
        reason = cov.reasons.get(p, "") if n in selected_names else ""
        fnum = int(p.stem.split("_")[1])
        bi, pos = batch_of.get(n, ("", ""))
        rows.append({
            "frame": n, "frame_number": fnum, "time_s": round(fnum / fps, 2) if fps else "",
            "classifier_score": round(score[n], 6), "classifier_label": label[n],
            "stage": stage,
            "scroll_stands_for": stands_for.get(n, "") if n not in selected_names else "",
            "coverage_reason": reason,
            "metrics_read": ";".join(sorted(cov.metrics.get(p, set()))) if n in selected_names else "",
            "sent_to_openrouter": int(n in batch_of), "batch_index": bi, "batch_position": pos,
        })
    with open(out / "frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for p in final:
        bi, pos = batch_of[p.name]
        copy(p, out / "final" / f"b{bi:02d}_{pos:02d}__{p.name}")
    for p in sel.selected:
        copy(p, out / "scroll_selected" / p.name)
    for p in cov.dropped:
        copy(p, out / "dropped_by_coverage" / f"{cov.reasons[p]}__{p.name}")

    metrics_covered = sorted(set().union(*(cov.metrics[p] for p in final)) if final else set())
    counts = {
        "extracted": len(frames),
        "classifier_good": len(good),
        "classifier_bad": len(cls["bad_frames"]),
        "scroll_selected": len(sel.selected),
        "final_sent": len(final),
        "openrouter_requests_that_would_be_made": len(batches),
        "coverage_reasons": {r: sum(1 for p in sel.selected if cov.reasons.get(p) == r) for r in REASON_TEXT},
    }
    summary = {
        "video": _rel(video),
        "video_md5": _md5(video),
        "frame_size_after_extraction": list(cv2.imread(str(frames[0])).shape[1::-1]),
        "fps": fps,
        "settings": eff,
        "model": {"run_id": model["run_id"], "savedmodel_md5": model["savedmodel_md5"]},
        "video_in_model_training_set": video.stem in model["train_videos"],
        "video_in_model_validation_set": video.stem in model["val_videos"],
        "counts": counts,
        "metrics_read_in_final_set": metrics_covered,
        "openrouter_called": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_html(out, video, summary, rows, final, batches, cov, lines)
    write_feedback_template(out, video, summary, final, batches, cov)
    return summary


def write_html(out, video, summary, rows, final, batches, cov, lines):
    e = html.escape
    c, s = summary["counts"], summary["settings"]

    def thumb(n, caption, cls=""):
        return (f'<figure class="{cls}"><a href="frames/{e(n)}" target="_blank">'
                f'<img loading="lazy" src="frames/{e(n)}"></a><figcaption>{caption}</figcaption></figure>')

    prov = ("TRAINING SET: this video was used to train the model"
            if summary["video_in_model_training_set"] else
            "validation set" if summary["video_in_model_validation_set"] else
            "unseen: not in this model's training or validation set")
    parts = [f"""<!doctype html><meta charset="utf-8"><title>{e(video.stem)} — frame review</title>
<style>
body{{font:14px -apple-system,system-ui,sans-serif;margin:24px;background:#f6f6f4;color:#222}}
h1{{font-size:20px}} h2{{margin-top:34px;border-bottom:1px solid #ccc;padding-bottom:4px}}
.meta{{background:#fff;padding:12px 16px;border-radius:6px;line-height:1.7}}
table{{border-collapse:collapse;background:#fff;margin:10px 0}} td,th{{border:1px solid #ddd;padding:5px 10px;text-align:left}}
.warn{{color:#a4001d;font-weight:600}} .ok{{color:#1d6b2f;font-weight:600}}
.grid{{display:flex;flex-wrap:wrap;gap:8px}}
figure{{margin:0;width:170px;background:#fff;border:3px solid #ddd;border-radius:4px}}
figure img{{width:100%;display:block}} figcaption{{font:11px ui-monospace,monospace;padding:3px 4px;line-height:1.35}}
figure.sent{{border-color:#1d6b2f}} figure.drop{{border-color:#b07a00}} figure.bad{{border-color:#c8323c}}
.strip{{display:flex;flex-wrap:wrap;gap:2px}} .strip a{{width:10px;height:22px;display:block}}
.k{{background:#1d6b2f}} .d{{background:#e0a800}} .m{{background:#9dc9a6}} .r{{background:#c8323c}} .f{{background:#000}}
details pre{{white-space:pre-wrap;font:11px ui-monospace,monospace;background:#fafafa;margin:0;padding:4px}}
</style>
<h1>{e(video.name)}</h1>
<div class="meta">
Model <b>{e(str(summary['model']['run_id']))}</b> · threshold <b>{s['classifier_threshold']}</b> ·
selection <b>{s['frame_selection']}</b> + metric coverage (Tesseract {e(s['coverage_ocr_lang'])}) ·
frame size {summary['frame_size_after_extraction']}<br>
Video provenance: <span class="{'warn' if summary['video_in_model_training_set'] else 'ok'}">{e(prov)}</span><br>
<i>No OpenRouter call was made.</i>
</div>
<table><tr><th></th><th>classifier</th><th>dedup / selection</th><th>passed classifier</th>
<th>after scroll selection</th><th>sent to OCR</th><th>requests</th></tr>
<tr><th>this run</th><td>{e(str(summary['model']['run_id']))} @ {s['classifier_threshold']}</td>
<td>scroll + coverage</td><td>{c['classifier_good']}</td><td>{c['scroll_selected']}</td>
<td><b>{c['final_sent']}</b></td><td>{c['openrouter_requests_that_would_be_made']}</td></tr></table>
<p>Metrics read in the final set: <b>{e(', '.join(summary['metrics_read_in_final_set']) or 'none')}</b></p>
<div class="meta"><b>Your feedback</b> goes in <code>feedback.json</code> next to this page: for each sent
frame, whether it was needed; any frame that should have been sent (its number from the timeline or the lists
below); metrics missing from the final set. Format: the round's README.</div>

<h2>Timeline</h2>
<p><span class="k">&nbsp;&nbsp;&nbsp;</span> sent ·
<span class="d">&nbsp;&nbsp;&nbsp;</span> scroll-selected, dropped by coverage ·
<span class="m">&nbsp;&nbsp;&nbsp;</span> passed classifier, merged by scroll selection ·
<span class="r">&nbsp;&nbsp;&nbsp;</span> rejected by classifier. Hover for details.</p><div class="strip">"""]
    cmap = {"sent": "k", "dropped_by_coverage": "d", "merged_by_scroll_selection": "m", "rejected_by_classifier": "r"}
    for r in rows:
        parts.append(f'<a class="{cmap.get(r["stage"], "f")}" href="frames/{e(r["frame"])}" target="_blank" '
                     f'title="{e(r["frame"])} t={r["time_s"]}s score={r["classifier_score"]:.3f} {e(r["stage"])}"></a>')
    parts.append("</div>")

    parts.append(f"<h2>Sent to OCR: {len(final)} frame(s) in {len(batches)} request(s)</h2>"
                 "<p>With the metrics Tesseract read on each (which metric is on screen, not its value: "
                 "the vision model reads the numbers). Expand a frame to see the raw OCR text.</p><div class='grid'>")
    for bi, b in enumerate(batches):
        for pos, p in enumerate(b):
            m = ", ".join(sorted(cov.metrics[p])) or "—"
            raw = e("\n".join(lines.get(p.name, [])))
            parts.append(thumb(p.name, f"<b>b{bi}.{pos} {p.name[6:12]}</b> · {e(cov.reasons[p])}<br>{e(m)}"
                                       f"<details><summary>OCR text</summary><pre>{raw}</pre></details>", "sent"))
    parts.append("</div>")

    parts.append(f"<h2>Dropped by the coverage step: {len(cov.dropped)} frame(s)</h2>"
                 "<p>Scroll selection kept these; the coverage step decided every metric on them is "
                 "already in a sent frame. <b>Check these first</b>: a metric wrongly dropped would show up here.</p>"
                 "<div class='grid'>")
    for p in cov.dropped:
        m = ", ".join(sorted(cov.metrics[p])) or "—"
        raw = e("\n".join(lines.get(p.name, [])))
        parts.append(thumb(p.name, f"<b>{p.name[6:12]}</b> · {e(REASON_TEXT[cov.reasons[p]])}<br>{e(m)}"
                                   f"<details><summary>OCR text</summary><pre>{raw}</pre></details>", "drop"))
    parts.append("</div>")

    rejected = sorted((r for r in rows if r["stage"] == "rejected_by_classifier"), key=lambda r: -r["classifier_score"])
    parts.append(f"<h2>Rejected by the classifier: {len(rejected)} frame(s) (score ≤ {s['classifier_threshold']})</h2>"
                 "<p>Sorted by score, highest first.</p><div class='grid'>")
    for r in rejected:
        parts.append(thumb(r["frame"], f"{r['frame'][6:12]}<br>score {r['classifier_score']:.4f}", "bad"))
    parts.append("</div>")
    (out / "review.html").write_text("\n".join(parts), encoding="utf-8")


def write_feedback_template(out, video, summary, final, batches, cov):
    batch_of = {p.name: f"b{bi}.{pos}" for bi, b in enumerate(batches) for pos, p in enumerate(b)}
    doc = {
        "video": video.name,
        "reviewer": "",
        "status": "todo",
        "sent_frames": [
            {"frame": p.name, "request": batch_of[p.name], "coverage_reason": cov.reasons[p],
             "metrics_read": sorted(cov.metrics[p]),
             "needed": None,
             "note": ""}
            for p in final
        ],
        "should_have_been_sent": [],
        "metrics_missing_from_final_set": [],
        "notes": "",
    }
    (out / "feedback.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_index(out_root, summaries):
    e = html.escape
    rows = []
    for s in summaries:
        stem = Path(s["video"]).stem
        c = s["counts"]
        prov = ("TRAINING SET" if s["video_in_model_training_set"] else
                "validation" if s["video_in_model_validation_set"] else "unseen")
        rows.append(f"<tr><td><a href=\"{e(stem)}/review.html\">{e(stem)}</a></td><td>{prov}</td>"
                    f"<td>{c['extracted']}</td><td>{c['classifier_good']}</td><td>{c['scroll_selected']}</td>"
                    f"<td><b>{c['final_sent']}</b></td><td>{c['openrouter_requests_that_would_be_made']}</td>"
                    f"<td>{e(', '.join(s['metrics_read_in_final_set']) or '—')}</td></tr>")
    (out_root / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Review round</title>"
        "<style>body{font:14px -apple-system,system-ui,sans-serif;margin:24px}"
        "table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:5px 9px;text-align:left}</style>"
        f"<h1>Review round: {e(out_root.name)}</h1><p>No OpenRouter call was made.</p>"
        "<table><tr><th>video</th><th>provenance</th><th>extracted</th><th>passed classifier</th>"
        "<th>after scroll selection</th><th>sent to OCR</th><th>requests</th><th>metrics read</th></tr>"
        + "".join(rows) + "</table>", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="video files, or folders (top level only)")
    ap.add_argument("--out", type=Path, required=True, help="round folder, e.g. review-rounds/round-01")
    args = ap.parse_args()

    videos = []
    for p in args.inputs:
        if p.is_dir():
            videos += sorted(q for q in p.iterdir() if q.is_file() and q.suffix.lower() in VIDEO_EXTS)
        elif p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)
        else:
            sys.exit(f"not a video or folder: {p}")
    if not videos:
        sys.exit("no videos found")
    stems = [v.stem for v in videos]
    if len(set(stems)) != len(stems):
        sys.exit("two inputs share a file name; their outputs would collide")
    clash = [str(args.out / s) for s in stems if (args.out / s).exists()]
    if clash:
        sys.exit("refusing to overwrite existing output: " + ", ".join(clash) + "\n  use a new --out folder")

    eff = effective_settings()
    model = model_identity()
    print(f"settings: {eff}")
    print(f"model: {model['run_id']} (savedmodel md5 {model['savedmodel_md5']})")
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = [review_video(v, args.out / v.stem, eff, model) for v in videos]

    index = args.out / "index.json"
    previous = json.loads(index.read_text()) if index.exists() else []
    index.write_text(json.dumps(previous + [{**s, "reviewed_at": datetime.now().isoformat(timespec="seconds")}
                                            for s in summaries], indent=2), encoding="utf-8")
    write_index(args.out, previous + summaries)
    print("\n" + "=" * 78)
    for s in summaries:
        c = s["counts"]
        print(f"{Path(s['video']).stem:<40} {c['extracted']:>4} -> {c['classifier_good']:>3} -> "
              f"{c['scroll_selected']:>3} -> {c['final_sent']:>2} sent ({c['openrouter_requests_that_would_be_made']} req)")
    print(f"\nopen {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
