#!/usr/bin/env python3
"""
SUPERSEDED — kept unchanged as the recipe for this folder's v1 results.

This script reviews the pipeline as it was on 2026-09-14: classifier run
2026-09-14-nodedup at threshold 0.95, then phash dedup (16/16). Production no
longer works that way (scroll-aware selection + Tesseract metric coverage, a
retrained model at 0.30), and its settings check reads today's config.py, so
running it now reproduces neither v1 nor production. For the current pipeline
use test-frame-output-v3/_generate_review.py (or training/trace_frames.py).

Per-frame review of the local pre-OCR pipeline: extract -> classify -> dedup -> batch.

NO OpenRouter call, no OCR, no API key. Uses the app's own modules and the
effective settings from instagram_analyzer_app/processing/config.py (refuses to
run if anything overrides them), so the result is what production would do.

For every video under test-frame/ it writes test-frame-output/<same relative
path>/<video stem>/ containing:

    frames/        every extracted frame (exactly what the classifier judged)
    kept/          the unique frames that would be sent onward, named b<batch>_<pos>__<frame>
    rejected/      frames the classifier scored below the threshold
    groups/        one folder per dedup group: the kept representative + the frames merged into it
    frames.csv     one row per frame: score, label, phash, dedup role, group, distance, batch
    groups.json    dedup groups with member scores and hash distances
    summary.json   settings, model identity, counts
    review.html    visual page: kept frames, dedup groups, rejected frames, timeline

Usage:
    .venv-train/bin/python test-frame-output/_generate_review.py [--force]
"""

import csv
import hashlib
import html
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "instagram_analyzer_app"
SRC_ROOT = REPO / "test-frame"
OUT_ROOT = REPO / "test-frame-output"
WORK = Path(os.environ.get("REVIEW_WORK_DIR", "/tmp/frame-review-work"))
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}

os.environ.setdefault("SECRET_KEY", "review-only-not-a-real-secret")
os.environ.setdefault("LOG_DIR", str(WORK / "logs"))
os.environ.setdefault("PROCESSING_DIR", str(WORK / "processing"))
sys.path.insert(0, str(APP))

from processing.config import Settings, settings  # noqa: E402
from processing import dedup  # noqa: E402
from processing.frame_classifier import classify_frames  # noqa: E402
from processing.frame_extractor import extract_frames_from_video  # noqa: E402
from processing.ocr_processor import chunk_batches  # noqa: E402

CHECKED = (
    "classifier_threshold", "classifier_image_size", "dark_frame_threshold",
    "frame_interval", "convert_to_720p", "perceptual_hash_size", "perceptual_hash_distance",
    "ocr_batch_size",
)


def effective_settings():
    """The settings actually in force, and a hard stop if any differ from config.py."""
    eff = {k: getattr(settings, k) for k in CHECKED}
    defaults = {k: Settings.model_fields[k].default for k in CHECKED}
    drift = {k: (eff[k], defaults[k]) for k in CHECKED if eff[k] != defaults[k]}
    if drift:
        sys.exit(f"refusing: settings differ from config.py defaults (env/.env override?): {drift}")
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
    return {
        "run_id": meta.get("run_id"),
        "savedmodel_md5": h.hexdigest(),
        "train_videos": set(ds.get("train_videos", [])),
        "val_videos": set(ds.get("val_videos", [])),
    }


def group_frames(good_paths, hash_size, distance):
    """Replicates dedup.dedupe_frames exactly (greedy, frame order, a frame is a
    duplicate if within `distance` of ANY kept frame) but records who merged
    with whom. Frames whose phash fails are dropped, as in the pipeline."""
    reps = []          # [(path, hash)]
    info = {}          # name -> dict
    for p in good_paths:
        try:
            h = dedup._phash(p)
        except Exception as exc:  # noqa: BLE001
            info[p.name] = {"role": "phash_failed_dropped", "phash": "", "error": str(exc)}
            continue
        dists = [(r, h - rh) for r, rh in reps]
        within = [(r, d) for r, d in dists if d <= distance]
        if within:
            rep, d = within[0]  # first kept frame that matches, in keep order
            nearest = min(dists, key=lambda x: x[1])
            info[p.name] = {
                "role": "merged_duplicate", "phash": str(h), "representative": rep.name,
                "distance": int(d), "reps_within_distance": [r.name for r, _ in within],
                "nearest_representative": nearest[0].name, "nearest_distance": int(nearest[1]),
            }
        else:
            reps.append((p, h))
            nearest = min(dists, key=lambda x: x[1]) if dists else None
            info[p.name] = {
                "role": "kept_unique", "phash": str(h), "representative": p.name, "distance": 0,
                # how close it came to being merged — useful when judging the 16/16 setting
                "nearest_other_representative": nearest[0].name if nearest else "",
                "nearest_other_distance": int(nearest[1]) if nearest else "",
            }
    return [p for p, _ in reps], info


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def review_video(video, eff, model):
    rel_parent = video.parent.relative_to(SRC_ROOT)
    out = OUT_ROOT / rel_parent / video.stem
    if out.exists() and any(out.iterdir()):
        if "--force" not in sys.argv:
            sys.exit(f"{out} already exists; pass --force to regenerate")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    print(f"\n=== {video.relative_to(REPO)}")

    # 1. extract — straight into the output so the review holds every frame judged
    count, frames = extract_frames_from_video(video, out / "frames", job_id=video.stem)
    meta = json.loads((out / "frames" / "extraction_metadata.json").read_text())
    fps = meta.get("fps") or 0
    print(f"    extracted {count}")

    # 2. classify
    cls = classify_frames(frames, output_dir=WORK / "cls" / video.stem, job_id=video.stem)
    if cls.get("error"):
        sys.exit(f"classifier error: {cls['error']}")
    score = {}
    label = {}
    for key, lab in (("good_frames", "GOOD"), ("bad_frames", "BAD"), ("failed_frames", "FAILED")):
        for f in cls[key]:
            n = Path(f["path"]).name
            score[n] = f["confidence"]
            label[n] = lab
    good = cls["good_paths"]
    print(f"    classifier kept {len(good)} / rejected {len(cls['bad_frames'])} / failed {len(cls['failed_frames'])}")

    # 3. dedup, with grouping — then prove it matches the pipeline's own function
    unique, dinfo = group_frames(good, eff["perceptual_hash_size"], eff["perceptual_hash_distance"])
    ref_unique, ref_dupes = dedup.dedupe_frames(good, eff["perceptual_hash_distance"])
    if [p.name for p in unique] != [p.name for p in ref_unique]:
        sys.exit("grouping does not reproduce dedup.dedupe_frames — aborting")
    print(f"    dedup kept {len(unique)} unique, merged {len(ref_dupes)} (verified against dedupe_frames)")

    # 4. batch
    batches = chunk_batches(unique)
    batch_of = {p.name: (bi, pos) for bi, b in enumerate(batches) for pos, p in enumerate(b)}
    print(f"    -> {len(batches)} OpenRouter request(s) would be made (none were)")

    groups = []
    for gi, rep in enumerate(unique):
        members = [rep] + [p for p in good if dinfo.get(p.name, {}).get("role") == "merged_duplicate"
                           and dinfo[p.name]["representative"] == rep.name]
        groups.append({
            "group_id": gi,
            "representative": rep.name,
            "batch": batch_of[rep.name][0],
            "batch_position": batch_of[rep.name][1],
            "size": len(members),
            "members": [{
                "frame": m.name,
                "classifier_score": round(score[m.name], 6),
                "distance_to_representative": dinfo[m.name]["distance"],
                "also_within_distance_of": [r for r in dinfo[m.name].get("reps_within_distance", [])
                                            if r != rep.name],
            } for m in members],
        })
    group_id = {m["frame"]: g["group_id"] for g in groups for m in g["members"]}

    # rows, frame order
    rows = []
    for p in frames:
        n = p.name
        fnum = int(p.stem.split("_")[1])
        d = dinfo.get(n, {})
        if label[n] == "BAD":
            role = "rejected_by_classifier"
        elif label[n] == "FAILED":
            role = "classifier_failed"
        else:
            role = d.get("role", "")
        bi, pos = batch_of.get(n, ("", ""))
        rows.append({
            "frame": n,
            "frame_number": fnum,
            "time_s": round(fnum / fps, 2) if fps else "",
            "classifier_score": round(score[n], 6),
            "classifier_label": label[n],
            "threshold": eff["classifier_threshold"],
            "dedup_role": role,
            "group_id": group_id.get(n, ""),
            "group_representative": d.get("representative", "") if role in ("kept_unique", "merged_duplicate") else "",
            "distance_to_representative": d.get("distance", "") if role in ("kept_unique", "merged_duplicate") else "",
            "reps_within_distance": ";".join(d.get("reps_within_distance", [])),
            "nearest_other_rep_distance": d.get("nearest_other_distance", ""),
            "sent_to_openrouter": int(n in batch_of),
            "batch_index": bi,
            "batch_position": pos,
            "phash": d.get("phash", ""),
        })

    with open(out / "frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "groups.json").write_text(json.dumps(groups, indent=1), encoding="utf-8")

    # image folders for manual review
    for p in unique:
        bi, pos = batch_of[p.name]
        copy(p, out / "kept" / f"b{bi:02d}_{pos:02d}__{p.name}")
    for f in cls["bad_frames"]:
        n = Path(f["path"]).name
        copy(Path(f["path"]), out / "rejected" / f"{score[n]:.4f}__{n}")
    for g in groups:
        gdir = out / "groups" / f"g{g['group_id']:03d}__{g['representative'][:-4]}__x{g['size']}"
        for m in g["members"]:
            tag = "REP" if m["frame"] == g["representative"] else f"d{m['distance_to_representative']:02d}"
            copy(out / "frames" / m["frame"], gdir / f"{tag}__{m['frame']}")

    in_train = video.stem in model["train_videos"]
    in_val = video.stem in model["val_videos"]
    counts = {
        "extracted": len(frames),
        "classifier_good": len(good),
        "classifier_bad": len(cls["bad_frames"]),
        "classifier_failed": len(cls["failed_frames"]),
        "phash_failed_dropped": sum(1 for d in dinfo.values() if d["role"] == "phash_failed_dropped"),
        "unique_after_dedup": len(unique),
        "merged_as_duplicate": len(ref_dupes),
        "dedup_groups": len(groups),
        "openrouter_requests_that_would_be_made": len(batches),
    }
    summary = {
        "video": str(video.relative_to(REPO)),
        "frame_size_after_extraction": list(__import__("cv2").imread(str(frames[0])).shape[1::-1]),
        "fps": fps,
        "settings": eff,
        "model": {"run_id": model["run_id"], "savedmodel_md5": model["savedmodel_md5"]},
        "video_in_model_training_set": in_train,
        "video_in_model_validation_set": in_val,
        "counts": counts,
        "openrouter_called": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_html(out, video, summary, rows, groups, batches)
    return summary


def write_html(out, video, summary, rows, groups, batches):
    e = html.escape
    c = summary["counts"]
    s = summary["settings"]
    by = {r["frame"]: r for r in rows}

    def thumb(n, caption, cls=""):
        return (f'<figure class="{cls}"><a href="frames/{e(n)}" target="_blank">'
                f'<img loading="lazy" src="frames/{e(n)}"></a><figcaption>{caption}</figcaption></figure>')

    provenance = ("TRAINING SET — this video was used to train the model; results here do not measure generalisation"
                  if summary["video_in_model_training_set"] else
                  "validation set" if summary["video_in_model_validation_set"] else
                  "unseen — not in this model's training or validation set")
    parts = [f"""<!doctype html><meta charset="utf-8"><title>{e(video.stem)} — frame review</title>
<style>
body{{font:14px -apple-system,system-ui,sans-serif;margin:24px;background:#f6f6f4;color:#222}}
h1{{font-size:20px}} h2{{margin-top:36px;border-bottom:1px solid #ccc;padding-bottom:4px}}
.meta{{background:#fff;padding:12px 16px;border-radius:6px;line-height:1.7}}
.warn{{color:#a4001d;font-weight:600}} .ok{{color:#1d6b2f;font-weight:600}}
.grid{{display:flex;flex-wrap:wrap;gap:8px}}
figure{{margin:0;width:150px;background:#fff;border:3px solid #ddd;border-radius:4px}}
figure img{{width:100%;display:block}} figcaption{{font:11px ui-monospace,monospace;padding:3px 4px}}
figure.rep{{border-color:#1d6b2f}} figure.dup{{border-color:#b0b0b0;opacity:.85}}
figure.bad{{border-color:#c8323c}} figure.good{{border-color:#1d6b2f}}
.group{{background:#fff;padding:8px;margin:10px 0;border-radius:6px}}
.group h3{{margin:0 0 6px;font-size:13px}}
.strip{{display:flex;flex-wrap:wrap;gap:2px}} .strip a{{width:10px;height:22px;display:block}}
.k{{background:#1d6b2f}} .m{{background:#8fc79a}} .r{{background:#c8323c}} .f{{background:#000}}
</style>
<h1>{e(video.name)}</h1>
<div class="meta">
Model <b>{e(str(summary['model']['run_id']))}</b> · threshold <b>{s['classifier_threshold']}</b> ·
phash <b>{s['perceptual_hash_size']}/{s['perceptual_hash_distance']}</b> · every {s['frame_interval']}rd frame ·
720p={s['convert_to_720p']} · frame size {summary['frame_size_after_extraction']} · batch size {s['ocr_batch_size']}<br>
Video provenance: <span class="{'warn' if summary['video_in_model_training_set'] else 'ok'}">{e(provenance)}</span><br>
<b>{c['extracted']}</b> extracted → classifier kept <b>{c['classifier_good']}</b> (rejected {c['classifier_bad']}) →
dedup kept <b>{c['unique_after_dedup']}</b> in {c['dedup_groups']} groups (merged {c['merged_as_duplicate']}) →
<b>{c['openrouter_requests_that_would_be_made']}</b> OpenRouter request(s) would be made. <i>None were.</i>
</div>
<h2>Timeline</h2>
<p>One cell per extracted frame, in order. <span class="k">&nbsp;&nbsp;&nbsp;</span> kept/sent ·
<span class="m">&nbsp;&nbsp;&nbsp;</span> passed classifier, merged by dedup ·
<span class="r">&nbsp;&nbsp;&nbsp;</span> rejected by classifier. Hover for details.</p>
<div class="strip">"""]
    cmap = {"kept_unique": "k", "merged_duplicate": "m", "rejected_by_classifier": "r"}
    for r in rows:
        parts.append(f'<a class="{cmap.get(r["dedup_role"], "f")}" href="frames/{e(r["frame"])}" target="_blank" '
                     f'title="{e(r["frame"])} t={r["time_s"]}s score={r["classifier_score"]:.4f} {e(r["dedup_role"])}"></a>')
    parts.append("</div>")

    parts.append(f"<h2>Sent onward — {c['unique_after_dedup']} unique frames in "
                 f"{c['openrouter_requests_that_would_be_made']} request(s)</h2>")
    for bi, b in enumerate(batches):
        parts.append(f"<h3>Request {bi} ({len(b)} images)</h3><div class='grid'>")
        for pos, p in enumerate(b):
            r = by[p.name]
            g = next(g for g in groups if g["representative"] == p.name)
            parts.append(thumb(p.name, f"b{bi}.{pos} {p.name[6:-4]}<br>score {r['classifier_score']:.4f}<br>"
                                       f"group of {g['size']}", "rep"))
        parts.append("</div>")

    parts.append("<h2>Dedup groups</h2><p>Green border = the representative that is sent; grey = merged into it "
                 "(d = Hamming distance to the representative, merge if ≤ "
                 f"{s['perceptual_hash_distance']}). Check that merged frames show the same numbers as their "
                 "representative.</p>")
    for g in groups:
        parts.append(f"<div class='group'><h3>g{g['group_id']:03d} · rep {e(g['representative'])} · "
                     f"{g['size']} frame(s) · request {g['batch']}.{g['batch_position']}</h3><div class='grid'>")
        for m in g["members"]:
            rep = m["frame"] == g["representative"]
            extra = f"<br>also ≤{s['perceptual_hash_distance']} of {len(m['also_within_distance_of'])} other" \
                if m["also_within_distance_of"] else ""
            parts.append(thumb(m["frame"], f"{m['frame'][6:-4]} {'REP' if rep else 'd=' + str(m['distance_to_representative'])}"
                                           f"<br>score {m['classifier_score']:.4f}{extra}", "rep" if rep else "dup"))
        parts.append("</div></div>")

    rejected = [r for r in rows if r["dedup_role"] == "rejected_by_classifier"]
    parts.append(f"<h2>Rejected by classifier — {len(rejected)} frames (score ≤ {s['classifier_threshold']})</h2>"
                 "<p>Check none of these carry a metric you need. Sorted by score, highest first.</p><div class='grid'>")
    for r in sorted(rejected, key=lambda r: -r["classifier_score"]):
        parts.append(thumb(r["frame"], f"{r['frame'][6:-4]}<br>score {r['classifier_score']:.4f}", "bad"))
    parts.append("</div>")
    (out / "review.html").write_text("\n".join(parts), encoding="utf-8")


def main():
    eff = effective_settings()
    model = model_identity()
    print(f"settings: {eff}")
    print(f"model: {model['run_id']} (savedmodel md5 {model['savedmodel_md5']})")
    videos = sorted(p for p in SRC_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        sys.exit(f"no videos under {SRC_ROOT}")
    OUT_ROOT.mkdir(exist_ok=True)
    summaries = [review_video(v, eff, model) for v in videos]
    (OUT_ROOT / "index.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("\n" + "=" * 70)
    for s in summaries:
        c = s["counts"]
        tag = "TRAIN-SET" if s["video_in_model_training_set"] else "unseen"
        print(f"{s['video']:<60} [{tag}] {c['extracted']} -> {c['classifier_good']} -> "
              f"{c['unique_after_dedup']} -> {c['openrouter_requests_that_would_be_made']} req")


if __name__ == "__main__":
    main()
