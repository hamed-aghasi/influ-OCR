#!/usr/bin/env python3
"""Extract raw frames from training videos, for manual sorting.

Drop videos into ``train/``, run this, and each one gets its own folder holding
every extracted frame plus empty ``good/`` and ``bad/`` folders to sort them
into::

    train/
      my-video.mp4
      my-video/
        frames/          every extracted frame
        good/            <- you move frames here
        bad/             <- you move frames here
        _shortlist/      symlinks: one representative per near-duplicate cluster
        clusters.json    representative -> the frames it stands in for
        video_info.json  resolution, fps, theme, extraction settings

**No classifier is loaded or run.** This stage is deliberately dumb: it
reproduces exactly what the production pipeline feeds its classifier, so the
labels you apply describe the same frames the app will have to judge.

Extraction is the app's own ``extract_frames_from_video`` — ffmpeg 720p
conversion then every Nth frame — so nothing here can drift from production.

Usage::

    python training/extract_frames.py                 # every unprocessed video
    python training/extract_frames.py --force         # re-extract everything
    python training/extract_frames.py --no-shortlist  # skip the dedup shortlist
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "instagram_analyzer_app"

from _bootstrap import bootstrap  # placeholder SECRET_KEY, LOG_DIR, app on sys.path — before processing.*

bootstrap(secret="training-only-not-a-real-secret", log_dir=REPO_ROOT / "training" / "logs")

import cv2  # noqa: E402

from processing.config import settings  # noqa: E402
from processing.filetypes import VIDEO_EXTS  # noqa: E402
from processing.frame_extractor import extract_frames_from_video  # noqa: E402


SORT_DIRS = ("good", "bad")
SHORTLIST_DIR = "_shortlist"
SKIP_NAMES = {".DS_Store"}

# Shortlist clustering settings. Production adopted the same 16/16 phash on
# 2026-09-14 (config.perceptual_hash_*), and since 2026-09-19 decides what
# reaches OCR by scroll-aware selection + metric coverage instead; the shortlist
# stays a plain 16/16 clustering, which is what labelling wants.
#
# Why not 8/4 (the old production setting): at that resolution a frame
# is reduced to 64 low-frequency bits, and in these recordings the story strip,
# header and nav bar are pixel-identical throughout while the numbers that
# differ are small text on a plain panel — exactly the high-frequency detail an
# 8x8 hash discards. Measured on 3c50a8ce: 47 visibly different screens
# (a loading skeleton, Views 22,360, a 24-donut, Back 2,507, an Audience split)
# collapsed into one cluster, and across the corpus 59% of multi-frame clusters
# held screens that differ.
#
# 16/16 is the same 6.25% bit tolerance as 4/64, just computed at higher
# resolution. Against pixel-level ground truth it keeps genuinely identical
# frames collapsed just as well (93%) while separating clearly different frames
# far better (56% -> 90%).
#
# The shortlist surfaces more frames than production sends, on purpose: a frame
# you never see is a frame you cannot label.
SHORTLIST_HASH_SIZE = 16
SHORTLIST_HASH_DISTANCE = 16


def find_videos(train_dir: Path) -> List[Path]:
    """Videos sitting directly in train/. Subfolders are extraction output."""
    return sorted(
        p
        for p in train_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("._")
    )


def content_hash(path: Path) -> str:
    """MD5 of the file. Cheap at these sizes (a few MB per clip)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def known_hashes(train_dir: Path) -> Dict[str, str]:
    """{content hash: folder name} for videos already extracted.

    Recorded in video_info.json at extraction time, so re-hashing the whole
    library on every run is unnecessary.
    """
    seen = {}
    for info_file in train_dir.glob("*/video_info.json"):
        try:
            info = json.loads(info_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        digest = info.get("content_md5")
        if digest:
            seen[digest] = info_file.parent.name
    return seen


def probe(video: Path) -> Dict:
    """Source-video properties, read before any conversion."""
    cap = cv2.VideoCapture(str(video))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return {
        "width": width,
        "height": height,
        "fps": round(fps, 3) if fps else None,
        "frame_count": frames,
        "duration_seconds": round(frames / fps, 2) if fps else None,
        "orientation": "portrait" if height > width else "landscape",
    }


def mean_luma(paths: List[Path], sample: int = 24) -> Optional[float]:
    """Average frame brightness, for dark/light-mode detection."""
    if not paths:
        return None
    step = max(1, len(paths) // sample)
    values = []
    for p in paths[::step][:sample]:
        img = cv2.imread(str(p))
        if img is not None:
            values.append(float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()))
    return round(sum(values) / len(values), 2) if values else None


def cluster_near_duplicates(
    paths: List[Path],
    hash_size: int = SHORTLIST_HASH_SIZE,
    distance: int = SHORTLIST_HASH_DISTANCE,
) -> Tuple[List[Path], Dict[str, List[str]]]:
    """Group near-identical frames for the labelling shortlist.

    Returns (representatives, {representative_name: [member names]}).
    See SHORTLIST_HASH_SIZE for why these settings differ from production's.
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        print("  [shortlist] needs ImageHash + Pillow; skipping")
        return list(paths), {}

    kept: List[Tuple[Path, "imagehash.ImageHash"]] = []
    clusters: Dict[str, List[str]] = {}
    for path in paths:
        try:
            with Image.open(path) as img:
                current = imagehash.phash(img, hash_size=hash_size)
        except Exception as exc:  # noqa: BLE001 — a bad frame must not stop extraction
            print(f"  [shortlist] phash failed for {path.name}: {exc}")
            kept.append((path, None))  # unhashable frames stand alone rather than vanish
            clusters[path.name] = [path.name]
            continue

        match = next(
            (rep for rep, h in kept if h is not None and (current - h) <= distance),
            None,
        )
        if match is not None:
            clusters[match.name].append(path.name)
        else:
            kept.append((path, current))
            clusters[path.name] = [path.name]
    return [p for p, _ in kept], clusters


def build_shortlist(frames: List[Path], out_dir: Path,
                    hash_size: int = SHORTLIST_HASH_SIZE,
                    distance: int = SHORTLIST_HASH_DISTANCE) -> Tuple[int, Dict[str, List[str]]]:
    reps, clusters = cluster_near_duplicates(frames, hash_size, distance)
    shortlist = out_dir / SHORTLIST_DIR
    if shortlist.exists():
        for old in shortlist.iterdir():
            old.unlink()
    else:
        shortlist.mkdir(parents=True)

    for rep in reps:
        link = shortlist / rep.name
        # Relative symlink so the folder survives being moved or copied.
        link.symlink_to(Path("..") / "frames" / rep.name)
    return len(reps), clusters


def already_done(out_dir: Path) -> bool:
    frames = out_dir / "frames"
    return frames.is_dir() and any(frames.glob("frame_*.jpg"))


def sorted_counts(out_dir: Path) -> Dict[str, int]:
    counts = {}
    for name in SORT_DIRS:
        d = out_dir / name
        counts[name] = len([p for p in d.rglob("*") if p.is_file() and p.name not in SKIP_NAMES]) if d.is_dir() else 0
    return counts


def process_video(video: Path, train_dir: Path, args, seen: Dict[str, str]) -> Optional[Dict]:
    out_dir = train_dir / video.stem

    # Two copies of one recording are worse than wasted disk: training splits by
    # video, so the same footage under two names lands one copy in train and one
    # in validation and quietly inflates the score.
    digest = content_hash(video)
    if digest in seen and seen[digest] != video.stem:
        print(f"\n{video.name}\n  byte-identical to '{seen[digest]}' (md5 {digest[:8]}) — skipping.\n"
              f"  Delete this file, or pass --allow-duplicates to extract it anyway.")
        return None
    seen.setdefault(digest, video.stem)

    if already_done(out_dir) and not args.force:
        counts = sorted_counts(out_dir)
        n = len(list((out_dir / "frames").glob("frame_*.jpg")))
        print(f"\n{video.name}\n  already extracted ({n} frames; sorted good={counts['good']} bad={counts['bad']}) — skipping, use --force to redo")
        return None

    print(f"\n{video.name}")
    source = probe(video)
    print(f"  source: {source['width']}x{source['height']} {source['orientation']}, "
          f"{source['fps']} fps, {source['frame_count']} frames, {source['duration_seconds']}s")

    frames_dir = out_dir / "frames"
    count, frames = extract_frames_from_video(
        video,
        frames_dir,
        job_id=video.stem,
        frame_interval=args.frame_interval,
        convert_to_720=not args.no_720p,
    )
    if not frames:
        print("  !! no frames extracted — unreadable or unsupported video")
        return None

    for name in SORT_DIRS:
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    first = cv2.imread(str(frames[0]))
    extracted_size = [int(first.shape[1]), int(first.shape[0])] if first is not None else None
    luma = mean_luma(frames)
    theme = None
    if luma is not None:
        theme = "dark" if luma < settings.dark_frame_threshold else "light"

    print(f"  extracted {count} frames at {extracted_size[0]}x{extracted_size[1]}"
          if extracted_size else f"  extracted {count} frames")
    print(f"  mean luma {luma} -> theme={theme}"
          + (f" (dark boost will fire; threshold {settings.dark_frame_threshold})" if theme == "dark" else ""))

    n_reps, clusters = (0, {})
    if not args.no_shortlist:
        n_reps, clusters = build_shortlist(frames, out_dir, args.shortlist_hash_size,
                                           args.shortlist_distance)
        pct = round(100 * n_reps / count) if count else 0
        print(f"  shortlist: {n_reps} representatives of {count} frames ({pct}%) -> {SHORTLIST_DIR}/")
        (out_dir / "clusters.json").write_text(json.dumps(clusters, indent=1), encoding="utf-8")

    info = {
        "video": video.name,
        "content_md5": digest,
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "extraction": {
            "frame_interval": args.frame_interval or settings.frame_interval,
            "convert_to_720p": not args.no_720p,
            "extracted_frames": count,
            "frame_size": extracted_size,
        },
        "appearance": {
            "mean_luma": luma,
            "theme": theme,
            "dark_frame_threshold": settings.dark_frame_threshold,
        },
        "shortlist": {
            "representatives": n_reps,
            "hash_size": args.shortlist_hash_size,
            "distance": args.shortlist_distance,
        },
        "shortlist_representatives": n_reps,
        # Fill these in by hand — training slices its metrics by them, which is
        # how you find out whether one UI version or language is dragging.
        "ui_version": "",
        "language": "",
        "notes": "",
    }
    (out_dir / "video_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"  -> {out_dir}/  (sort frames/ into good/ and bad/)")
    return info


def rebuild_shortlists(train_dir: Path, only: set, args) -> None:
    """Recluster existing frames into a new shortlist, without re-extracting.

    Frames are untouched, so anything already moved into good/ or bad/ keeps
    working — those are symlinks into frames/, and the filenames do not change.
    Refuses to touch a folder that has been sorted unless --force, since the
    shortlist is the thing you sorted from and silently redefining it mid-way
    would leave you unsure which frames you had actually reviewed.
    """
    targets = sorted(
        d for d in train_dir.iterdir()
        if d.is_dir() and (d / "frames").is_dir() and (not only or d.name in only)
    )
    if not targets:
        print(f"No matching video folders in {train_dir}"
              + (f" for --only {sorted(only)}" if only else ""))
        return

    missing = only - {d.name for d in targets}
    if missing:
        print(f"[warn] --only named folders that do not exist: {sorted(missing)}")

    print(f"Rebuilding {len(targets)} shortlist(s) at "
          f"hash_size={args.shortlist_hash_size} distance<={args.shortlist_distance}\n")
    total_before = total_after = 0
    for d in targets:
        sorted_n = sum(sorted_counts(d).values())
        if sorted_n and not args.force:
            print(f"{d.name}\n  already sorted ({sorted_n} frames in good/bad) — skipping, "
                  f"use --force to rebuild anyway")
            continue

        frames = sorted((d / "frames").glob("frame_*.jpg"))
        before = len(list((d / SHORTLIST_DIR).iterdir())) if (d / SHORTLIST_DIR).is_dir() else 0
        n_reps, clusters = build_shortlist(frames, d, args.shortlist_hash_size, args.shortlist_distance)
        (d / "clusters.json").write_text(json.dumps(clusters, indent=1), encoding="utf-8")

        info_file = d / "video_info.json"
        if info_file.is_file():
            info = json.loads(info_file.read_text(encoding="utf-8"))
            info["shortlist"] = {
                "representatives": n_reps,
                "hash_size": args.shortlist_hash_size,
                "distance": args.shortlist_distance,
                "rebuilt_at": datetime.now().isoformat(timespec="seconds"),
            }
            info["shortlist_representatives"] = n_reps
            info_file.write_text(json.dumps(info, indent=2), encoding="utf-8")

        total_before += before
        total_after += n_reps
        print(f"{d.name}\n  {len(frames)} frames: {before} -> {n_reps} representatives "
              f"(+{n_reps - before} newly surfaced)")

    print(f"\n{'=' * 60}\nshortlists: {total_before} -> {total_after} "
          f"representatives (+{total_after - total_before})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--train-dir", type=Path, default=REPO_ROOT / "train",
                    help="folder holding the videos (default: <repo>/train)")
    ap.add_argument("--frame-interval", type=int, default=None,
                    help=f"keep every Nth frame (default: config value, {settings.frame_interval})")
    ap.add_argument("--no-720p", action="store_true",
                    help="skip the ffmpeg 720p downscale (production does apply it)")
    ap.add_argument("--shortlist-hash-size", type=int, default=SHORTLIST_HASH_SIZE,
                    help=f"phash size for shortlist clustering (default {SHORTLIST_HASH_SIZE})")
    ap.add_argument("--shortlist-distance", type=int, default=SHORTLIST_HASH_DISTANCE,
                    help=f"max phash distance within a cluster (default {SHORTLIST_HASH_DISTANCE})")
    ap.add_argument("--rebuild-shortlist", action="store_true",
                    help="rebuild shortlists from existing frames; does not re-extract")
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated video folder names to act on (default: all)")
    ap.add_argument("--no-shortlist", action="store_true",
                    help="do not build the near-duplicate shortlist")
    ap.add_argument("--allow-duplicates", action="store_true",
                    help="extract a video even if it is byte-identical to one already extracted")
    ap.add_argument("--force", action="store_true",
                    help="re-extract videos that already have frames")
    args = ap.parse_args()

    train_dir: Path = args.train_dir
    if not train_dir.is_dir():
        train_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created {train_dir}\nPut your videos in it and run this again.")
        return

    only = {n.strip() for n in args.only.split(",") if n.strip()}

    if args.rebuild_shortlist:
        rebuild_shortlists(train_dir, only, args)
        return

    videos = find_videos(train_dir)
    if only:
        videos = [v for v in videos if v.stem in only]
    if not videos:
        print(f"No videos found in {train_dir}\n"
              f"Supported extensions: {', '.join(sorted(VIDEO_EXTS))}")
        return

    print(f"{len(videos)} video(s) in {train_dir}")
    seen = {} if args.allow_duplicates else known_hashes(train_dir)
    processed = [info for v in videos
                 if (info := process_video(v, train_dir, args, seen)) is not None]

    if processed:
        total = sum(i["extraction"]["extracted_frames"] for i in processed)
        shortlisted = sum(i["shortlist_representatives"] for i in processed)
        themes = {}
        for i in processed:
            themes[i["appearance"]["theme"]] = themes.get(i["appearance"]["theme"], 0) + 1
        print(f"\n{'=' * 60}")
        print(f"{len(processed)} video(s), {total} frames to sort"
              + (f" ({shortlisted} if you sort the shortlists)" if shortlisted else ""))
        print(f"themes: {themes}")
        print("\nNext: move each video's frames/ into its good/ and bad/ folders,")
        print("then run training/train_frame_classifier.py --data-dir train")


if __name__ == "__main__":
    main()
