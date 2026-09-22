#!/usr/bin/env python3
"""Apply (or undo) the round-02 'loading state' labels.

Copies the labelled frames from review-rounds/round-02/<video>/frames into
train/<video>/{good,bad}, and the video itself into train/ so the trainer can
split by video. Frames already labelled are left alone and reported.

    python training/relabel/2026-09-22-loading-2/apply.py --apply
    python training/relabel/2026-09-22-loading-2/apply.py --restore
"""
import argparse, json, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROUND = REPO / "review-rounds" / "round-02b"
TRAIN = REPO / "train"
LABELS = json.loads((HERE / "labels.json").read_text())
ADDED = HERE / "added.json"


def stem(video_dir: str) -> str:
    return video_dir[:-4] if video_dir.endswith(" (1)") else video_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--restore", action="store_true")
    a = ap.parse_args()
    if a.restore:
        if not ADDED.exists():
            raise SystemExit("nothing recorded as added")
        for rel in json.loads(ADDED.read_text()):
            p = REPO / rel
            if p.exists():
                p.unlink()
        print(f"removed {len(json.loads(ADDED.read_text()))} frames")
        return
    if not a.apply:
        raise SystemExit("pass --apply or --restore")

    added, skipped = [], 0
    for video_dir, spec in LABELS.items():
        if video_dir.startswith("_"):
            continue
        src = ROUND / video_dir / "frames"
        name = stem(video_dir)
        mp4 = ROUND.parent / f"{video_dir}.mp4"
        if mp4.exists() and not (TRAIN / f"{name}.mp4").exists():
            shutil.copy2(mp4, TRAIN / f"{name}.mp4")
        for label in ("good", "bad"):
            dst = TRAIN / name / label
            dst.mkdir(parents=True, exist_ok=True)
            for n in spec[label]:
                f = f"frame_{n:06d}.jpg"
                if (TRAIN / name / "good" / f).exists() or (TRAIN / name / "bad" / f).exists():
                    skipped += 1
                    continue
                shutil.copy2(src / f, dst / f)
                added.append(str((dst / f).relative_to(REPO)))
    ADDED.write_text(json.dumps(added, indent=1))
    print(f"added {len(added)} frames, skipped {skipped} already labelled")


if __name__ == "__main__":
    main()
