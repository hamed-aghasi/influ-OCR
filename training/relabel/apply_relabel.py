#!/usr/bin/env python3
"""Apply the good -> bad relabel decisions to train/<video>/good|bad.

    python training/relabel/apply_relabel.py            # dry run: print what would move
    python training/relabel/apply_relabel.py --apply    # move the files
    python training/relabel/apply_relabel.py --restore  # put every label back as snapshotted

Decisions live in decisions/<video>.json ({frame: {"label": "good"|"bad", "why": ...}}),
one per video, covering every frame that was in good/. Only good -> bad moves are
made; bad/ is never touched. Entries are renamed, not copied: the relative
symlinks into ../frames/ stay valid because good/ and bad/ are siblings.

The pre-relabel state is in labels_before_2026-09-19.json; --restore returns
every file to the folder it was in then.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TRAIN = REPO / "train"
SNAPSHOT = HERE / "labels_before_2026-09-19.json"


def plan():
    moves = []
    for dec_file in sorted((HERE / "decisions").glob("*.json")):
        video = dec_file.stem
        decisions = json.loads(dec_file.read_text(encoding="utf-8"))
        for frame, d in sorted(decisions.items()):
            if d["label"] != "bad":
                continue
            src = TRAIN / video / "good" / frame
            dst = TRAIN / video / "bad" / frame
            if not os.path.lexists(src):
                if os.path.lexists(dst):
                    continue  # already applied
                sys.exit(f"missing: {src}")
            if os.path.lexists(dst):
                sys.exit(f"would overwrite: {dst}")
            moves.append((src, dst))
    return moves


def restore():
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["labels"]
    moved = 0
    for video, classes in snap.items():
        for cls, frames in classes.items():
            other = "bad" if cls == "good" else "good"
            for frame in frames:
                want = TRAIN / video / cls / frame
                there = TRAIN / video / other / frame
                if not os.path.lexists(want) and os.path.lexists(there):
                    os.rename(there, want)
                    moved += 1
    print(f"restored {moved} file(s) to their pre-relabel folders")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()
    if args.restore:
        restore()
        return
    moves = plan()
    by_video = {}
    for src, _ in moves:
        by_video[src.parent.parent.name] = by_video.get(src.parent.parent.name, 0) + 1
    for video, n in sorted(by_video.items()):
        print(f"  {video}: {n} good -> bad")
    print(f"{len(moves)} move(s) across {len(by_video)} video(s)")
    if not args.apply:
        print("dry run — pass --apply to move them")
        return
    for src, dst in moves:
        os.rename(src, dst)
    print("applied")


if __name__ == "__main__":
    main()
