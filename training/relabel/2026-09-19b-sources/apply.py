#!/usr/bin/env python3
"""Apply the 2026-09-19b label moves (JSON move lists in this folder).

    python training/relabel/2026-09-19b-sources/apply.py            # dry run
    python training/relabel/2026-09-19b-sources/apply.py --apply
    python training/relabel/2026-09-19b-sources/apply.py --restore  # back to labels_before.json

Each move file lists {"video", "frame", "from", "to"}; entries are renamed
between the sibling good/ and bad/ folders, so relative symlinks stay valid.
New per-video folders created by add_round01.py are not touched here; --restore
leaves them (delete train/<video>/ by hand to remove a round-01 video).
"""
import json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parents[2] / "train"


def moves():
    out = []
    for f in sorted(HERE.glob("moves_*.json")):
        out += json.loads(f.read_text())["moves"]
    return out


def main():
    if "--restore" in sys.argv:
        snap = json.loads((HERE / "labels_before.json").read_text())["labels"]
        n = 0
        for video, classes in snap.items():
            for cls, frames in classes.items():
                other = "bad" if cls == "good" else "good"
                for fr in frames:
                    want, there = TRAIN / video / cls / fr, TRAIN / video / other / fr
                    if not os.path.lexists(want) and os.path.lexists(there):
                        os.rename(there, want); n += 1
        print(f"restored {n} file(s)")
        return
    todo = []
    for m in moves():
        src = TRAIN / m["video"] / m["from"] / m["frame"]
        dst = TRAIN / m["video"] / m["to"] / m["frame"]
        if os.path.lexists(dst) and not os.path.lexists(src):
            continue                                   # already applied
        if not os.path.lexists(src):
            sys.exit(f"missing: {src}")
        if os.path.lexists(dst):
            sys.exit(f"would overwrite: {dst}")
        todo.append((src, dst))
    print(f"{len(todo)} move(s) pending")
    if "--apply" in sys.argv:
        for s, d in todo:
            os.rename(s, d)
        print("applied")
    else:
        print("dry run: pass --apply")


if __name__ == "__main__":
    main()
