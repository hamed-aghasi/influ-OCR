"""Chaining aligned frames into pages at known scroll offsets."""

from typing import List, Optional, Tuple

import numpy as np

from .align import vertical_shift
from .structs import LINK_SAMPLES, _Frame


def _stillness(frames: List[_Frame]) -> None:
    diffs = []
    for a, b in zip(frames, frames[1:]):
        if a.gray is None or b.gray is None or a.gray.shape != b.gray.shape:
            diffs.append(255.0)
        else:
            diffs.append(float(np.abs(a.gray - b.gray).mean()))
    for i, f in enumerate(frames):
        around = ([diffs[i - 1]] if i > 0 else []) + ([diffs[i]] if i < len(diffs) else [])
        f.stillness = max(around) if around else 0.0


def _link_to_earlier(f: _Frame, pages: List[List[_Frame]]) -> Optional[Tuple[int, int]]:
    """After a cut, try to place the frame on a page already seen — the
    recorder scrolling back up, or returning from another tab."""
    for pid in range(len(pages) - 1, -1, -1):
        samples, last_y = [], None
        for g in sorted(pages[pid], key=lambda g: g.y):
            if last_y is None or g.y - last_y >= f.gray.shape[0] * 0.25:
                samples.append(g)
                last_y = g.y
        for g in samples[:LINK_SAMPLES]:
            dy = vertical_shift(g.gray, f.gray)
            if dy is not None:
                return pid, g.y + dy
    return None


def _build_pages(frames: List[_Frame]) -> List[List[_Frame]]:
    pages: List[List[_Frame]] = []
    prev: Optional[_Frame] = None
    for f in frames:
        if f.gray is None:
            continue
        placed = None
        if prev is not None:
            dy = vertical_shift(prev.gray, f.gray)
            if dy is not None:
                placed = (prev.page, prev.y + dy)
            else:
                placed = _link_to_earlier(f, pages)
        if placed is None:
            pages.append([])
            placed = (len(pages) - 1, 0)
        f.page, f.y = placed
        pages[f.page].append(f)
        prev = f
    return pages
