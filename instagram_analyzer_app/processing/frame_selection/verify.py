"""Row-by-row verification that every good frame is readable in some pick.

_ALIGN_CACHE is the one shared alignment memo: select.py imports it by name and
only ever clears it, so there is exactly one dict in the process.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .align import row_match, vertical_shift
from .structs import ROW_HEIGHT, ROW_SLACK, _Frame


_ALIGN_CACHE: Dict[Tuple[int, int], Optional[Tuple[int, np.ndarray]]] = {}


def _rows_matching(a: _Frame, b: _Frame, dy: int) -> np.ndarray:
    """Per row of `a`: does row r show the same content as row r + dy of `b`?
    False where `b` does not reach. Shared by the direct and page-offset paths."""
    h = a.gray.shape[0]
    match = np.zeros(h, bool)
    lo, hi = max(0, -dy), min(h, h - dy)
    if hi > lo:
        match[lo:hi] = row_match(a.gray[lo:hi], b.gray[lo + dy:hi + dy])
    return match


def _aligned_rows(a: _Frame, b: _Frame) -> Optional[Tuple[int, np.ndarray]]:
    """Directly align b onto a. Returns (dy, match) where match[r] says whether
    row r of `a` shows the same content as the corresponding row of `b`
    (False where b does not reach). None when the two cannot be aligned."""
    key = (id(a), id(b))
    if key not in _ALIGN_CACHE:
        _ALIGN_CACHE[key] = _align_uncached(a, b)
    return _ALIGN_CACHE[key]


def _align_uncached(a: _Frame, b: _Frame) -> Optional[Tuple[int, np.ndarray]]:
    dy = vertical_shift(b.gray, a.gray)   # row r of a == row r + dy of b
    if dy is None:
        back = vertical_shift(a.gray, b.gray)
        dy = -back if back is not None else None
    if dy is None:
        return None
    return dy, _rows_matching(a, b, dy)


def _match_at(a: _Frame, b: _Frame) -> Optional[Tuple[int, np.ndarray]]:
    """Row matches of a against b. Same page: check content at the offset the
    page chain implies (works however far apart they are; a wrong offset just
    fails to match). Different pages: align them directly."""
    if a.page == b.page and a.page >= 0:
        dy = a.y - b.y
        match = _rows_matching(a, b, dy)
        if match.any():
            return dy, match
    return _aligned_rows(a, b)


def _stable_rows(frames: List[_Frame], i: int) -> np.ndarray:
    """Rows of frame i whose content also appears in a time neighbour. Rows
    that change frame to frame (a playing thumbnail, a moving touch dot) are
    not content anyone needs; everything else must be covered. When no
    neighbour can be aligned, every row counts — the safe default."""
    f = frames[i]
    stable = np.zeros(f.gray.shape[0], bool)
    aligned = False
    for j in (i - 1, i + 1):
        if 0 <= j < len(frames) and frames[j].gray is not None and frames[j].gray.shape == f.gray.shape:
            r = _match_at(f, frames[j])
            if r is not None:
                aligned = True
                stable |= r[1]
    return stable if aligned else np.ones_like(stable)


def _covered_rows(g: _Frame, chosen: List[_Frame], row: int) -> np.ndarray:
    """Rows of g readable, with the same content, in some chosen frame —
    counting only content at least `row` away from a chosen frame's cut edge."""
    h = g.gray.shape[0]
    covered = np.zeros(h, bool)
    for c in chosen:
        if c is g:
            return np.ones(h, bool)
        if c.gray is None or c.gray.shape != g.gray.shape:
            continue
        r = _match_at(g, c)
        if r is None:
            continue
        dy, match = r
        lo, hi = max(0, -dy), min(h, h - dy)
        if dy < 0:
            lo += row      # c's top edge falls inside g: a row cut there is unreadable in c
        if dy > 0:
            hi -= row
        if hi > lo:
            covered[lo:hi] |= match[lo:hi]
    return covered


def _verify(frames: List[_Frame], chosen: List[_Frame], bands: Dict[int, Tuple[int, int]],
            h: int) -> List[_Frame]:
    """Make sure every good frame's content is in some chosen frame, by direct
    comparison — never trusting page offsets, which accumulate error along a
    chain of frames and cannot see content change under a pinned header (tab
    switches, numbers loading in). Adds frames until nothing is uncovered."""
    row = max(1, int(h * ROW_HEIGHT))
    need = {}
    for i, f in enumerate(frames):
        if not f.good or f.gray is None:
            continue
        top, bottom = bands.get(f.page, (int(h * 0.10), int(h * 0.08)))
        mask = _stable_rows(frames, i)
        mask[:top] = False
        mask[h - bottom:] = False
        need[f.order] = mask

    def uncovered(g: _Frame, pool: List[_Frame]) -> bool:
        missing = need[g.order] & ~_covered_rows(g, pool, row)
        return int(missing.sum()) > ROW_SLACK

    chosen = list(chosen)
    todo = [f for f in frames if f.order in need and f not in chosen and uncovered(f, chosen)]
    while todo:
        # the uncovered frame that would cover the most other uncovered frames; stillest on ties
        best = max(todo, key=lambda f: (sum(1 for g in todo if g is f or not uncovered(g, [f])),
                                        -f.stillness, f.score, -f.order))
        chosen.append(best)
        todo = [g for g in todo if g is not best and uncovered(g, chosen)]
    return chosen
