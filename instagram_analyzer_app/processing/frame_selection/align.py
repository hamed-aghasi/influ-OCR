"""Aligning two frames of a scrolled page: is one a vertical shift of the other?"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .structs import (
    ALIGN_WIDTH, BLOCK, BLOCK_MATCH, MIN_EVIDENCE, MIN_MATCH_FRAC, PEAK_RATIO, STILL_FRAC,
    TEXTURE_STD,
)


def _load_gray(path: Path) -> Optional[np.ndarray]:
    try:
        with open(path, "rb") as f:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    except OSError:
        return None
    if img is None:
        return None
    h, w = img.shape
    return cv2.resize(img, (ALIGN_WIDTH, max(1, round(h * ALIGN_WIDTH / w))),
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def _blocks(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    w = w // BLOCK * BLOCK
    return gray[:, :w].reshape(h, -1, BLOCK).mean(axis=2)


def row_match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row: do two equally sized grey regions show the same content?"""
    return np.abs(_blocks(a) - _blocks(b)).max(axis=1) < BLOCK_MATCH


def vertical_shift(prev: np.ndarray, cur: np.ndarray) -> Optional[int]:
    """How far the content moved up between two frames (scrolling down is
    positive), in ALIGN_WIDTH pixels: row r of `cur` shows row r + dy of
    `prev`. None when no offset is clearly supported.

    Every offset is tried; each scores the number of textured rows that match
    block for block. Whole-frame evidence rather than one template strip,
    because Insights screens are mostly white space and repeated widget rows —
    a single strip matches the wrong row of a list far too easily. A pinned
    header only supports offset 0, so on a real scroll it loses to the content.
    """
    if prev.shape != cur.shape:
        return None
    h = prev.shape[0]
    pb, cb = _blocks(prev), _blocks(cur)
    ptex, ctex = prev.std(axis=1) > TEXTURE_STD, cur.std(axis=1) > TEXTURE_STD

    # Pinned bands: rows from each edge inward that are unchanged in place
    # (blank rows pass through). Headers, the Story carousel, tab bars and the
    # nav bar all sit still while the content between them scrolls.
    same = np.abs(cb - pb).max(axis=1) < BLOCK_MATCH
    textured = ctex | ptex
    if textured.sum() and (same & textured).sum() >= STILL_FRAC * textured.sum():
        return 0
    still = same | ~textured
    top = 0
    while top < h and still[top]:
        top += 1
    if top == h:
        return 0
    bottom = h
    while bottom > top and still[bottom - 1]:
        bottom -= 1

    scores = {}
    for dy in range(-(h * 3) // 4, (h * 3) // 4 + 1):
        lo, hi = max(top, -dy), min(bottom, h - dy)
        if hi - lo < 8:
            continue
        tex = ctex[lo:hi] | ptex[lo + dy:hi + dy]
        n = int(tex.sum())
        if n < MIN_EVIDENCE // 2:
            continue
        match = (np.abs(cb[lo:hi] - pb[lo + dy:hi + dy]).max(axis=1) < BLOCK_MATCH) & tex
        scores[dy] = (int(match.sum()), n)
    if not scores:
        return None
    best = max(scores, key=lambda d: (scores[d][0], -abs(d)))
    good, n = scores[best]
    if good < MIN_EVIDENCE // 2 or good < MIN_MATCH_FRAC * n:
        return None
    runner = max((scores[d][0] for d in scores if abs(d - best) > 3), default=0)
    if runner > PEAK_RATIO * good:
        return None
    return best
