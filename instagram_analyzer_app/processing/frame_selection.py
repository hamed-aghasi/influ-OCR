"""Choosing which classifier-good frames go to OCR.

Two strategies, picked by `settings.frame_selection`:

- ``phash``  — `dedup.dedupe_frames`: drop frames whose perceptual hash is
  within a distance of one already kept. Blind to scrolling: a page shifted by
  a few pixels is a different picture to a whole-image hash, so a slow scroll
  through one screen is sent as dozens of "unique" frames.

- ``scroll`` — reconstruct the scroll. Consecutive frames are aligned by
  vertical template matching; frames linked by a pure vertical shift belong to
  one *page* at a known y-offset. From each page it keeps the fewest good
  frames whose visible windows together cover everything any good frame on
  that page showed. Among frames that cover about the same span, the one the
  recorder held still wins, then the higher classifier score (a tie-breaker
  only — stillness never filters anything out).

The scroll strategy fails safe. Page offsets only *propose* picks; a
verification pass then checks every good frame against the picks by direct
alignment and row-by-row content comparison, and adds frames until every
stable row of every good frame is readable in some pick. Alignment that is
uncertain (fast scroll, motion blur, tab switch, content loading in) leaves a
frame uncovered, which adds a frame — it never drops content.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import settings
from .logger import get_logger


logger = get_logger(__name__)

# Frames are aligned at this width; offsets below are in these pixels.
ALIGN_WIDTH = 160
FIXED_ROW_DIFF = 3.0     # rows this similar in two scrolled frames are pinned UI
ROW_HEIGHT = 0.10        # tallest metric row (label + number), as a fraction of frame height
ROW_SLACK = 2            # uncovered rows tolerated per frame (alignment rounding at window edges)
LINK_SAMPLES = 8         # frames per earlier page tried when re-linking after a cut


@dataclass
class Selection:
    selected: List[Path]                       # frames to send, in time order
    duplicates: List[Path]                     # good frames not sent
    groups: Dict[Path, List[Path]] = field(default_factory=dict)  # sent frame -> good frames it stands for
    pages: int = 0
    strategy: str = "scroll"
    # scroll strategy only: each frame's alignment state and its page's pinned
    # bands, kept so later passes (drop_duplicate_content) can compare frames by
    # content without re-aligning the whole video
    frame_info: Dict[Path, "_Frame"] = field(default_factory=dict, repr=False)
    sequence_of: Dict[Path, int] = field(default_factory=dict, repr=False)
    bands: Dict[Tuple[int, int], Tuple[int, int]] = field(default_factory=dict, repr=False)


@dataclass
class _Frame:
    path: Path
    order: int
    good: bool
    score: float
    gray: Optional[np.ndarray]
    page: int = -1
    y: int = 0
    stillness: float = 0.0   # max grey difference to either time neighbour; lower = held still


# ---------------------------------------------------------------- alignment


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


BLOCK = 8                # column block width for content comparison (ALIGN_WIDTH pixels)
BLOCK_MATCH = 20.0       # a row matches when every block's mean abs difference is under this;
                         # a changed digit in one block is enough to break it
TEXTURE_STD = 4.0        # rows flatter than this (blank panel, white gap) carry no evidence
MIN_EVIDENCE = 25        # matched textured rows needed to accept an offset
MIN_MATCH_FRAC = 0.6     # ...and this share of the textured rows in the overlap
PEAK_RATIO = 0.8         # runner-up offset (>3 px away) must score below this share of the best
STILL_FRAC = 0.85        # share of textured rows matching in place that makes a frame "not scrolled"
                         # even if something local changed (touch dot, a number loading in)


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


# ---------------------------------------------------------------- coverage


def _fixed_bands(page: List[_Frame], h: int) -> Tuple[int, int]:
    """Rows pinned to the screen (headers, tab bars, nav bar): identical in two
    frames of the page taken at different scroll offsets. They are visible in
    every frame, so they must not count as scrolled content a frame covers."""
    lo = min(page, key=lambda f: f.y)
    hi = max(page, key=lambda f: f.y)
    if hi.y - lo.y < 8:
        return int(h * 0.10), int(h * 0.08)
    rows = np.abs(lo.gray - hi.gray).mean(axis=1) < FIXED_ROW_DIFF
    top = 0
    while top < len(rows) and rows[top]:
        top += 1
    bottom = 0
    while bottom < len(rows) and rows[len(rows) - 1 - bottom]:
        bottom += 1
    cap = int(h * 0.45)
    return min(top, cap), min(bottom, cap)


def _cover_page(page: List[_Frame], h: int) -> Dict[int, List[_Frame]]:
    """Fewest good frames such that every row fully readable in some good frame
    is fully readable in a chosen one. With rows up to ROW_HEIGHT tall, that
    means chosen windows must overlap by ROW_HEIGHT wherever good windows do.
    Stillness then breaks ties among picks that keep the count minimal — it
    never adds a frame. Returns {chosen frame order: good frames it stands for}."""
    good = [f for f in page if f.good]
    if not good:
        return {}
    top, bottom = _fixed_bands(page, h)
    row = max(1, int(h * ROW_HEIGHT))
    spans = sorted((((f.y + top, max(f.y + h - bottom, f.y + top + 1)), f) for f in good),
                   key=lambda s: (s[0], s[1].order))

    # connected stretches of the page that good frames showed
    components: List[List[Tuple[Tuple[int, int], _Frame]]] = []
    for s in spans:
        if components and s[0][0] < max(c[0][1] for c in components[-1]):
            components[-1].append(s)
        else:
            components.append([s])

    chosen: List[Tuple[Tuple[int, int], _Frame]] = []
    for comp in components:
        start, end = comp[0][0][0], max(c[0][1] for c in comp)

        def candidates(reach: Optional[int]):
            if reach is None:
                return [c for c in comp if c[0][0] <= start]
            strict = [c for c in comp if c[0][0] <= reach - row and c[0][1] > reach]
            # No frame overlaps by a full row: a row straddling `reach` was never
            # fully readable in any good frame, so plain continuity suffices.
            return strict or [c for c in comp if c[0][0] <= reach and c[0][1] > reach]

        def picks_needed(reach: int) -> int:
            n = 0
            while reach < end:
                nxt = candidates(reach)
                if not nxt:
                    return 1 << 30
                reach = max(c[0][1] for c in nxt)
                n += 1
            return n

        reach: Optional[int] = None
        while reach is None or reach < end:
            cands = candidates(reach)
            if not cands:
                break
            best = min(picks_needed(c[0][1]) for c in cands)
            pool = [c for c in cands if picks_needed(c[0][1]) == best]
            pick = min(pool, key=lambda c: (c[1].stillness, -c[1].score, c[1].order))
            chosen.append(pick)
            reach = pick[0][1]

    groups: Dict[int, List[_Frame]] = {c[1].order: [] for c in chosen}
    for (s, e), f in spans:
        best = max(chosen, key=lambda c: min(e, c[0][1]) - max(s, c[0][0]))
        groups[best[1].order].append(f)
    return groups


# ---------------------------------------------------------------- verification


_ALIGN_CACHE: Dict[Tuple[int, int], Optional[Tuple[int, np.ndarray]]] = {}


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
    h = a.gray.shape[0]
    match = np.zeros(h, bool)
    lo, hi = max(0, -dy), min(h, h - dy)
    if hi > lo:
        match[lo:hi] = row_match(a.gray[lo:hi], b.gray[lo + dy:hi + dy])
    return dy, match


def _match_at(a: _Frame, b: _Frame) -> Optional[Tuple[int, np.ndarray]]:
    """Row matches of a against b. Same page: check content at the offset the
    page chain implies (works however far apart they are; a wrong offset just
    fails to match). Different pages: align them directly."""
    if a.page == b.page and a.page >= 0:
        dy = a.y - b.y
        h = a.gray.shape[0]
        match = np.zeros(h, bool)
        lo, hi = max(0, -dy), min(h, h - dy)
        if hi > lo:
            match[lo:hi] = row_match(a.gray[lo:hi], b.gray[lo + dy:hi + dy])
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


# ---------------------------------------------------------------- entry points


def select_by_scroll(
    sequences: Sequence[Sequence[Path]],
    good: Iterable[Path],
    scores: Optional[Mapping[Path, float]] = None,
) -> Selection:
    """`sequences` are the extracted frames of each source video, in time
    order (bad frames included — they carry the scroll between good ones)."""
    good_set = {Path(p) for p in good}
    scores = {Path(k): v for k, v in (scores or {}).items()}
    chosen: List[_Frame] = []
    groups: Dict[Path, List[Path]] = {}
    n_pages = 0
    unhashable = set()
    _ALIGN_CACHE.clear()
    frame_info: Dict[Path, _Frame] = {}
    sequence_of: Dict[Path, int] = {}
    all_bands: Dict[Tuple[int, int], Tuple[int, int]] = {}

    for si, seq in enumerate(sequences):
        frames = [_Frame(Path(p), i, Path(p) in good_set, scores.get(Path(p), 0.0), _load_gray(Path(p)))
                  for i, p in enumerate(seq)]
        _stillness(frames)
        for f in frames:
            if f.good and f.gray is None:   # unreadable here but good to the classifier: keep, don't guess
                unhashable.add(f.path)
                chosen.append(f)
                groups[f.path] = [f.path]
        pages = _build_pages(frames)
        n_pages += len(pages)
        h = next((f.gray.shape[0] for f in frames if f.gray is not None), 0)
        by_order = {f.order: f for f in frames}
        picks: List[_Frame] = []
        bands: Dict[int, Tuple[int, int]] = {}
        for pid, page in enumerate(pages):
            if page:
                bands[pid] = _fixed_bands(page, h)
            picks.extend(by_order[o] for o in _cover_page(page, h))
        picks = _verify(frames, picks, bands, h)
        chosen.extend(picks)
        for f in frames:
            if f.gray is not None:
                frame_info[f.path] = f
                sequence_of[f.path] = si
        all_bands.update({(si, pid): b for pid, b in bands.items()})
        for f in frames:           # each good frame is represented by the pick most like it
            if not f.good or f.gray is None:
                continue
            rep = f if f in picks else max(
                picks, key=lambda c: int(_covered_rows(f, [c], 0).sum()) if c.gray is not None else -1)
            groups.setdefault(rep.path, []).append(f.path)

    # Frames outside any sequence (e.g. a lone image) are kept as they are.
    in_sequences = {Path(p) for seq in sequences for p in seq}
    for p in sorted(good_set - in_sequences):
        chosen.append(_Frame(p, 0, True, scores.get(p, 0.0), None))
        groups[p] = [p]

    seq_index = {Path(p): (si, i) for si, seq in enumerate(sequences) for i, p in enumerate(seq)}
    ordered = sorted({f.path for f in chosen}, key=lambda p: seq_index.get(p, (len(sequences), 0)))

    # No phash pass here: _verify already compared every good frame against the
    # picks by content, and a whole-image hash merges screens that differ only
    # in one number row (it drops Profile visits/Follows on 7cb649a2).
    survivors = ordered
    survivors_set = set(survivors)
    duplicates = [p for p in sorted(good_set, key=lambda p: seq_index.get(p, (len(sequences), 0)))
                  if p not in survivors_set]
    _ALIGN_CACHE.clear()
    return Selection(selected=survivors, duplicates=duplicates,
                     groups={s: groups.get(s, [s]) for s in survivors}, pages=n_pages,
                     frame_info=frame_info, sequence_of=sequence_of, bands=all_bands)


def content_covered(selection: Selection, target: Path, others: Sequence[Path]) -> bool:
    """Is every textured row of `target` (pinned bands excluded) readable, with
    the same content and away from a cut edge, in one of `others` from the same
    video? False when that cannot be established (no alignment data, another
    video, nothing to compare with) — never a guess towards "covered"."""
    info, seq_of = selection.frame_info, selection.sequence_of
    gf = info.get(Path(target))
    if gf is None or gf.gray is None:
        return False
    peers = [info[Path(o)] for o in others
             if Path(o) != Path(target) and Path(o) in info and seq_of.get(Path(o)) == seq_of.get(Path(target))]
    if not peers:
        return False
    h = gf.gray.shape[0]
    top, bottom = selection.bands.get((seq_of[Path(target)], gf.page), (int(h * 0.10), int(h * 0.08)))
    need = gf.gray.std(axis=1) > TEXTURE_STD
    need[:top] = False
    need[h - bottom:] = False
    missing = need & ~_covered_rows(gf, peers, max(1, int(h * ROW_HEIGHT)))
    return int(missing.sum()) <= ROW_SLACK


def drop_duplicate_content(
    selection: Selection,
    paths: Sequence[Path],
    rank: Optional[Mapping[Path, float]] = None,
) -> Tuple[List[Path], List[Path]]:
    """Drop frames whose content is entirely readable in another kept frame.

    A frame is a duplicate when every textured row of it (pinned header and nav
    bands excluded) appears, with the same content and away from a cut edge, in
    some other kept frame of the same video: the row-by-row check _verify
    already relies on. It needs no OCR, so it catches duplicates that OCR
    misreads make look different ("Likes 124" read as "'3 124"). Frames are
    tried least valuable first (lowest `rank`, then earliest), so of two
    identical frames the better-ranked one stays. Only works on a scroll
    Selection; for anything else nothing is dropped. Returns (kept, dropped).
    """
    rank = rank or {}
    keep = list(paths)
    position = {p: i for i, p in enumerate(paths)}
    dropped: List[Path] = []
    _ALIGN_CACHE.clear()
    for g in sorted(paths, key=lambda p: (rank.get(p, 0), position[p])):
        if content_covered(selection, g, [o for o in keep if o != g]):
            keep.remove(g)
            dropped.append(g)
    _ALIGN_CACHE.clear()
    return keep, dropped


def select_frames(
    sequences: Sequence[Sequence[Path]],
    good: Sequence[Path],
    scores: Optional[Mapping[Path, float]] = None,
    strategy: Optional[str] = None,
) -> Selection:
    """The pipeline's entry point: which good frames go to OCR."""
    strategy = strategy or settings.frame_selection
    if strategy == "scroll":
        sel = select_by_scroll(sequences, good, scores)
    elif strategy == "phash":
        from .dedup import dedupe_frames

        unique, dupes = dedupe_frames(list(good))
        sel = Selection(selected=unique, duplicates=dupes, groups={u: [u] for u in unique},
                        strategy="phash")
    else:
        raise ValueError(f"unknown frame_selection strategy: {strategy!r}")
    sel.strategy = strategy
    return sel
