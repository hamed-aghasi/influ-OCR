"""Entry points: scroll-aware selection, content-duplicate dropping, strategy switch."""

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


from ..config import settings
from .align import _load_gray
from .cover import _cover_page, _fixed_bands
from .pages import _build_pages, _stillness
from .structs import ROW_HEIGHT, ROW_SLACK, TEXTURE_STD, Selection, _Frame
from .verify import _ALIGN_CACHE, _covered_rows, _verify


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
        from ..dedup import dedupe_frames

        unique, dupes = dedupe_frames(list(good))
        sel = Selection(selected=unique, duplicates=dupes, groups={u: [u] for u in unique},
                        strategy="phash")
    else:
        raise ValueError(f"unknown frame_selection strategy: {strategy!r}")
    sel.strategy = strategy
    return sel
