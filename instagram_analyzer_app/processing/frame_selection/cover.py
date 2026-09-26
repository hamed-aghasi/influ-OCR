"""The fewest frames whose windows cover every row a page showed."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .structs import FIXED_ROW_DIFF, ROW_HEIGHT, _Frame


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
            # Classifier score first, stillness second. Stillness alone picked
            # the wrong frame whenever a page was still loading: a placeholder
            # panel is perfectly still, while the frame where the numbers have
            # just rendered differs from its predecessor and so looks "moving".
            # Round-03: ten frames of 634d6f63 scored 0.996+ and none was sent.
            # With equal scores this is the old behaviour, so a genuine hold
            # still wins its tie.
            pick = min(pool, key=lambda c: (-round(c[1].score, 2), c[1].stillness, c[1].order))
            chosen.append(pick)
            reach = pick[0][1]

    groups: Dict[int, List[_Frame]] = {c[1].order: [] for c in chosen}
    for (s, e), f in spans:
        best = max(chosen, key=lambda c: min(e, c[0][1]) - max(s, c[0][0]))
        groups[best[1].order].append(f)
    return groups
