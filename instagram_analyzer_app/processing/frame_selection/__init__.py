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


from ..logger import get_logger
from . import align, cover, pages, select, structs, verify  # noqa: F401  (eager: same import footprint)
from .align import _blocks, _load_gray, row_match, vertical_shift
from .cover import _cover_page, _fixed_bands
from .pages import _build_pages, _link_to_earlier, _stillness
from .select import content_covered, drop_duplicate_content, select_by_scroll, select_frames
from .structs import (
    ALIGN_WIDTH, BLOCK, BLOCK_MATCH, FIXED_ROW_DIFF, LINK_SAMPLES, MIN_EVIDENCE, MIN_MATCH_FRAC,
    PEAK_RATIO, ROW_HEIGHT, ROW_SLACK, STILL_FRAC, TEXTURE_STD, Selection, _Frame,
)
from .verify import (
    _ALIGN_CACHE, _align_uncached, _aligned_rows, _covered_rows, _match_at, _rows_matching,
    _stable_rows, _verify,
)


# The flat module did this at import; the name is pinned so the log file is unchanged.
logger = get_logger("processing.frame_selection")

__all__ = [
    "Selection", "select_frames", "select_by_scroll", "drop_duplicate_content", "content_covered",
    "vertical_shift", "row_match",
    "ALIGN_WIDTH", "FIXED_ROW_DIFF", "ROW_HEIGHT", "ROW_SLACK", "LINK_SAMPLES",
    "BLOCK", "BLOCK_MATCH", "TEXTURE_STD", "MIN_EVIDENCE", "MIN_MATCH_FRAC", "PEAK_RATIO", "STILL_FRAC",
    "_Frame", "_load_gray", "_blocks", "_stillness", "_link_to_earlier", "_build_pages",
    "_fixed_bands", "_cover_page", "_ALIGN_CACHE", "_rows_matching", "_aligned_rows",
    "_align_uncached", "_match_at", "_stable_rows", "_covered_rows", "_verify", "logger",
]
