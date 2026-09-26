"""Shared data types and the tuning constants for scroll-aware selection."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


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


BLOCK = 8                # column block width for content comparison (ALIGN_WIDTH pixels)
BLOCK_MATCH = 20.0       # a row matches when every block's mean abs difference is under this;
                         # a changed digit in one block is enough to break it
TEXTURE_STD = 4.0        # rows flatter than this (blank panel, white gap) carry no evidence
MIN_EVIDENCE = 25        # matched textured rows needed to accept an offset
MIN_MATCH_FRAC = 0.6     # ...and this share of the textured rows in the overlap
PEAK_RATIO = 0.8         # runner-up offset (>3 px away) must score below this share of the best
STILL_FRAC = 0.85        # share of textured rows matching in place that makes a frame "not scrolled"
                         # even if something local changed (touch dot, a number loading in)
