"""Metric-coverage selection: the fewest frames that show every metric.

Runs after frame selection (frame_selection.py), on the frames it chose. Each
is read locally with Tesseract (English + Persian) to find which schema
metrics appear with a number, and a greedy set cover keeps the fewest frames
that together show every metric found anywhere in the video.

Fail-safe where it can fail:
- A frame where no metric is recognised but a number is on screen is kept. It
  may be a garbled read of a screen that matters.
- A frame whose labelled numbers are not all visible in some kept frame is
  kept (the value backstop), so a label read but mis-paired cannot cost a value.
- If Tesseract is not installed, or errors, the input is returned unchanged.

Evaluated with training/eval_selection.py (ground truth, Top sources and the
gender split included: no metric lost) and training/eval_feedback.py (human
review verdicts from review-rounds/).

Detection is deliberately about *which* metric is on screen, never its value:
the vision model still reads every number. A mis-read digit here changes nothing.
"""


from . import detect, labels, select, tesseract, text  # noqa: F401  (eager: same import footprint)
from .detect import (
    _CLOCK, _INFO_ICON, _PAGE_TITLE, _WORD, _count_other_labels, _is_section_header,
    _panel_start, _skip_line, _split_kind, evidence_values, has_real_number,
    metric_value_pairs, metrics_in_text, values_in,
)
from .labels import (
    ICON_STRIP, _DEMOGRAPHICS, _GENDER, _GENDER_PATTERNS, _LABELS, _LEADING_JUNK, _MONTHS,
    _NOT_A_VALUE, _NOT_EVIDENCE, _OTHER_LABELS, _PANEL_HEADERS, _PATTERNS,
    _SECTION_INTERACTIONS, _SECTION_VIEWS, _SOURCES, _SOURCE_HEADER, _SOURCE_PATTERNS, _TABS,
    _TRAILING_JUNK, _gender, _is_filter_row, _labels, _source,
)
from .select import Coverage, select_by_coverage
from .tesseract import _lock, read_lines, tesseract_available   # not _available: rebound via `global`
from .text import (
    _DECIMAL, _NUMBER, _PERSIAN_DIGITS, _canonical, _fold, _has_number, _is_guardable,
    _is_percent_line, _mostly_numbers, _norm, _numbers,
)

__all__ = [
    "Coverage", "select_by_coverage",
    "metrics_in_text", "metric_value_pairs", "values_in", "evidence_values", "has_real_number",
    "read_lines", "tesseract_available", "ICON_STRIP",
    "_norm", "_fold", "_numbers", "_has_number", "_canonical", "_mostly_numbers", "_is_percent_line",
    "_is_guardable", "_PERSIAN_DIGITS", "_NUMBER", "_DECIMAL",
    "_LABELS", "_GENDER", "_SOURCES", "_NOT_A_VALUE", "_TABS", "_PATTERNS", "_GENDER_PATTERNS",
    "_SOURCE_PATTERNS", "_SOURCE_HEADER", "_LEADING_JUNK", "_TRAILING_JUNK", "_OTHER_LABELS",
    "_SECTION_VIEWS", "_SECTION_INTERACTIONS", "_DEMOGRAPHICS", "_MONTHS", "_NOT_EVIDENCE",
    "_PANEL_HEADERS", "_labels", "_gender", "_source", "_is_filter_row",
    "_CLOCK", "_WORD", "_PAGE_TITLE", "_INFO_ICON", "_split_kind", "_is_section_header",
    "_count_other_labels", "_panel_start", "_skip_line", "_lock",
]
