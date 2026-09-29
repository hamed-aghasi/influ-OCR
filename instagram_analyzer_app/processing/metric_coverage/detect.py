"""Which metrics a frame shows, from its OCR'd lines (Tesseract output)."""

import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Set

from .labels import (
    ICON_STRIP, _DEMOGRAPHICS, _MONTHS, _NOT_A_VALUE, _NOT_EVIDENCE, _OTHER_LABELS,
    _PANEL_HEADERS, _SECTION_INTERACTIONS, _SECTION_VIEWS, _SOURCE_HEADER, _TABS, _gender,
    _is_filter_row, _labels, _source,
)
from .text import (
    _PERSIAN_DIGITS, _canonical, _fold, _is_digit_run, _is_guardable, _is_percent_line,
    _mostly_numbers, _norm, _numbers,
)

_CLOCK = re.compile(r"\b\d{1,2}:\d{2}\b")


_WORD = re.compile(r"[a-z\u0600-\u06ff]{3,}")


def _skip_line(line: str, n: str) -> bool:
    """A line that never carries a metric value: the status-bar clock, a chart
    header or tab bar, or the All / Followers / Non-followers filter row."""
    return bool(_CLOCK.search(line) or any(x in n for x in _NOT_A_VALUE) or _is_filter_row(n))


def _split_kind(norm: Sequence[str], i: int, section: str) -> str:
    """Whose Followers/Non-followers split is line i? The rows right after it
    say, even when the section header has scrolled off: Likes/Replies/Shares
    follow the Interactions split, Accounts reached/Viewers the Views split."""
    for j in range(i + 1, min(len(norm), i + 4)):
        after = set(_labels(norm[j]))
        if after & {"likes", "replies", "shares", "sticker_taps"}:
            return "interactions"
        if after & {"accounts_reached", "views"}:
            return "views"
    return section


_INFO_ICON = re.compile(r"[©®@ⓘ]\s*$|^[\s\u200e\u200f]*[©®@ⓘ]")


def _is_section_header(line: str) -> bool:
    """"Interactions ⓘ": a section title (OCR reads the info icon as ©, ®, @ or
    ⓘ, at the end, or at the start in RTL). Its value is the donut's inner label
    two lines down, not the next line, which at a screen edge is often the
    donut's arc read as digits (4eb4b131 frame 075: "Interactions ©" then
    "6-۷", credited Interactions it did not show)."""
    return bool(_INFO_ICON.search(line.strip().replace("\u200e", "").replace("\u200f", "")))


def _count_other_labels(n: str) -> int:
    count, rest = 0, n
    for pat in sorted(_OTHER_LABELS, key=len, reverse=True):
        if pat in rest:
            count += 1
            rest = rest.replace(pat, "\x00")
    return count


def metrics_in_text(lines: Sequence[str]) -> Set[str]:
    """Schema metrics shown with a value, from OCR'd lines in reading order.

    - A label counts when a number is on its own line, or when the next line
      holds at least as many numbers as the line holds labels (Summary tiles
      put two labels over two values; "Average watch time / Follows" over a
      lone "4s" credits nobody).
    - The Reel icon strip is a mostly-numeric line directly above the
      Overview/Engagement/Audience tab bar.
    - Followers/Non-followers under the *Interactions* donut are not the
      schema's split (that one is under Views), so they are skipped there.
    """
    found: Set[str] = set()
    norm = [_norm(l) for l in lines]
    section = ""
    sources_ctx = (any(h in n for n in norm for h in _SOURCE_HEADER)
                   or sum(1 for n in norm if _source(n)) >= 2)
    for i, n in enumerate(norm):
        if _CLOCK.search(lines[i]):
            continue
        src = _source(n) if sources_ctx else None
        if src:
            # its percentage on the same line, or the next line or two (bar glyphs first)
            for j in (i, i + 1, i + 2):
                if j >= len(lines) or (j > i and (_source(norm[j]) or _labels(norm[j]))):
                    break
                if re.search(r"\d", lines[j].translate(_PERSIAN_DIGITS)):
                    found.add(src)
                    break
            continue
        tab_hits = sum(t in n for t in _TABS[0] + _TABS[1])
        if tab_hits >= 2:
            j = i - 1
            if j >= 0 and not _CLOCK.search(lines[j]) and _mostly_numbers(lines[j]):
                found.update(ICON_STRIP)
            continue
        if any(x in n for x in _NOT_A_VALUE) or _is_filter_row(n):
            continue
        gender = _gender(n)
        if gender:
            # "Men 95.9%" on one line, or "Men" over its bar and "95.9%" (bar
            # glyphs OCR as noise, e.g. "اه 95.9%") on the next non-noise line.
            # On the Persian layout the bar and "۸۴٫۸٪" fuse into one digit run
            # with the percent sign lost ("۸۱۴۸۸۹ ۳۳000"), so a digit-only
            # next line counts too (68301206 frames 237-261, seen 2026-09-29).
            for j in (i, i + 1, i + 2):
                if j >= len(lines) or (j > i and (_gender(norm[j]) or _labels(norm[j]))):
                    break
                if _is_percent_line(lines[j]) or (j == i + 1 and _is_digit_run(lines[j])):
                    found.add(gender)
                    break
            continue
        labels = _labels(n)
        own = _numbers(lines[i])
        if any(n.startswith(h) for h in _SECTION_INTERACTIONS) and not own:
            section = "interactions"
        elif any(n.startswith(h) for h in _SECTION_VIEWS) or "navigation" in labels or "profile_activity" in labels:
            section = "views"
        if not labels:
            continue
        credited = labels if own else []
        if not own and not _is_section_header(lines[i]):
            # value line: the next line, or the one after if the next is OCR noise ("G", "—")
            for j in (i + 1, i + 2):
                if j >= len(lines) or _CLOCK.search(lines[j]):
                    break
                nxt = _numbers(lines[j])
                if nxt:
                    if not _labels(norm[j]) and len(nxt) >= len(labels) + _count_other_labels(n):
                        credited = labels
                    break
                if len(norm[j]) > 2:
                    break
        for m in credited:
            if m in ("followers", "non_followers") and _split_kind(norm, i, section) == "interactions":
                continue
            found.add(m)
    return found


def metric_value_pairs(lines: Sequence[str]) -> Dict[str, Set[str]]:
    """{metric: {value}} for the unambiguous rows only: exactly one schema label
    on the line, and exactly one number on that line or on a bare line right
    under it. Used as cross-frame evidence (see select_by_coverage)."""
    norm = [_norm(l) for l in lines]
    pairs: Dict[str, Set[str]] = {}
    for i, n in enumerate(norm):
        if _skip_line(lines[i], n):
            continue
        labels = _labels(n)
        if len(labels) != 1:
            continue
        own = _numbers(lines[i])
        if len(own) == 1:
            pairs.setdefault(labels[0], set()).add(_canonical(own[0]))
        elif not own and i + 1 < len(lines):
            nxt = _numbers(lines[i + 1])
            bare = not _WORD.findall(unicodedata.normalize("NFKC", lines[i + 1]).lower())
            if len(nxt) == 1 and bare and not _CLOCK.search(lines[i + 1]):
                pairs.setdefault(labels[0], set()).add(_canonical(nxt[0]))
    return pairs


def values_in(lines: Sequence[str]) -> Set[str]:
    """Numbers on rows that carry a schema label (or sit right under one) —
    the backstop for a label that was read but could not be paired with its
    value. Rates, audience splits, chart axes and the clock are left out:
    guarding those kept ~50 extra frames on the ground truth for nothing."""
    norm = [_norm(l) for l in lines]
    vals: Set[str] = set()
    for i, line in enumerate(lines):
        if _skip_line(line, norm[i]):
            continue
        if any(norm[i].startswith(d) for d in _DEMOGRAPHICS):
            continue
        above = i > 0 and (bool(_labels(norm[i - 1])) or bool(_gender(norm[i - 1]))
                           or bool(_source(norm[i - 1]))) and not _is_filter_row(norm[i - 1])
        own_label = bool(_labels(norm[i])) or bool(_gender(norm[i])) or bool(_source(norm[i]))
        # a value line under a label has no label words of its own ("Saves 36" under
        # "Likes 124" is its own, non-schema row)
        bare = not _WORD.findall(unicodedata.normalize("NFKC", line).lower())
        if own_label or (above and bare):
            vals.update(_canonical(t) for t in _numbers(line))
    return vals


_PAGE_TITLE = ("insights", "اطلاعاتآماری")


def _panel_start(norm: Sequence[str]) -> Optional[int]:
    """Index of the first metric section header ("Overview", "Views ⓘ", …).

    Everything above it is chrome: the status bar, the back arrow, and — on
    Story and Reel screens — the thumbnail carousel, whose per-thumbnail viewer
    badges are 3-5 digit numbers sitting right above the panel. Round-02b: on a
    loading screen those badges were the only numbers on the frame, so they
    alone kept it ("6777", "2430", "3408", "5325").
    """
    for i, n in enumerate(norm):
        if any(n.startswith(h) for h in _PANEL_HEADERS):
            return i
    return None


def evidence_values(lines: Sequence[str]) -> Set[str]:
    """Candidate metric values on a frame whose labels were NOT recognised.

    The same scan as has_real_number, but returning the numbers rather than a
    yes/no, so a frame with no recognised metric is kept only when it shows a
    value no kept frame shows (round-02: 25 of the 29 frames the reviewer
    rejected were kept by the old "a number and a word" rule — loading
    skeletons whose icon-strip numbers were already on a sent frame, and rates
    tabs whose only numbers are percentages).

    Counted only inside the metrics panel, below its first section header when
    one is readable: a number above it belongs to the carousel, not to a
    metric. Also excludes what is never a metric count: the status bar and
    anything above the page title, the clock and durations, dates, rates, chart
    axes and filter rows, watch time.
    """
    norm = [_norm(l) for l in lines]
    title = next((i for i, n in enumerate(norm[:6]) if any(t in n for t in _PAGE_TITLE)), None)
    panel = _panel_start(norm)
    if title is None and panel is None:
        # Neither the page title nor a section header is readable, so this is
        # not an Insights screen at all — it is the Reel/Story player, the
        # feed, a profile grid or a settings page. The net exists for a garbled
        # read of a screen that matters, not for every screen with a number on
        # it (round-03: 12 of the 21 rejected frames were the player itself,
        # kept on its like/comment counts).
        return set()
    start = title + 1 if title is not None else 1
    if panel is not None:
        start = max(start, panel + 1)
    found: Set[str] = set()
    for i in range(start, len(lines)):
        n = norm[i]
        if _skip_line(lines[i], n) or any(x in n for x in _NOT_EVIDENCE):
            continue
        text = _fold(lines[i])
        if any(_norm(w) in _MONTHS or _norm(w)[:3] in _MONTHS for w in _WORD.findall(text)):
            continue                      # "July 29", "Jul 9 Jul 30": a date, not a value
        found |= {_canonical(t) for t in _numbers(lines[i]) if _is_guardable(t, min_digits=3)}
    return found


def has_real_number(lines: Sequence[str]) -> bool:
    """Is there anything on screen the vision model could need, though no
    metric label was recognised? Decides whether an unrecognised frame is kept.

    Counts a number of three or more digits that is not a percentage or an axis
    tick (e.g. "23,813"), or any number beside a word (e.g. a garbled label with
    its value). Ignores what is never a metric: the status bar and anything
    above the page title, rates, chart axes and dates, watch time. A Top sources
    header with numbered rows under it counts even if no row was read.
    Round-01 review: rate screens ("Skip rate 46.0%", "100%", "0:00 1:15") and a
    loading skeleton with status-bar noise ("19 , ۳3") were kept by the older,
    any-two-digits rule.
    """
    norm = [_norm(l) for l in lines]
    title = next((i for i, n in enumerate(norm[:6]) if any(t in n for t in _PAGE_TITLE)), None)
    start = title + 1 if title is not None else 1
    for i in range(start, len(lines)):
        n = norm[i]
        if any(h in n for h in _SOURCE_HEADER):
            # counts only with rows under it (a header on the last line shows no data)
            if any(re.search(r"\d", l.translate(_PERSIAN_DIGITS)) for l in lines[i + 1:]):
                return True
            continue
        if _skip_line(lines[i], n) or any(x in n for x in _NOT_EVIDENCE):
            continue
        text = _fold(lines[i])
        tokens = _numbers(lines[i])      # "%" read as "°" is normalised here
        if not tokens:
            continue
        words = [w for w in _WORD.findall(text) if _norm(w) not in _MONTHS]
        if words:
            return True
        for t in tokens:
            digits = re.sub(r"\D", "", t)
            if len(digits) >= 3 and not t.endswith(("%", "k", "m")):
                return True
    return False
