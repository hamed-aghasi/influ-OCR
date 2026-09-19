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

Against training/ground_truth/ (10 videos, 21 metrics incl. the gender split)
after scroll selection: 168 -> 68 frames, no metric lost.

Detection is deliberately about *which* metric is on screen, never its value:
the vision model still reads every number. A mis-read digit here changes nothing.
"""

import re
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import cv2

from .config import settings
from .logger import job_logger


# Longest first: "profile visits" must win over "views" when both could match,
# and Persian "بازدیدهای نمایه" over "بازدیدها". Patterns are matched against
# normalised text (lower case, no spaces/ZWNJ/punctuation, Persian letters unified).
_LABELS: Dict[str, Sequence[str]] = {
    "external_link_taps": ("external link taps", "ضربه روی پیوند خارجی", "ضربه‌های پیوند خارجی"),
    "profile_activity": ("profile activity", "فعالیت نمایه"),
    "profile_visits": ("profile visits", "بازدیدهای نمایه", "بازدید از نمایه"),
    "accounts_reached": ("accounts reached", "viewers", "بینندگان", "بازدیدکنندگان",
                         "حسابهایی که بازدید کردهاند", "حسابهای دسترسی یافته"),
    "non_followers": ("non-followers", "nonfollowers", "غیردنبالکنندگان"),
    "followers": ("followers", "دنبالکنندگان"),
    "interactions": ("interactions", "تعاملات"),
    "sticker_taps": ("sticker taps", "ضربه روی برچسب", "ضربه‌های برچسب"),
    "links_clicks": ("link clicks", "bio link clicks", "کلیکهای پیوند"),
    "navigation": ("navigation", "پیمایش"),
    "next_story": ("next story", "داستان بعدی"),
    "forward": ("forward", "بازارسال"),
    "exited": ("exited", "خروج از داستان"),
    "back": ("back", "برگشت"),
    "likes": ("likes", "پسندها", "پسند"),
    "replies": ("replies", "comments", "پاسخها", "پاسخ", "نظرها"),
    "shares": ("shares", "اشتراکگذاریها", "اشتراکگذاری"),
    "follows": ("follows", "دنبال کردن", "دنبالکردن"),
    "views": ("views", "بازدیدها"),
}

# Audience > Gender rows ("Men 95.9%"). Matched only at the start of a line and
# only with a percentage: "men" is a substring of "comments" and "engagement",
# and Persian "زن" of "بازنشر" (repost).
_GENDER = {"gender_men": ("men", "مردان", "مرد"), "gender_women": ("women", "زنان", "زن")}

# Lines that contain a metric word but are chart headers or tab bars, not values.
_NOT_A_VALUE = ("viewsovertime", "درطولزمان", "whenpeoplelikedyourreel", "whatimpacts",
                "whataffects", "howlongpeoplewatched", "topsourcesofviews", "rate",
                "نرخ", "منابعبرتر", "summary", "خلاصه")
_TABS = (("overview", "engagement", "audience"), ("مرورکلی", "تعامل", "مخاطب"))


def _is_filter_row(n: str) -> bool:
    """The "All · Followers · Non-followers" buttons above the Views-over-time
    chart (Persian: همه · دنبال‌کنندگان · غیردنبال‌کنندگان). They name the
    split but show no value; the line under them is a chart axis or tooltip
    ("200K", "24.4"), which must not be read as the split's value."""
    return (n.startswith("all") or n.startswith("همه")) and ("followers" in n or "دنبالکنندگان" in n)
ICON_STRIP = ("likes", "replies", "shares")   # the Reel icon strip carries these, unlabelled

_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_NUMBER = re.compile(r"\d[\d,.٫٬]*\s*[km%]?|\d+s\b")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_PERSIAN_DIGITS).lower()
    text = text.replace("ي", "ی").replace("ك", "ک").replace("‌", "").replace("‏", "").replace("‎", "")
    return re.sub(r"[\s\-_:.,،()©®|\"'`’‘]+", "", text)


# Every (pattern, metric), longest pattern first, so a longer label always wins
# the characters it covers ("profile visits" before "views").
_PATTERNS = sorted(((_norm(p), m) for m, pats in _LABELS.items() for p in pats),
                   key=lambda pm: -len(pm[0]))
_GENDER_PATTERNS = sorted(((_norm(p), m) for m, pats in _GENDER.items() for p in pats),
                          key=lambda pm: -len(pm[0]))
_LEADING_JUNK = re.compile(r"^[^a-z\u0600-\u06ff]+")


def _has_number(line: str) -> bool:
    return bool(_NUMBER.search(unicodedata.normalize("NFKC", line).translate(_PERSIAN_DIGITS).lower()))


# Labels that are not schema metrics but sit in the same tiles/rows; counted so
# a value line with fewer numbers than labels credits nobody.
_OTHER_LABELS = tuple(_norm(p) for p in (
    "average watch time", "watch time", "reposts", "saves", "متوسط زمان تماشا", "بازنشرها", "ذخیره‌ها"))
_SECTION_VIEWS = tuple(_norm(p) for p in ("views", "بازدیدها", "who viewed", "چه کسانی"))
_SECTION_INTERACTIONS = tuple(_norm(p) for p in ("interactions", "تعاملات"))
_CLOCK = re.compile(r"\b\d{1,2}:\d{2}\b")


def _numbers(line: str) -> List[str]:
    return [t.strip() for t in _NUMBER.findall(unicodedata.normalize("NFKC", line).translate(_PERSIAN_DIGITS).lower())]


def _mostly_numbers(line: str) -> bool:
    tokens = [t for t in re.split(r"\s+", line.translate(_PERSIAN_DIGITS)) if t]
    nums = [t for t in tokens if re.fullmatch(r"[\d,.]+[km]?", t.lower())]
    return len(nums) >= 3 and len(nums) >= 0.6 * len(tokens)


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


def _count_other_labels(n: str) -> int:
    count, rest = 0, n
    for pat in sorted(_OTHER_LABELS, key=len, reverse=True):
        if pat in rest:
            count += 1
            rest = rest.replace(pat, "\x00")
    return count


def _is_percent_line(line: str) -> bool:
    text = unicodedata.normalize("NFKC", line).translate(_PERSIAN_DIGITS)
    return bool(re.search(r"\d\s*[%٪]", text))


def _gender(n: str) -> Optional[str]:
    head = _LEADING_JUNK.sub("", n)
    for pat, metric in _GENDER_PATTERNS:
        if head.startswith(pat):
            return metric
    return None


def _labels(n: str) -> List[str]:
    """Schema metrics whose label is in normalised line n, longest match first."""
    hits, rest = [], n
    for pat, metric in _PATTERNS:
        if pat and pat in rest and metric not in hits:
            hits.append(metric)
            rest = rest.replace(pat, "\x00")   # consume: "views" can't re-match inside "profile visits"
    return hits


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
    for i, n in enumerate(norm):
        if _CLOCK.search(lines[i]):
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
            # glyphs OCR as noise, e.g. "اه 95.9%") on the next non-noise line
            for j in (i, i + 1, i + 2):
                if j >= len(lines) or (j > i and (_gender(norm[j]) or _labels(norm[j]))):
                    break
                if _is_percent_line(lines[j]):
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
        if not own:
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


_DEMOGRAPHICS = tuple(_norm(p) for p in ("age", "country", "gender", "سن", "کشور", "جنسیت"))


def values_in(lines: Sequence[str]) -> Set[str]:
    """Numbers on rows that carry a schema label (or sit right under one) —
    the backstop for a label that was read but could not be paired with its
    value. Rates, audience splits, chart axes and the clock are left out:
    guarding those kept ~50 extra frames on the ground truth for nothing."""
    norm = [_norm(l) for l in lines]
    vals: Set[str] = set()
    for i, line in enumerate(lines):
        if _CLOCK.search(line) or any(x in norm[i] for x in _NOT_A_VALUE) or _is_filter_row(norm[i]):
            continue
        if any(norm[i].startswith(d) for d in _DEMOGRAPHICS):
            continue
        above = i > 0 and (bool(_labels(norm[i - 1])) or bool(_gender(norm[i - 1]))) \
            and not _is_filter_row(norm[i - 1])
        labelled = bool(_labels(norm[i])) or bool(_gender(norm[i])) or (above and not _labels(norm[i]))
        if labelled:
            vals.update(t.replace(",", "").replace(" ", "") for t in _numbers(line))
    return vals


def has_real_number(lines: Sequence[str]) -> bool:
    """Any number of two or more digits outside the clock — a frame without
    one has nothing for the vision model to read."""
    return any(len(re.sub(r"\D", "", t)) >= 2
               for line in lines if not _CLOCK.search(line) for t in _numbers(line))


# ---------------------------------------------------------------- OCR

_lock = threading.Lock()
_available: Optional[bool] = None


def tesseract_available() -> bool:
    global _available
    with _lock:
        if _available is None:
            try:
                import pytesseract

                pytesseract.get_tesseract_version()
                langs = set(pytesseract.get_languages(config=""))
                _available = all(l in langs for l in settings.coverage_ocr_lang.split("+"))
            except Exception:  # noqa: BLE001 — missing binary, missing package, missing language
                _available = False
    return _available


def read_lines(path: Path) -> List[str]:
    import pytesseract

    img = cv2.imread(str(path))
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 100:            # dark mode: Tesseract wants dark text on light
        gray = 255 - gray
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(gray, lang=settings.coverage_ocr_lang, config="--psm 6")
    return [l for l in text.splitlines() if l.strip()]


# ---------------------------------------------------------------- selection


@dataclass
class Coverage:
    selected: List[Path]
    dropped: List[Path]
    metrics: Dict[Path, Set[str]] = field(default_factory=dict)
    applied: bool = True
    reason: str = ""
    # per frame: "cover" (needed for a metric), "unrecognised" (no metric read but a
    # number on screen), "backstop" (shows a labelled value no other kept frame
    # shows), "covered" (dropped: its metrics are in kept frames), "blank"
    # (dropped: no metric and no number)
    reasons: Dict[Path, str] = field(default_factory=dict)


def select_by_coverage(frames: Sequence[Path], job_id: Optional[str] = None,
                       detect=None) -> Coverage:
    """Keep the fewest `frames` that show every metric any of them shows.

    `frames` should already be in time order. `detect(path)` returns a set of
    metrics, or (metrics, values); injectable for tests. By default it OCRs
    the frame.
    """
    log = job_logger(__name__, job_id)
    frames = list(frames)
    if detect is None:
        if not tesseract_available():
            log.warning("Tesseract (%s) unavailable; metric-coverage selection skipped", settings.coverage_ocr_lang)
            return Coverage(frames, [], applied=False, reason="tesseract unavailable")
        def detect(p):
            lines = read_lines(p)
            return metrics_in_text(lines), values_in(lines), has_real_number(lines)

    metrics: Dict[Path, Set[str]] = {}
    values: Dict[Path, Set[str]] = {}
    numeric: Dict[Path, bool] = {}
    for p in frames:
        try:
            got = detect(p)
            if not isinstance(got, tuple):
                got = (got, set(), True)
            metrics[p], values[p], numeric[p] = set(got[0]), set(got[1]), bool(got[2])
        except Exception as exc:  # noqa: BLE001 — one unreadable frame must not drop anything
            log.warning("Coverage OCR failed on %s (%s); keeping it", p.name, exc)
            metrics[p], values[p], numeric[p] = set(), set(), True

    # Nothing recognised but a number on screen: keep it, don't guess. Nothing
    # recognised and no number (skeleton, blank panel): nothing to read, drop.
    unknown = [p for p in frames if not metrics[p] and numeric[p]]
    todo = set().union(*metrics.values()) if metrics else set()
    chosen: List[Path] = []
    order = {p: i for i, p in enumerate(frames)}
    while todo:
        best = max((p for p in frames if p not in chosen and metrics[p]),
                   key=lambda p: (len(metrics[p] & todo), len(metrics[p]), -order[p]))
        if not metrics[best] & todo:
            break
        chosen.append(best)
        todo -= metrics[best]

    # Backstop: a frame showing a number no kept frame shows is kept, whatever
    # the label detection said — a missed label must not become a dropped value.
    keep = set(chosen) | set(unknown)
    reasons = {p: "cover" for p in chosen}
    reasons.update({p: "unrecognised" for p in unknown})
    while True:
        seen = set().union(*(values[p] for p in keep)) if keep else set()
        orphans = [p for p in frames if p not in keep and values[p] - seen]
        if not orphans:
            break
        extra = max(orphans, key=lambda p: (len(values[p] - seen), -order[p]))
        keep.add(extra)
        reasons[extra] = "backstop"
    for p in frames:
        reasons.setdefault(p, "covered" if metrics[p] else "blank")

    selected = [p for p in frames if p in keep]
    dropped = [p for p in frames if p not in keep]
    log.info("Coverage: %d -> %d frames (%d unrecognised kept), %d metrics",
             len(frames), len(selected), len(unknown), len(set().union(*metrics.values())) if metrics else 0)
    return Coverage(selected, dropped, metrics, reasons=reasons)
