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

# Reel "Top sources of views" rows: the label alone on its line, its percentage on
# the next (after bar glyphs). Matched only as a whole line and only on a frame
# that shows the section header or at least two source rows, because "Profile"
# and "Feed" are ordinary words elsewhere.
_SOURCES = {
    "source_feed": ("feed", "فید", "قید"),          # "قید": Tesseract's usual misread of فید
    "source_reels_tab": ("reels tab", "برگه reels", "reels برگه", "زبانه reels", "reels زبانه"),
    "source_profile": ("profile", "نمایه"),
    "source_explore": ("explore", "کاوش"),
    "source_stories": ("stories", "استوری‌ها", "داستان‌ها"),
    "source_search": ("search", "جستجو"),
    "source_other": ("other", "سایر"),
}

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
_TRAILING_JUNK = re.compile(r"[^a-z\u0600-\u06ff]+$")
_SOURCE_PATTERNS = {_norm(p): m for m, pats in _SOURCES.items() for p in pats}
_SOURCE_HEADER = tuple(_norm(p) for p in ("top sources of views", "منابع برتر", "برتر بازدید"))


def _source(n: str) -> Optional[str]:
    """The source a normalised line names, if the line is just that label (plus
    its value and OCR junk)."""
    label = _TRAILING_JUNK.sub("", _LEADING_JUNK.sub("", re.sub(r"[\d%٪]+", "", n)))
    return _SOURCE_PATTERNS.get(label)


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
    # Tesseract often reads "%" as "°"; Persian uses "٪"
    text = unicodedata.normalize("NFKC", line).translate(_PERSIAN_DIGITS).lower().replace("°", "%").replace("٪", "%")
    return [t.strip() for t in _NUMBER.findall(text)]


_WORD = re.compile(r"[a-z\u0600-\u06ff]{3,}")
_DECIMAL = re.compile(r"^(\d+)\.(\d)%?$")


def _canonical(token: str) -> str:
    """A value as compared across frames. Insights shows one-decimal values only
    as percentages (never above 100), and bar glyphs before them OCR as an extra
    leading digit ("96.5%" read as "896.5", "99.0%" as "699.0%"): drop leading
    digits until the value is a possible percentage, and write it with "%"."""
    t = token.replace(",", "").replace(" ", "")
    m = _DECIMAL.match(t)
    if not m:
        return t
    whole, frac = m.group(1), m.group(2)
    while len(whole) > 1 and int(whole) > 100:
        whole = whole[1:]
    return f"{int(whole)}.{frac}%"


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
        if _CLOCK.search(lines[i]) or any(x in n for x in _NOT_A_VALUE) or _is_filter_row(n):
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
        above = i > 0 and (bool(_labels(norm[i - 1])) or bool(_gender(norm[i - 1]))
                           or bool(_source(norm[i - 1]))) and not _is_filter_row(norm[i - 1])
        own_label = bool(_labels(norm[i])) or bool(_gender(norm[i])) or bool(_source(norm[i]))
        # a value line under a label has no label words of its own ("Saves 36" under
        # "Likes 124" is its own, non-schema row)
        bare = not _WORD.findall(unicodedata.normalize("NFKC", line).lower())
        if own_label or (above and bare):
            vals.update(_canonical(t) for t in _numbers(line))
    return vals


_MONTHS = tuple(_norm(m) for m in (
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "ژانویه", "فوریه", "مارس", "آوریل", "مه", "ژوئن", "ژوئیه", "اوت", "سپتامبر", "اکتبر", "نوامبر", "دسامبر"))
_NOT_EVIDENCE = tuple(_norm(p) for p in ("watch time", "زمان تماشا"))
_PAGE_TITLE = ("insights", "اطلاعاتآماری")
# Section headers of the metrics panel: the first one marks where the panel
# begins, and the thumbnail carousel above it ends. See _panel_start.
_PANEL_HEADERS = tuple(_norm(p) for p in (
    "overview", "summary", "interactions", "profile activity", "audience", "navigation",
    "مرور کلی", "خلاصه", "تعاملات", "فعالیت نمایه", "مخاطب", "پیمایش"))


def _is_guardable(token: str, min_digits: int = 2) -> bool:
    """Could this token be a metric count worth keeping a frame for?

    Not a percentage (OCR reads small ones inconsistently), not a K/M axis tick
    ("500K" on the views chart), and enough digits to identify anything.

    Two digits is the bar for a value that sits beside a label the detector
    recognised. With no label to anchor it, three is: round-01's bf96a4ca frame
    033 is a loading screen whose only "numbers" are a garbled status bar
    ("19 , ۳3"), and two digits let it through.
    """
    return (not token.endswith(("%", "k", "m"))
            and len(re.sub(r"\D", "", token)) >= min_digits)


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
        if (_CLOCK.search(lines[i]) or any(x in n for x in _NOT_A_VALUE) or _is_filter_row(n)
                or any(x in n for x in _NOT_EVIDENCE)):
            continue
        text = unicodedata.normalize("NFKC", lines[i]).translate(_PERSIAN_DIGITS).lower()
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
        if (_CLOCK.search(lines[i]) or any(x in n for x in _NOT_A_VALUE) or _is_filter_row(n)
                or any(x in n for x in _NOT_EVIDENCE)):
            continue
        text = unicodedata.normalize("NFKC", lines[i]).translate(_PERSIAN_DIGITS).lower()
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
    values: Dict[Path, Set[str]] = field(default_factory=dict)   # labelled values read per frame
    reason: str = ""
    # per frame: "cover" (needed for a metric), "backstop" (shows a labelled
    # value no other kept frame shows), "unrecognised" (no metric read, but a
    # value no other kept frame shows), "covered" (dropped: its metrics are in
    # kept frames), "blank" (dropped: nothing it shows is new)
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
            return (metrics_in_text(lines), values_in(lines),
                    evidence_values(lines), metric_value_pairs(lines))

    metrics: Dict[Path, Set[str]] = {}
    values: Dict[Path, Set[str]] = {}
    evidence: Dict[Path, Set[str]] = {}          # values found without a recognised label
    pairs: Dict[Path, Dict[str, Set[str]]] = {}
    for p in frames:
        try:
            got = detect(p)
            if not isinstance(got, tuple):
                got = (got, set(), set())
            third = got[2] if len(got) > 2 else set()
            # A detector may still answer the old yes/no; True then means
            # "something is on screen but I cannot say what", which must keep
            # the frame rather than silently drop it.
            if isinstance(third, bool):
                third = {f"\x00unreadable:{p}"} if third else set()
            metrics[p], values[p], evidence[p] = set(got[0]), set(got[1]), set(third)
            pairs[p] = got[3] if len(got) > 3 else {}
        except Exception as exc:  # noqa: BLE001 — one unreadable frame must not drop anything
            log.warning("Coverage OCR failed on %s (%s); keeping it", p.name, exc)
            # A value unique to this frame: nothing else can cover it, so a
            # frame we could not read is always kept.
            metrics[p], values[p], evidence[p], pairs[p] = set(), set(), {f"\x00unreadable:{p}"}, {}

    # Cross-frame evidence: when one frame pairs a metric with a value of three or
    # more digits ("Likes 124"), another frame showing that same value in a
    # labelled list shows that metric too, even if OCR garbled its label there
    # (round-01, 8102e03d: "Likes 124" read as "'3 124" made two identical
    # frames both look necessary).
    for a in frames:
        for metric, vals in pairs[a].items():
            for v in vals:
                if len(re.sub(r"\D", "", v)) < 3:
                    continue
                for b in frames:
                    if b is not a and metric not in metrics[b] and v in values[b]:
                        metrics[b].add(metric)

    todo = set().union(*metrics.values()) if metrics else set()
    chosen: List[Path] = []
    order = {p: i for i, p in enumerate(frames)}
    while todo:
        best = max((p for p in frames if p not in chosen and metrics[p]),
                   # ties go to the most complete frame, then the later one: a page
                   # still loading reads fewer values, and loads top to bottom over time
                   key=lambda p: (len(metrics[p] & todo), len(metrics[p]), len(values[p]), order[p]))
        if not metrics[best] & todo:
            break
        chosen.append(best)
        todo -= metrics[best]

    # One rule for everything the set cover did not pick: keep a frame only if
    # it shows a value no kept frame shows. A missed label must not become a
    # dropped value — but "I recognised nothing here" is not by itself a reason
    # to spend an OCR slot (round-02: loading skeletons whose numbers were
    # already on a sent frame, and rates tabs with no counts at all).
    #
    # For a frame whose labels were read, the guarded values are the labelled
    # ones; for a frame with no recognised metric, they are the evidence values
    # (evidence_values above). Percentages and K/M ticks are excluded either
    # way: OCR reads small percentages inconsistently ("1.1%" as "11%"), so
    # guarding them kept a new frame per misread (round-01b: three copies of one
    # sources list), and a single digit identifies nothing.
    keep = set(chosen)
    reasons = {p: "cover" for p in chosen}
    guarded = {p: ({v for v in values[p] if _is_guardable(v)} if metrics[p] else evidence[p])
               for p in frames}
    shown = {p: values[p] | (evidence[p] if not metrics[p] else set()) for p in frames}
    while True:
        seen = set().union(*(shown[p] for p in keep)) if keep else set()
        orphans = [p for p in frames if p not in keep and guarded[p] - seen]
        if not orphans:
            break
        # among frames showing the missing value, the most complete one (round-01:
        # a half-loaded frame was kept over the fully loaded frame_000060)
        extra = max(orphans, key=lambda p: (len(guarded[p] - seen), len(metrics[p]), len(values[p]), order[p]))
        keep.add(extra)
        reasons[extra] = "backstop" if metrics[extra] else "unrecognised"
    for p in frames:
        reasons.setdefault(p, "covered" if metrics[p] else "blank")

    selected = [p for p in frames if p in keep]
    dropped = [p for p in frames if p not in keep]
    unknown_kept = sum(1 for p in selected if reasons.get(p) == "unrecognised")
    log.info("Coverage: %d -> %d frames (%d unrecognised kept), %d metrics",
             len(frames), len(selected), unknown_kept, len(set().union(*metrics.values())) if metrics else 0)
    return Coverage(selected, dropped, metrics, values=values, reasons=reasons)
