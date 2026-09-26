"""The metric vocabulary: on-screen labels (English / Persian) and their matchers.

Every *_PATTERNS / *_HEADERS table is derived at import by running _norm over the
label dicts below, so the dicts, _norm and the derived tables must stay together.
Dict key order is behaviour: sorted(..., key=-len) is stable, and ties between
equal-length patterns are broken by insertion order.
"""

import re
from typing import Dict, List, Optional, Sequence

from .text import _norm

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


# Labels that are not schema metrics but sit in the same tiles/rows; counted so
# a value line with fewer numbers than labels credits nobody.
_OTHER_LABELS = tuple(_norm(p) for p in (
    "average watch time", "watch time", "reposts", "saves", "متوسط زمان تماشا", "بازنشرها", "ذخیره‌ها"))
_SECTION_VIEWS = tuple(_norm(p) for p in ("views", "بازدیدها", "who viewed", "چه کسانی"))
_SECTION_INTERACTIONS = tuple(_norm(p) for p in ("interactions", "تعاملات"))


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


_DEMOGRAPHICS = tuple(_norm(p) for p in ("age", "country", "gender", "سن", "کشور", "جنسیت"))


_MONTHS = tuple(_norm(m) for m in (
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "ژانویه", "فوریه", "مارس", "آوریل", "مه", "ژوئن", "ژوئیه", "اوت", "سپتامبر", "اکتبر", "نوامبر", "دسامبر"))
_NOT_EVIDENCE = tuple(_norm(p) for p in ("watch time", "زمان تماشا"))


# Section headers of the metrics panel: the first one marks where the panel
# begins, and the thumbnail carousel above it ends. See _panel_start.
_PANEL_HEADERS = tuple(_norm(p) for p in (
    "overview", "summary", "interactions", "profile activity", "audience", "navigation",
    "مرور کلی", "خلاصه", "تعاملات", "فعالیت نمایه", "مخاطب", "پیمایش"))
