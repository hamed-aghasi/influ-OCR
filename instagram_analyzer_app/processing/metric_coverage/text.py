"""Text normalisation and numeric tokens, English and Persian."""

import re
import unicodedata
from typing import List

_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_NUMBER = re.compile(r"\d[\d,.٫٬]*\s*[km%]?|\d+s\b")


def _fold(text: str) -> str:
    """NFKC, Persian and Arabic-Indic digits to ASCII, lower case."""
    return unicodedata.normalize("NFKC", text).translate(_PERSIAN_DIGITS).lower()


def _norm(text: str) -> str:
    text = _fold(text)
    text = text.replace("ي", "ی").replace("ك", "ک").replace("‌", "").replace("‏", "").replace("‎", "")
    return re.sub(r"[\s\-_:.,،()©®|\"'`’‘]+", "", text)


def _has_number(line: str) -> bool:
    return bool(_NUMBER.search(_fold(line)))


def _numbers(line: str) -> List[str]:
    # Tesseract often reads "%" as "°"; Persian uses "٪"
    text = _fold(line).replace("°", "%").replace("٪", "%")
    return [t.strip() for t in _NUMBER.findall(text)]


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


def _is_percent_line(line: str) -> bool:
    text = unicodedata.normalize("NFKC", line).translate(_PERSIAN_DIGITS)
    return bool(re.search(r"\d\s*[%٪]", text))


def _is_digit_run(line: str) -> bool:
    """Nothing but digits and their separators, at least two digits.

    A gender bar's percentage as Tesseract reads it on the Persian layout:
    "۸۴٫۸٪" fused with the bar glyphs into "۸۱۴۸۸۹ ۳۳000", the percent sign
    lost. No letters, so it cannot be an axis tick ("200K") or another label.
    Tesseract wraps RTL lines in bidi marks (U+200E/F), which are not digits."""
    text = unicodedata.normalize("NFKC", line).translate(_PERSIAN_DIGITS)
    text = re.sub(r"[​-‏⁦-⁩]", "", text)
    return bool(re.fullmatch(r"[\d\s,.٫٬]+", text)) and sum(c.isdigit() for c in text) >= 2


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
