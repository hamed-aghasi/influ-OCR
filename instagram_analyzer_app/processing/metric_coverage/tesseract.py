"""Local Tesseract: availability probe (memoised once per process) and line reading."""

import threading
from pathlib import Path
from typing import List, Optional

import cv2

from ..config import settings

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
