"""Perceptual-hash deduplication of frames.

Run before OCR to drop visually-near-identical frames. Cheaper and more
deterministic than asking the LLM to find duplicates.
"""

from pathlib import Path
from typing import Iterable, List, Tuple

import imagehash
from PIL import Image

from .config import settings
from .logger import get_logger


logger = get_logger(__name__)


def _phash(path: Path) -> imagehash.ImageHash:
    with Image.open(path) as img:
        return imagehash.phash(img, hash_size=settings.perceptual_hash_size)


def dedupe_frames(
    paths: Iterable[Path],
    distance_threshold: int = settings.perceptual_hash_distance,
) -> Tuple[List[Path], List[Path]]:
    """Return (unique, duplicates). A frame is a duplicate if its phash is
    within `distance_threshold` Hamming distance of any frame already kept.
    """
    unique: List[Tuple[Path, imagehash.ImageHash]] = []
    duplicates: List[Path] = []

    for path in paths:
        try:
            current = _phash(path)
        except Exception as exc:  # noqa: BLE001 - want to keep going on bad frames
            logger.warning("phash failed for %s: %s", path.name, exc)
            continue

        if any((current - kept) <= distance_threshold for _, kept in unique):
            duplicates.append(path)
        else:
            unique.append((path, current))

    return [p for p, _ in unique], duplicates
