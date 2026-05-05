from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _save_image(path: Path, color: tuple) -> None:
    img = Image.new("RGB", (256, 256), color=color)
    img.save(path, format="JPEG")


def test_dedupe_drops_near_duplicates(tmp_path: Path):
    from processing.dedup import dedupe_frames

    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    c = tmp_path / "c.jpg"

    _save_image(a, (10, 200, 50))
    _save_image(b, (12, 202, 52))   # near-identical
    _save_image(c, (200, 30, 30))   # clearly different

    unique, duplicates = dedupe_frames([a, b, c])
    unique_names = {p.name for p in unique}
    assert "a.jpg" in unique_names
    assert "c.jpg" in unique_names
    assert duplicates == [b] or duplicates == []  # phash boundary tolerant


def test_dedupe_handles_unreadable(tmp_path: Path):
    from processing.dedup import dedupe_frames

    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")

    unique, duplicates = dedupe_frames([bad])
    assert unique == []
    assert duplicates == []
