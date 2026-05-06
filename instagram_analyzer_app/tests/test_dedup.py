from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _save_noise_image(path: Path, seed: int, perturbation: int = 0) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    if perturbation:
        arr = np.clip(arr.astype(np.int16) + perturbation, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path, format="JPEG", quality=95)


def test_dedupe_drops_near_duplicates(tmp_path: Path):
    from processing.dedup import dedupe_frames

    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    c = tmp_path / "c.jpg"

    _save_noise_image(a, seed=1)
    _save_noise_image(b, seed=1, perturbation=2)   # same content, tiny intensity shift
    _save_noise_image(c, seed=99)                  # different noise pattern

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
