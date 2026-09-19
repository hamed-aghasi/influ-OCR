"""Scroll-aware frame selection, on synthetic screen recordings whose true
scroll offsets are known: a tall textured page viewed through a phone-shaped
window with a pinned header and nav bar, like Instagram Insights."""

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

W, H = 320, 720          # extracted frame size (portrait, after the 720p downscale)
HEADER, NAV = 90, 50     # pinned rows
VIEW = H - HEADER - NAV  # scrolling window


def _page(seed: int, height: int = 2600) -> np.ndarray:
    """Distinct blocks and text-like marks, so every offset is locatable."""
    rng = np.random.default_rng(seed)
    page = np.full((height, W, 3), 245, np.uint8)
    y = 0
    while y < height:
        h = int(rng.integers(40, 120))
        shade = int(rng.integers(120, 230))
        cv2.rectangle(page, (10, y + 4), (W - 10, y + h - 4), (shade, shade, shade), -1)
        for _ in range(int(rng.integers(2, 6))):
            x0 = int(rng.integers(20, W - 80))
            cv2.putText(page, str(int(rng.integers(0, 99999))), (x0, y + int(rng.integers(20, max(21, h - 8)))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
        y += h
    return page


def _frame(page: np.ndarray, y: int, dot=None) -> np.ndarray:
    img = np.empty((H, W, 3), np.uint8)
    img[:HEADER] = 30
    cv2.putText(img, "Reel insights", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
    img[HEADER:H - NAV] = page[y:y + VIEW]
    img[H - NAV:] = 60
    if dot is not None:  # a screen-recorder touch indicator
        cv2.circle(img, dot, 12, (128, 128, 128), -1)
    return img


def _write(tmp: Path, frames) -> list:
    paths = []
    for i, img in enumerate(frames):
        p = tmp / f"frame_{i * 3:06d}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(p)
    return paths


def _covered(offsets_of_selected, offsets_of_good):
    """Every scrolled row shown by some good frame is shown by a selected frame."""
    need = set()
    for y in offsets_of_good:
        need.update(range(y, y + VIEW))
    have = set()
    for y in offsets_of_selected:
        have.update(range(y, y + VIEW))
    return need <= have


# ----- alignment -----

def test_vertical_shift_recovers_known_scroll(tmp_path):
    from processing.frame_selection import ALIGN_WIDTH, _load_gray, vertical_shift

    page = _page(1)
    a, b = _write(tmp_path, [_frame(page, 200), _frame(page, 260)])
    dy = vertical_shift(_load_gray(a), _load_gray(b))
    assert dy is not None
    assert abs(dy - 60 * ALIGN_WIDTH / W) <= 1


def test_vertical_shift_static_and_unrelated(tmp_path):
    from processing.frame_selection import _load_gray, vertical_shift

    a, b, c = _write(tmp_path, [_frame(_page(1), 0), _frame(_page(1), 0), _frame(_page(2), 0)])
    assert vertical_shift(_load_gray(a), _load_gray(b)) == 0
    assert vertical_shift(_load_gray(a), _load_gray(c)) is None


# ----- selection -----

def test_slow_scroll_collapses_to_a_few_covering_frames(tmp_path):
    from processing.dedup import dedupe_frames
    from processing.frame_selection import select_by_scroll

    page = _page(3)
    offsets = list(range(0, 1900, 18))
    paths = _write(tmp_path, [_frame(page, y) for y in offsets])
    sel = select_by_scroll([paths], paths)

    phash_unique, _ = dedupe_frames(paths)
    assert len(sel.selected) <= 6 < len(phash_unique)
    chosen = [offsets[paths.index(p)] for p in sel.selected]
    assert _covered(chosen, offsets)
    assert sel.pages == 1


def test_bad_frames_carry_the_scroll_but_are_never_sent(tmp_path):
    from processing.frame_selection import select_by_scroll

    page = _page(4)
    offsets = list(range(0, 1500, 15))
    paths = _write(tmp_path, [_frame(page, y) for y in offsets])
    good = paths[:5] + paths[-5:]          # only the two ends are Insights-worthy
    sel = select_by_scroll([paths], good)

    assert set(sel.selected) <= set(good)
    assert sel.pages == 1                  # one page: the middle linked the ends
    assert _covered([offsets[paths.index(p)] for p in sel.selected],
                    [offsets[paths.index(p)] for p in good])
    # a 60px wiggle at each end shows 60px of new rows: two frames per end, plus
    # at most one the content check adds when alignment rounding lands on an edge
    assert len(sel.selected) <= 5


def test_tab_switch_starts_a_new_page_and_both_are_covered(tmp_path):
    from processing.frame_selection import select_by_scroll

    a, b = _page(5), _page(6)
    frames = [_frame(a, y) for y in range(0, 300, 20)] + [_frame(b, y) for y in range(0, 300, 20)]
    paths = _write(tmp_path, frames)
    sel = select_by_scroll([paths], paths)

    assert sel.pages == 2
    assert any(p in sel.selected for p in paths[:15])
    assert any(p in sel.selected for p in paths[15:])


def test_scrolling_back_to_the_top_does_not_resend_it(tmp_path):
    """Down, then a jump straight back to the top (no frames in between):
    the revisit must link to the page already seen."""
    from processing.frame_selection import select_by_scroll

    page = _page(7)
    offsets = list(range(0, 900, 20)) + [0, 0, 0]
    paths = _write(tmp_path, [_frame(page, y) for y in offsets])
    sel = select_by_scroll([paths], paths)

    assert sel.pages == 1
    tops = [p for p in sel.selected if offsets[paths.index(p)] == 0]
    assert len(tops) == 1


def test_held_frame_wins_the_tie(tmp_path):
    """Same scroll position throughout; a touch dot moves around except while
    the recorder holds still. The held frame is the one sent."""
    from processing.frame_selection import select_by_scroll

    page = _page(8)
    frames = [_frame(page, 400, dot=(60, 300)), _frame(page, 400, dot=(160, 420)),
              _frame(page, 400), _frame(page, 400), _frame(page, 400),
              _frame(page, 400, dot=(250, 500))]
    paths = _write(tmp_path, frames)
    sel = select_by_scroll([paths], paths)

    assert len(sel.selected) == 1
    assert sel.selected[0] in paths[2:5]
    assert sel.selected[0] == paths[3]     # neighbours on both sides identical


def test_every_good_frame_is_accounted_for(tmp_path):
    from processing.frame_selection import select_by_scroll

    page = _page(9)
    paths = _write(tmp_path, [_frame(page, y) for y in range(0, 1200, 25)])
    sel = select_by_scroll([paths], paths)

    assert set(sel.selected) | set(sel.duplicates) == set(paths)
    assert not set(sel.selected) & set(sel.duplicates)
    grouped = [m for members in sel.groups.values() for m in members]
    assert sorted(grouped) == sorted(paths)


def test_sequences_are_aligned_independently(tmp_path):
    """Two videos in one ZIP never share a page, even when identical."""
    from processing.frame_selection import select_by_scroll

    page = _page(10)
    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    v1 = _write(tmp_path / "v1", [_frame(page, y) for y in range(0, 200, 20)])
    v2 = _write(tmp_path / "v2", [_frame(_page(11), y) for y in range(0, 200, 20)])
    sel = select_by_scroll([v1, v2], v1 + v2)
    assert sel.pages == 2
    assert any(p in sel.selected for p in v1) and any(p in sel.selected for p in v2)


def test_unreadable_good_frame_is_kept_not_dropped(tmp_path):
    from processing.frame_selection import select_by_scroll

    page = _page(12)
    paths = _write(tmp_path, [_frame(page, y) for y in range(0, 200, 20)])
    broken = tmp_path / "frame_999999.jpg"
    broken.write_bytes(b"not a jpeg")
    sel = select_by_scroll([paths + [broken]], paths + [broken])
    assert broken in sel.selected


def test_lone_image_upload_passes_through(tmp_path):
    from processing.frame_selection import select_frames

    (p,) = _write(tmp_path, [_frame(_page(13), 0)])
    sel = select_frames([[p]], [p], strategy="scroll")
    assert sel.selected == [p]


def test_phash_strategy_is_the_old_dedup(tmp_path):
    from processing.dedup import dedupe_frames
    from processing.frame_selection import select_frames

    page = _page(14)
    paths = _write(tmp_path, [_frame(page, y) for y in range(0, 600, 30)])
    sel = select_frames([paths], paths, strategy="phash")
    unique, dupes = dedupe_frames(paths)
    assert (sel.selected, sel.duplicates, sel.strategy) == (unique, dupes, "phash")


def test_unknown_strategy_is_rejected(tmp_path):
    from processing.frame_selection import select_frames

    with pytest.raises(ValueError):
        select_frames([[]], [], strategy="magic")


# ----- wiring -----

def test_run_job_passes_each_video_as_its_own_sequence(tmp_path, monkeypatch):
    import tasks
    from processing.frame_selection import Selection

    frames = [tmp_path / f"frame_{i:06d}.jpg" for i in range(4)]
    seen = {}

    monkeypatch.setattr(tasks, "update_job_status", lambda *a, **k: True)
    monkeypatch.setattr(tasks, "update_job_progress", lambda *a, **k: True)
    monkeypatch.setattr(tasks.settings, "processing_dir", tmp_path / "processing")
    monkeypatch.setattr(tasks, "extract_frames_from_video", lambda *a, **k: (4, frames))
    monkeypatch.setattr(tasks, "classify_frames", lambda *a, **k: {
        "good_paths": frames[1:3], "bad_frames": [{}, {}],
        "good_frames": [{"path": str(frames[1]), "confidence": 0.99},
                        {"path": str(frames[2]), "confidence": 0.97}],
    })

    def fake_select(sequences, good, scores):
        seen.update(sequences=sequences, good=good, scores=scores)
        return Selection(selected=[], duplicates=list(good))

    import processing.selection_pipeline as pipeline
    monkeypatch.setattr(pipeline, "select_frames", fake_select)
    monkeypatch.setattr(pipeline.settings, "coverage_selection", False)
    monkeypatch.setattr(tasks.finalize_job, "delay", lambda *a, **k: None)

    tasks.run_job("j-seq", str(tmp_path / "v.mp4"), "video")

    assert seen["sequences"] == [frames]           # all frames, bad ones included, in order
    assert seen["good"] == frames[1:3]
    assert seen["scores"] == {frames[1]: 0.99, frames[2]: 0.97}


# ----- duplicate content after coverage (round-01 review, 8102e03d) -----

def test_frame_whose_content_is_all_in_another_kept_frame_is_dropped(tmp_path):
    """Two frames of the same page, one scrolled 30 px: every row of the first
    is readable in the second only if the second's window contains it. Here
    frame B (y=60) sits inside A's window (y=40) except its last 20 rows, so
    neither alone covers the other; frame C (y=40) is an exact duplicate of A."""
    from processing.frame_selection import drop_duplicate_content, select_by_scroll

    page = _page(21)
    offsets = [0, 20, 40, 40, 40]
    paths = _write(tmp_path, [_frame(page, y) for y in offsets])
    sel = select_by_scroll([paths], paths)
    kept, dropped = drop_duplicate_content(sel, [paths[2], paths[4]])
    assert len(kept) == 1 and len(dropped) == 1


def test_different_content_is_never_dropped(tmp_path):
    from processing.frame_selection import drop_duplicate_content, select_by_scroll

    page = _page(22)
    offsets = list(range(0, 1200, 30))
    paths = _write(tmp_path, [_frame(page, y) for y in offsets])
    sel = select_by_scroll([paths], paths)
    far_apart = [paths[0], paths[-1]]
    assert drop_duplicate_content(sel, far_apart) == (far_apart, [])


def test_duplicate_pass_keeps_the_better_ranked_frame(tmp_path):
    from processing.frame_selection import drop_duplicate_content, select_by_scroll

    page = _page(23)
    paths = _write(tmp_path, [_frame(page, 200) for _ in range(4)])
    sel = select_by_scroll([paths], paths)
    kept, dropped = drop_duplicate_content(sel, [paths[1], paths[3]], rank={paths[1]: 5, paths[3]: 1})
    assert kept == [paths[1]] and dropped == [paths[3]]


def test_pipeline_runs_coverage_then_the_duplicate_pass(tmp_path, monkeypatch):
    import dataclasses

    import processing.selection_pipeline as pipeline
    from processing.frame_selection import select_by_scroll
    from processing.metric_coverage import Coverage

    page = _page(24)
    paths = _write(tmp_path, [_frame(page, 300) for _ in range(3)])

    def two_identical_picks(seqs, good, scores):
        # scroll selection itself would collapse identical frames to one; force two
        # through so the duplicate pass has something to do
        return dataclasses.replace(select_by_scroll(seqs, good, scores), selected=[paths[0], paths[2]])

    def keep_all(ps, job_id=None):
        return Coverage(selected=list(ps), dropped=[], metrics={p: {"views"} for p in ps},
                        reasons={p: "cover" for p in ps})

    monkeypatch.setattr(pipeline, "select_frames", two_identical_picks)
    monkeypatch.setattr(pipeline, "select_by_coverage", keep_all)
    monkeypatch.setattr(pipeline.settings, "coverage_selection", True)

    choice = pipeline.choose_ocr_frames([paths], paths)
    assert len(choice.final) == 1 and len(choice.dropped_as_duplicate) == 1
    assert choice.coverage.reasons[choice.dropped_as_duplicate[0]] == "duplicate"
