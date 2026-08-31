"""Frame extraction: sampling, 720p conversion, and ZIP handling.

Videos here are generated with cv2.VideoWriter rather than mocked, so the
sampling loop runs against real decoded frames.
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def _write_video(path: Path, frames: int = 9, size=(64, 48)) -> Path:
    """A real, decodable video. Each frame is a distinct flat grey."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, size)
    if not writer.isOpened():
        pytest.skip("no MJPG encoder available in this OpenCV build")
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), (i * 20) % 256, np.uint8))
    writer.release()
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("VideoWriter produced no output in this environment")
    return path


# ----- filename sanitising -----

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("normal_name", "normal_name"),
        ("  padded  ", "padded"),
        ("...dots...", "dots"),
        ('bad<>:"|?*chars', "bad_______chars"),
        ("tab\tand\nnewline", "tabandnewline"),
    ],
)
def test_sanitize_filename(raw, expected):
    from processing.frame_extractor import _sanitize_filename

    assert _sanitize_filename(raw) == expected


# ----- frame sampling -----

def test_extracts_every_nth_frame(tmp_path):
    from processing.frame_extractor import extract_frames_from_video

    video = _write_video(tmp_path / "clip.avi", frames=9)
    out = tmp_path / "frames"

    count, paths = extract_frames_from_video(video, out, frame_interval=3, convert_to_720=False)

    assert count == 3           # frames 0, 3, 6 of 9
    assert len(paths) == 3
    assert [p.name for p in paths] == [
        "frame_000000.jpg", "frame_000003.jpg", "frame_000006.jpg",
    ]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_interval_of_one_keeps_every_frame(tmp_path):
    from processing.frame_extractor import extract_frames_from_video

    video = _write_video(tmp_path / "clip.avi", frames=5)
    count, paths = extract_frames_from_video(
        video, tmp_path / "frames", frame_interval=1, convert_to_720=False
    )
    assert count == 5
    assert len(paths) == 5


def test_frames_are_readable_jpegs(tmp_path):
    from processing.frame_extractor import extract_frames_from_video

    video = _write_video(tmp_path / "clip.avi", frames=4)
    _, paths = extract_frames_from_video(
        video, tmp_path / "frames", frame_interval=2, convert_to_720=False
    )
    decoded = cv2.imread(str(paths[0]))
    assert decoded is not None
    assert decoded.shape[:2] == (48, 64)


def test_writes_extraction_metadata(tmp_path):
    from processing.frame_extractor import extract_frames_from_video

    video = _write_video(tmp_path / "clip.avi", frames=6)
    out = tmp_path / "frames"
    extract_frames_from_video(video, out, frame_interval=2, convert_to_720=False)

    meta = json.loads((out / "extraction_metadata.json").read_text())
    assert meta["extracted_frames"] == 3
    assert meta["fps"] == pytest.approx(10.0, abs=0.5)
    assert "extraction_date" in meta


def test_stale_frames_are_cleared_before_extraction(tmp_path):
    """Re-running a job must not leave a previous run's frames behind, or the
    classifier would receive images from an unrelated video."""
    from processing.frame_extractor import extract_frames_from_video

    out = tmp_path / "frames"
    out.mkdir()
    stale = out / "frame_999999.jpg"
    stale.write_bytes(b"stale")
    keep = out / "notes.txt"
    keep.write_text("not a frame")

    video = _write_video(tmp_path / "clip.avi", frames=3)
    extract_frames_from_video(video, out, frame_interval=1, convert_to_720=False)

    assert not stale.exists()
    assert keep.exists()  # only frame_*.jpg is cleared


def test_unreadable_video_yields_no_frames(tmp_path):
    from processing.frame_extractor import extract_frames_from_video

    broken = tmp_path / "broken.avi"
    broken.write_bytes(b"this is not a video")

    count, paths = extract_frames_from_video(
        broken, tmp_path / "frames", frame_interval=1, convert_to_720=False
    )
    assert count == 0
    assert paths == []


# ----- 720p conversion -----

def test_conversion_skipped_when_ffmpeg_missing(tmp_path, monkeypatch):
    from processing import frame_extractor as fe

    monkeypatch.setattr(fe, "_check_ffmpeg", lambda: False)
    assert fe._convert_to_720p(tmp_path / "in.mp4", tmp_path / "out.mp4", fe.job_logger(__name__)) is False


def test_conversion_skipped_for_sources_at_or_below_720p(tmp_path, monkeypatch):
    """The generated clip is 48px tall, so a real VideoCapture reports <720
    and ffmpeg must never be invoked."""
    from processing import frame_extractor as fe

    monkeypatch.setattr(fe, "_check_ffmpeg", lambda: True)
    ran = []
    monkeypatch.setattr(fe.subprocess, "run", lambda *a, **k: ran.append(a) or None)

    video = _write_video(tmp_path / "small.avi", frames=2)
    assert fe._convert_to_720p(video, tmp_path / "out.mp4", fe.job_logger(__name__)) is False
    assert ran == []


def test_conversion_reports_ffmpeg_failure(tmp_path, monkeypatch):
    from processing import frame_extractor as fe

    class _Result:
        returncode = 1
        stderr = "ffmpeg exploded"

    monkeypatch.setattr(fe, "_check_ffmpeg", lambda: True)
    monkeypatch.setattr(fe.cv2, "VideoCapture", lambda *_: _FakeCap(height=1080))
    monkeypatch.setattr(fe.subprocess, "run", lambda *a, **k: _Result())

    assert fe._convert_to_720p(tmp_path / "in.mp4", tmp_path / "out.mp4", fe.job_logger(__name__)) is False


def test_conversion_timeout_is_not_fatal(tmp_path, monkeypatch):
    import subprocess

    from processing import frame_extractor as fe

    monkeypatch.setattr(fe, "_check_ffmpeg", lambda: True)
    monkeypatch.setattr(fe.cv2, "VideoCapture", lambda *_: _FakeCap(height=1080))

    def _timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1)

    monkeypatch.setattr(fe.subprocess, "run", _timeout)
    assert fe._convert_to_720p(tmp_path / "in.mp4", tmp_path / "out.mp4", fe.job_logger(__name__)) is False


def test_check_ffmpeg_handles_missing_binary(monkeypatch):
    from processing import frame_extractor as fe

    def _missing(*a, **k):
        raise FileNotFoundError("no ffmpeg")

    monkeypatch.setattr(fe.subprocess, "run", _missing)
    assert fe._check_ffmpeg() is False


class _FakeCap:
    """Minimal VideoCapture stand-in for the height probe."""

    def __init__(self, height):
        self._height = height

    def get(self, prop):
        return float(self._height)

    def release(self):
        pass


# ----- ZIP handling -----

def _zip_with(tmp_path: Path, members: dict) -> Path:
    z = tmp_path / "campaign.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return z


def test_process_campaign_zip_extracts_each_video(tmp_path):
    from processing.frame_extractor import process_campaign_zip

    video_bytes = _write_video(tmp_path / "src.avi", frames=6).read_bytes()
    z = _zip_with(tmp_path, {"one.avi": video_bytes, "two.avi": video_bytes})

    result = process_campaign_zip(z, tmp_path / "out", job_id="job-1")

    assert len(result["results"]) == 2
    assert all(r["status"] == "success" for r in result["results"])
    assert result["total_frames"] == 4  # 2 videos x frames 0 and 3
    assert len(result["frame_paths"]) == 4
    assert Path(result["summary_path"]).exists()


def test_process_campaign_zip_ignores_macosx_and_non_videos(tmp_path):
    from processing.frame_extractor import process_campaign_zip

    video_bytes = _write_video(tmp_path / "src.avi", frames=3).read_bytes()
    z = _zip_with(
        tmp_path,
        {
            "real.avi": video_bytes,
            "__MACOSX/._real.avi": b"junk",
            "readme.txt": b"notes",
        },
    )

    result = process_campaign_zip(z, tmp_path / "out", job_id="job-2")
    assert [r["name"] for r in result["results"]] == ["real.avi"]


def test_zip_of_screenshots_yields_no_frames(tmp_path):
    """KNOWN GAP, pinned deliberately: README and the upload page both offer
    'a ZIP of screenshots', but process_campaign_zip filters members to video
    extensions only. Such a ZIP produces zero frames and no error, and the job
    then fails as 'No Instagram Insights screens detected'. Change this test
    when image members are supported.
    """
    from processing.frame_extractor import process_campaign_zip

    z = _zip_with(tmp_path, {"shot1.png": b"\x89PNG\r\n\x1a\n", "shot2.jpg": b"\xff\xd8\xff"})
    result = process_campaign_zip(z, tmp_path / "out", job_id="job-3")

    assert result["results"] == []
    assert result["frame_paths"] == []
    assert result["total_frames"] == 0


def test_corrupt_member_is_reported_not_raised(tmp_path):
    from processing.frame_extractor import process_campaign_zip

    z = _zip_with(tmp_path, {"broken.avi": b"not really a video"})
    result = process_campaign_zip(z, tmp_path / "out", job_id="job-4")

    # A member that decodes to nothing still completes: zero frames, no crash.
    assert len(result["results"]) == 1
    assert result["total_frames"] == 0


def test_zip_slip_member_cannot_escape_the_temp_dir(tmp_path):
    """zipfile.extract() sanitises member paths; this pins that guarantee so a
    future switch to a manual open()/write() loop cannot silently reintroduce
    path traversal."""
    from processing.frame_extractor import process_campaign_zip

    video_bytes = _write_video(tmp_path / "src.avi", frames=2).read_bytes()
    z = _zip_with(tmp_path, {"../../escaped.avi": video_bytes})

    marker = tmp_path.parent / "escaped.avi"
    existed_before = marker.exists()

    process_campaign_zip(z, tmp_path / "out", job_id="job-5")

    assert marker.exists() == existed_before  # nothing written outside the sandbox


def test_campaign_summary_records_counts(tmp_path):
    from processing.frame_extractor import process_campaign_zip

    video_bytes = _write_video(tmp_path / "src.avi", frames=3).read_bytes()
    z = _zip_with(tmp_path, {"a.avi": video_bytes, "bad.avi": b"junk"})

    result = process_campaign_zip(z, tmp_path / "out", job_id="job-6")
    summary = json.loads(Path(result["summary_path"]).read_text())

    assert summary["job_id"] == "job-6"
    assert summary["total_videos"] == 2
    assert summary["successful"] == 2  # both "succeed"; the junk one yields 0 frames
    assert len(summary["results"]) == 2
