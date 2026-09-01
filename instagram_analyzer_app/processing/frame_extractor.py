"""Frame extraction: ZIP+video → JPEG frames on disk."""

import json
import os
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from .config import settings
from .logger import job_logger


def _sanitize_filename(name: str) -> str:
    name = name.strip(" .")
    for char in '<>:"|?*':
        name = name.replace(char, "_")
    return "".join(c for c in name if ord(c) >= 32)


def _check_ffmpeg() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _convert_to_720p(input_path: Path, output_path: Path, log) -> bool:
    if not _check_ffmpeg():
        log.warning("ffmpeg not available, skipping 720p conversion")
        return False

    cap = cv2.VideoCapture(str(input_path))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if height <= 720:
        log.info("Source is %dp; skipping conversion", height)
        return False

    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", "scale=-2:720",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "copy", str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=settings.ffmpeg_timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        log.error("ffmpeg timed out after %ds", settings.ffmpeg_timeout_seconds)
        return False

    if result.returncode != 0:
        log.error("ffmpeg failed: %s", result.stderr[-500:])
        return False
    log.info("720p conversion complete")
    return True


def _extract_with_ffmpeg(video_path: Path, output_folder: Path, do_scale: bool, log) -> Optional[List[Path]]:
    """Single-pass fast path: one ffmpeg decode samples extract_fps frames/sec
    and caps height at 720, replacing the old re-encode-then-decode-every-frame
    flow (which decoded the video twice and saved ~10 frames/sec)."""
    if not _check_ffmpeg():
        return None

    vf = f"fps={settings.extract_fps}"
    if do_scale:
        vf += ",scale=-2:min(720\\,ih)"  # downscale only; -2 keeps width even

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", vf, "-q:v", "3",
        str(output_folder / "frame_%06d.jpg"),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=settings.ffmpeg_timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        log.error("ffmpeg extraction timed out after %ds", settings.ffmpeg_timeout_seconds)
        return None
    if result.returncode != 0:
        log.error("ffmpeg extraction failed: %s", result.stderr[-500:])
        return None
    return sorted(output_folder.glob("frame_*.jpg"))


def _write_metadata(video_path: Path, output_folder: Path, saved_count: int, log) -> None:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    metadata = {
        "total_frames": total_frames,
        "extracted_frames": saved_count,
        "fps": fps,
        "extraction_date": datetime.now().isoformat(),
    }
    try:
        with open(output_folder / "extraction_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    except OSError as exc:
        log.warning("Could not write extraction metadata: %s", exc)


def extract_frames_from_video(
    video_path: Path,
    output_folder: Path,
    job_id: Optional[str] = None,
    frame_interval: Optional[int] = None,
    convert_to_720: Optional[bool] = None,
) -> Tuple[int, List[Path]]:
    log = job_logger(__name__, job_id)
    interval = frame_interval if frame_interval is not None else settings.frame_interval
    do_convert = convert_to_720 if convert_to_720 is not None else settings.convert_to_720p

    output_folder.mkdir(parents=True, exist_ok=True)
    for old in output_folder.glob("frame_*.jpg"):
        try:
            old.unlink()
        except OSError as exc:
            log.warning("Could not remove %s: %s", old.name, exc)

    # Fast path: one ffmpeg decode does sampling + scaling. Explicit
    # frame_interval / convert_to_720 overrides keep the exact every-Nth-frame
    # OpenCV semantics below.
    if frame_interval is None and convert_to_720 is None and settings.extract_fps > 0:
        paths = _extract_with_ffmpeg(video_path, output_folder, do_convert, log)
        if paths is not None:
            _write_metadata(video_path, output_folder, len(paths), log)
            log.info("Extracted %d frames from %s (ffmpeg fast path)", len(paths), video_path.name)
            return len(paths), paths
        for leftover in output_folder.glob("frame_*.jpg"):  # partial ffmpeg output
            try:
                leftover.unlink()
            except OSError:
                pass
        log.warning("ffmpeg fast path unavailable; falling back to OpenCV decode")

    video_to_process = video_path
    temp_video: Optional[Path] = None
    if do_convert:
        temp_video = output_folder / f"temp_720p_{video_path.name}"
        if _convert_to_720p(video_path, temp_video, log):
            video_to_process = temp_video
        elif temp_video.exists():
            try:
                temp_video.unlink()
            except OSError:
                pass
            temp_video = None

    cap = cv2.VideoCapture(str(video_to_process))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    saved_count = 0
    saved_paths: List[Path] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0 and frame is not None and frame.size > 0:
            path = output_folder / f"frame_{frame_count:06d}.jpg"
            success, encoded = cv2.imencode(".jpg", frame)
            if success:
                try:
                    with open(path, "wb") as f:
                        f.write(encoded.tobytes())
                    saved_count += 1
                    saved_paths.append(path)
                except OSError as exc:
                    log.error("Failed to save frame: %s", exc)
        frame_count += 1
    cap.release()

    if temp_video and temp_video.exists():
        try:
            temp_video.unlink()
        except OSError:
            pass

    metadata = {
        "total_frames": total_frames,
        "extracted_frames": saved_count,
        "fps": fps,
        "extraction_date": datetime.now().isoformat(),
    }
    try:
        with open(output_folder / "extraction_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    except OSError as exc:
        log.warning("Could not write extraction metadata: %s", exc)

    log.info("Extracted %d frames from %s", saved_count, video_path.name)
    return saved_count, saved_paths


def _process_video_from_zip(
    zip_path: Path,
    video_name: str,
    video_index: int,
    output_base: Path,
    campaign_name: str,
    job_id: Optional[str],
) -> Dict:
    log = job_logger(__name__, job_id)
    output_folder = output_base / campaign_name / f"{campaign_name}_{video_index}"
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extract(video_name, temp_dir)
            extracted = Path(temp_dir) / video_name
            count, paths = extract_frames_from_video(extracted, output_folder, job_id=job_id)
            return {
                "type": "video",
                "name": video_name,
                "status": "success",
                "frames": count,
                "frame_paths": [str(p) for p in paths],
                "output": str(output_folder),
            }
    except (zipfile.BadZipFile, OSError) as exc:
        log.error("Error processing %s: %s", video_name, exc)
        return {
            "type": "video",
            "name": video_name,
            "status": "error",
            "error": str(exc),
            "frames": 0,
            "output": None,
        }


def process_campaign_zip(
    zip_path: Path,
    output_base: Optional[Path] = None,
    job_id: Optional[str] = None,
) -> Dict:
    log = job_logger(__name__, job_id)
    if output_base is None:
        output_base = Path(os.getenv("OUTPUT_BASE", "/tmp/extracted_frames"))

    campaign_name = job_id if job_id else _sanitize_filename(zip_path.stem)

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_files = zf.namelist()
        videos = [
            f
            for f in all_files
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
            and not f.startswith("__MACOSX")
        ]

    log.info("ZIP contains %d videos", len(videos))

    results = []
    all_paths: List[str] = []
    for idx, video_name in enumerate(videos, 1):
        log.info("Processing video %d/%d: %s", idx, len(videos), video_name)
        result = _process_video_from_zip(zip_path, video_name, idx, output_base, campaign_name, job_id)
        results.append(result)
        if result["status"] == "success":
            all_paths.extend(result.get("frame_paths", []))

    summary_path = output_base / campaign_name / "campaign_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "campaign": campaign_name,
                    "original_filename": zip_path.stem,
                    "job_id": job_id,
                    "processed_date": datetime.now().isoformat(),
                    "total_videos": len(videos),
                    "successful": sum(1 for r in results if r["status"] == "success"),
                    "results": results,
                },
                f,
                indent=2,
            )
    except OSError as exc:
        log.warning("Could not write campaign summary: %s", exc)

    return {
        "results": results,
        "frame_paths": all_paths,
        "summary_path": str(summary_path),
        "total_frames": sum(r.get("frames", 0) for r in results if r["status"] == "success"),
        "output_directory": str(output_base / campaign_name),
    }
