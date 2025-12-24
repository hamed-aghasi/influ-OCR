"""
Frame Extraction Module

Extracts frames from video files and ZIP archives.
Supports 720p conversion for faster processing.
"""

import zipfile
import cv2
import os
from pathlib import Path
from datetime import datetime
import json
import tempfile
import subprocess
from typing import Dict, List, Optional, Tuple

from .logger import setup_logger, get_logger

logger = get_logger(__name__)


def sanitize_filename(name: str) -> str:
    """Sanitize filename for Windows filesystem."""
    name = name.strip(' .')
    invalid_chars = '<>:"|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    name = ''.join(char for char in name if ord(char) >= 32)
    return name


def check_ffmpeg() -> bool:
    """Check if ffmpeg is installed."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def convert_to_720p(input_path: Path, output_path: Path) -> bool:
    """Convert video to 720p using FFmpeg."""
    try:
        if not check_ffmpeg():
            logger.warning("FFmpeg not available, skipping 720p conversion")
            return False

        cap = cv2.VideoCapture(str(input_path))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if original_height <= 720:
            logger.info(f"Video is already {original_height}p, skipping conversion")
            return False

        cmd = [
            'ffmpeg', '-y', '-i', str(input_path),
            '-vf', 'scale=-2:720',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'copy', str(output_path)
        ]

        logger.info(f"Converting video to 720p...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("720p conversion successful")
            return True
        else:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"720p conversion failed: {e}")
        return False


def extract_frames_from_video(
    video_path: Path,
    output_folder: Path,
    frame_interval: int = 3,
    convert_to_720: bool = True
) -> Tuple[int, List[Path]]:
    """
    Extract frames from a video file.

    Args:
        video_path: Path to the video file
        output_folder: Where to save extracted frames
        frame_interval: Save every Nth frame (default: 3)
        convert_to_720: Whether to convert video to 720p before extraction

    Returns:
        Tuple of (frame_count, list_of_frame_paths)
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    # Clear existing frames
    for old_frame in output_folder.glob("frame_*.jpg"):
        old_frame.unlink()

    # Convert to 720p if needed
    video_to_process = video_path
    temp_720p_video = None

    if convert_to_720:
        temp_720p_video = output_folder / f"temp_720p_{video_path.name}"
        if convert_to_720p(video_path, temp_720p_video):
            video_to_process = temp_720p_video
        elif temp_720p_video.exists():
            temp_720p_video.unlink()
            temp_720p_video = None

    # Extract frames
    cap = cv2.VideoCapture(str(video_to_process))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    saved_count = 0
    frame_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0 and frame is not None and frame.size > 0:
            frame_path = output_folder / f"frame_{frame_count:06d}.jpg"
            try:
                success, encoded_img = cv2.imencode('.jpg', frame)
                if success:
                    with open(frame_path, 'wb') as f:
                        f.write(encoded_img.tobytes())
                    saved_count += 1
                    frame_paths.append(frame_path)
            except Exception as e:
                logger.error(f"Failed to save frame: {e}")

        frame_count += 1

    cap.release()

    # Cleanup temp video
    if temp_720p_video and temp_720p_video.exists():
        try:
            temp_720p_video.unlink()
        except:
            pass

    # Save metadata
    metadata = {
        "total_frames": total_frames,
        "extracted_frames": saved_count,
        "fps": fps,
        "extraction_date": datetime.now().isoformat()
    }
    with open(output_folder / "extraction_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Extracted {saved_count} frames from {video_path.name}")
    return saved_count, frame_paths


def process_video_from_zip(
    zip_path: Path,
    video_name: str,
    video_index: int,
    output_base: Path,
    campaign_name: Optional[str] = None
) -> Dict:
    """Extract frames from a video in a ZIP file."""
    if not campaign_name:
        campaign_name = sanitize_filename(zip_path.stem)

    output_folder = output_base / campaign_name / f"{campaign_name}_{video_index}"

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            logger.info(f"Extracting {video_name} from zip")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extract(video_name, temp_path)

            extracted_video = temp_path / video_name
            frame_count, frame_paths = extract_frames_from_video(extracted_video, output_folder)

            return {
                "type": "video",
                "name": video_name,
                "status": "success",
                "frames": frame_count,
                "frame_paths": [str(p) for p in frame_paths],
                "output": str(output_folder)
            }

    except Exception as e:
        logger.error(f"Error processing {video_name}: {e}")
        return {
            "type": "video",
            "name": video_name,
            "status": "error",
            "error": str(e),
            "frames": 0,
            "output": None
        }


def process_campaign_zip(
    zip_path: Path,
    output_base: Optional[Path] = None,
    job_id: Optional[str] = None
) -> Dict:
    """Process entire campaign ZIP file."""
    global logger
    if job_id:
        logger = setup_logger(__name__, job_id)

    if output_base is None:
        output_base = Path(os.getenv('OUTPUT_BASE', '/tmp/extracted_frames'))

    original_name = zip_path.stem
    campaign_name = job_id if job_id else sanitize_filename(original_name)

    logger.info(f"Processing campaign: {campaign_name}")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        all_files = zf.namelist()
        video_files = [f for f in all_files
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
                      and not f.startswith('__MACOSX')]

    logger.info(f"Found {len(video_files)} videos")

    results = []
    all_frame_paths = []

    for idx, video_name in enumerate(video_files, 1):
        logger.info(f"Processing video {idx}/{len(video_files)}: {video_name}")
        result = process_video_from_zip(zip_path, video_name, idx, output_base, campaign_name)
        results.append(result)
        if result["status"] == "success":
            all_frame_paths.extend(result.get("frame_paths", []))

    # Save summary
    summary_path = output_base / campaign_name / "campaign_summary.json"
    summary_path.parent.mkdir(exist_ok=True, parents=True)

    summary_data = {
        "campaign": campaign_name,
        "original_filename": original_name,
        "job_id": job_id,
        "processed_date": datetime.now().isoformat(),
        "total_videos": len(video_files),
        "successful": len([r for r in results if r["status"] == "success"]),
        "results": results
    }
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Campaign extraction complete: {campaign_name}")

    return {
        "results": results,
        "frame_paths": all_frame_paths,
        "summary_path": str(summary_path),
        "total_frames": sum(r.get("frames", 0) for r in results if r["status"] == "success"),
        "output_directory": str(output_base / campaign_name)
    }
