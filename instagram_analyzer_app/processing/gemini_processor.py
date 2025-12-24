"""
Gemini OCR Processing Module

Extracts Instagram metrics from frames using Gemini API via OpenRouter.
"""

import base64
import json
import requests
from pathlib import Path
from datetime import datetime
import time
import os
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum

from .logger import setup_logger, get_logger
from .s3_storage import upload_json, is_s3_configured

logger = get_logger(__name__)

# API Configuration
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3-flash-preview"
BATCH_SIZE = 50
DELAY_BETWEEN_REQUESTS = 2
MAX_RETRIES = 5


# Pydantic Models (simplified)
class Metrics(BaseModel):
    views: Optional[int] = None
    followers: Optional[float] = None
    non_followers: Optional[float] = None
    accounts_reached: Optional[int] = None
    interactions: Optional[int] = None
    likes: Optional[int] = None
    replies: Optional[int] = None
    shares: Optional[int] = None
    links_clicks: Optional[int] = None
    sticker_taps: Optional[int] = None
    navigation: Optional[int] = None
    forward: Optional[int] = None
    next_story: Optional[int] = None
    back: Optional[int] = None
    exited: Optional[int] = None
    profile_activity: Optional[int] = None
    profile_visits: Optional[int] = None
    external_link_taps: Optional[int] = None
    follows: Optional[int] = None


class Metadata(BaseModel):
    language: Optional[str] = None
    date_range: Optional[str] = None
    content_type: Optional[str] = None


class FrameResult(BaseModel):
    frame_index: int = Field(ge=0)
    is_duplicate: bool = False
    duplicate_of_frame: Optional[int] = None
    metrics: Optional[Metrics] = None
    metadata: Optional[Metadata] = None
    actual_frame: Optional[str] = None


EXTRACTION_PROMPT = """
You are analyzing Instagram Insights screenshots. Identify UNIQUE data vs duplicates.

LANGUAGE: Screenshots may be in Persian or English. Persian numbers: ۰-۹ = 0-9

METRICS TO EXTRACT:
- views, followers, non_followers, accounts_reached
- interactions, likes, replies, shares
- links_clicks, sticker_taps, navigation
- forward, next_story, back, exited
- profile_activity, profile_visits, external_link_taps, follows

OUTPUT FORMAT (JSON array):
[
  {
    "frame_index": 0,
    "is_duplicate": false,
    "metrics": {"views": 1234, "likes": 30, ...},
    "metadata": {"language": "fa", "content_type": "story"}
  },
  {
    "frame_index": 1,
    "is_duplicate": true,
    "duplicate_of_frame": 0
  }
]

Return ONLY valid JSON.
"""


def encode_images_batch(image_paths: List[Path]) -> List[Tuple[str, str]]:
    """Convert batch of images to base64."""
    encoded = []
    for path in image_paths:
        try:
            with open(path, "rb") as f:
                encoded.append((path.name, base64.b64encode(f.read()).decode('utf-8')))
        except Exception as e:
            logger.error(f"Failed to encode {path}: {e}")
    return encoded


def call_gemini_api(encoded_images: List[Tuple[str, str]], api_key: str) -> Optional[List[Dict]]:
    """Send batch of images to Gemini API."""
    if not api_key:
        logger.error("API key is empty or None!")
        return None

    logger.info(f"API key present: {api_key[:20]}...{api_key[-10:]}")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    content = [{"type": "text", "text": EXTRACTION_PROMPT}]
    for i, (frame_name, base64_img) in enumerate(encoded_images):
        content.append({"type": "text", "text": f"\nFrame {i}: {frame_name}"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})

    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        "max_tokens": 4000
    }

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Sending batch of {len(encoded_images)} images to API...")
            response = requests.post(API_URL, headers=headers, json=data, timeout=60)

            if response.status_code == 200:
                content_text = response.json()['choices'][0]['message']['content']

                # Clean markdown
                if "```json" in content_text:
                    content_text = content_text.split("```json")[1].split("```")[0]
                elif "```" in content_text:
                    content_text = content_text.split("```")[1].split("```")[0]

                try:
                    raw_data = json.loads(content_text.strip())
                    validated = []
                    for frame_data in raw_data:
                        try:
                            validated.append(FrameResult(**frame_data).dict())
                        except:
                            validated.append(frame_data)
                    if validated:
                        logger.info(f"Validated {len(validated)} frames")
                        return validated
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {e}")

            elif response.status_code == 429:
                wait_time = (attempt + 1) * 10
                logger.warning(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"API error {response.status_code}: {response.text[:500]}")

        except requests.exceptions.Timeout:
            logger.error(f"Attempt {attempt + 1}: Request timed out")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Attempt {attempt + 1}: Connection error - {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)

    logger.error("All API attempts failed - returning None")
    return None


def aggregate_metrics(unique_results: List[Dict]) -> Dict:
    """Aggregate metrics from unique results."""
    if not unique_results:
        return {}

    metric_names = [
        "views", "followers", "non_followers", "accounts_reached",
        "interactions", "likes", "replies", "shares", "follows",
        "profile_visits", "links_clicks", "sticker_taps", "navigation",
        "forward", "next_story", "back", "exited", "profile_activity",
        "external_link_taps"
    ]

    summary = {}
    for metric in metric_names:
        values = [r["metrics"][metric] for r in unique_results
                  if r.get("metrics") and r["metrics"].get(metric) is not None]
        if values:
            summary[metric] = {"max": max(values), "min": min(values), "avg": sum(values)/len(values), "last": values[-1]}

    return summary


def process_frames(
    frame_paths: List[Path],
    output_dir: Optional[Path] = None,
    job_id: Optional[str] = None
) -> Dict:
    """Process frames to extract Instagram metrics using Gemini."""
    global logger
    if job_id:
        logger = setup_logger(__name__, job_id)

    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set")
        return {"error": "API key not configured", "total_frames": len(frame_paths)}

    logger.info(f"Processing {len(frame_paths)} frames for OCR")

    all_results, unique_results = [], []

    for batch_start in range(0, len(frame_paths), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(frame_paths))
        batch_frames = frame_paths[batch_start:batch_end]

        logger.info(f"Processing batch: frames {batch_start}-{batch_end-1}")

        encoded_batch = encode_images_batch(batch_frames)
        if not encoded_batch:
            logger.warning(f"No images encoded for batch {batch_start}-{batch_end-1}")
            continue

        logger.info(f"Calling Gemini API with {len(encoded_batch)} encoded images...")
        batch_results = call_gemini_api(encoded_batch, api_key)
        logger.info(f"API returned: {len(batch_results) if batch_results else 0} results")

        if batch_results:
            for result in batch_results:
                if result.get("frame_index") is not None and result["frame_index"] < len(batch_frames):
                    result["actual_frame"] = batch_frames[result["frame_index"]].name
                all_results.append(result)
                if not result.get("is_duplicate", False):
                    unique_results.append(result)

        if batch_end < len(frame_paths):
            time.sleep(DELAY_BETWEEN_REQUESTS)

    final_metrics = {
        "extraction_date": datetime.now().isoformat(),
        "total_frames": len(frame_paths),
        "unique_frames": len(unique_results),
        "duplicate_frames": len(all_results) - len(unique_results),
        "all_frames_data": all_results,
        "unique_metrics": unique_results,
        "summary": aggregate_metrics(unique_results)
    }

    # Save locally
    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        with open(output_dir / "instagram_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(final_metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metrics to: {output_dir / 'instagram_metrics.json'}")

    # Upload to S3 if configured
    if job_id and is_s3_configured():
        if upload_json(job_id, final_metrics):
            logger.info(f"Uploaded metrics to S3 for job {job_id}")
        else:
            logger.warning(f"Failed to upload metrics to S3 for job {job_id}")

    logger.info(f"Extraction complete: {len(unique_results)} unique metrics found")
    return final_metrics


def extract_metrics_from_good_frames(
    good_folder: Path,
    output_dir: Optional[Path] = None,
    job_id: Optional[str] = None
) -> Dict:
    """Extract metrics from a folder containing good frames."""
    frames = sorted(list(good_folder.glob("frame_*.jpg")))
    if not frames:
        logger.warning(f"No frames found in {good_folder}")
        return {"error": "No frames found", "total_frames": 0}
    return process_frames(frames, output_dir, job_id)
