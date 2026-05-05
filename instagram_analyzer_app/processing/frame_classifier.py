"""
Frame Classification Module

Classifies frames as 'good' or 'bad' using a pre-trained TensorFlow SavedModel.
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime
import json
import os
from typing import List, Dict, Any, Optional, Tuple

import threading

from .config import settings
from .logger import get_logger, job_logger

logger = get_logger(__name__)

_cached_model = None
_model_lock = threading.Lock()


def load_model():
    """Load the classification model (SavedModel format)."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    with _model_lock:
        if _cached_model is not None:
            return _cached_model

        savedmodel_path = settings.model_dir / 'frame_classifier_savedmodel'
        if not savedmodel_path.exists():
            logger.error(f"SavedModel not found at: {savedmodel_path}")
            return None

        try:
            import keras
            logger.info(f"Loading SavedModel from {savedmodel_path}")
            _cached_model = keras.layers.TFSMLayer(str(savedmodel_path), call_endpoint='serving_default')
            return _cached_model
        except (ImportError, OSError) as e:
            logger.error(f"Failed to load model: {e}")
            return None


def preprocess_image(img: np.ndarray) -> Optional[np.ndarray]:
    """Preprocess image for model prediction."""
    if img is None:
        return None

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < settings.dark_frame_threshold:
            img = cv2.convertScaleAbs(img, alpha=1.5, beta=40)
        size = settings.classifier_image_size
        img_resized = cv2.resize(img, (size, size))
        img_normalized = img_resized.astype(np.float32) / 255.0
        return np.expand_dims(img_normalized, axis=0)
    except cv2.error as e:
        logger.error(f"Preprocessing failed: {e}")
        return None


def classify_frame(frame_path: Path, model: Any = None) -> Tuple[Optional[str], float]:
    """Classify a single frame."""
    if model is None:
        model = load_model()
        if model is None:
            return None, 0.0

    try:
        # Read image
        with open(frame_path, 'rb') as f:
            nparr = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return None, 0.0

        processed_img = preprocess_image(img)
        if processed_img is None:
            return None, 0.0

        # Get prediction
        prediction = model(processed_img)

        # Handle TFSMLayer output (returns dict)
        if isinstance(prediction, dict):
            output_key = list(prediction.keys())[0]
            pred_value = prediction[output_key].numpy()
            confidence = float(pred_value[0][0]) if len(pred_value.shape) > 1 else float(pred_value[0])
        else:
            pred_np = prediction.numpy() if hasattr(prediction, 'numpy') else prediction
            confidence = float(pred_np[0][0]) if len(pred_np.shape) > 1 else float(pred_np[0])

        label = "GOOD" if confidence > settings.classifier_threshold else "BAD"
        return label, confidence

    except (OSError, RuntimeError, ValueError) as e:
        logger.error(f"Error processing {frame_path.name}: {e}")
        return None, 0.0


def classify_frames(
    frame_paths: List[Path],
    organize_files: bool = False,
    output_dir: Optional[Path] = None,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    """Classify multiple frames."""
    log = job_logger(__name__, job_id)

    start_time = datetime.now()

    model = load_model()
    if model is None:
        return {
            'error': 'Model could not be loaded',
            'good_frames': [], 'bad_frames': [], 'failed_frames': [],
            'total_frames': len(frame_paths)
        }

    log.info(f"Classifying {len(frame_paths)} frames")

    good_frames, bad_frames, failed_frames = [], [], []

    for i, frame_path in enumerate(frame_paths, 1):
        frame_path = Path(frame_path)
        label, confidence = classify_frame(frame_path, model)

        frame_info = {'path': str(frame_path), 'filename': frame_path.name, 'confidence': float(confidence)}

        if label == "GOOD":
            good_frames.append(frame_info)
        elif label == "BAD":
            bad_frames.append(frame_info)
        else:
            failed_frames.append({**frame_info, 'error': 'Processing failed'})

        if i % 50 == 0 or i == len(frame_paths):
            log.info(f"Progress: {i}/{len(frame_paths)} - Good:{len(good_frames)} Bad:{len(bad_frames)}")

    # Organize files
    if organize_files and output_dir:
        _organize_files(good_frames, bad_frames, output_dir)

    processing_time = (datetime.now() - start_time).total_seconds()

    results = {
        'good_frames': good_frames,
        'bad_frames': bad_frames,
        'failed_frames': failed_frames,
        'total_frames': len(frame_paths),
        'statistics': {
            'good_count': len(good_frames),
            'bad_count': len(bad_frames),
            'failed_count': len(failed_frames),
            'processing_time_seconds': processing_time,
            'processed_date': datetime.now().isoformat()
        }
    }

    if output_dir:
        try:
            with open(output_dir / "classification_summary.json", 'w') as f:
                json.dump(results['statistics'], f, indent=2)
        except OSError as e:
            log.warning(f"Could not write classification summary: {e}")

    log.info(f"Classification complete: Good={len(good_frames)}, Bad={len(bad_frames)}")
    return results


def _organize_files(good_frames: List[Dict], bad_frames: List[Dict], output_dir: Path) -> None:
    """Organize classified frames into good/bad folders."""
    good_folder = output_dir / "good"
    bad_folder = output_dir / "bad"
    good_folder.mkdir(exist_ok=True, parents=True)
    bad_folder.mkdir(exist_ok=True, parents=True)

    for frame_info in good_frames:
        src = Path(frame_info['path'])
        if src.exists():
            try:
                shutil.copy2(src, good_folder / src.name)
            except (OSError, IOError) as e:
                logger.warning(f"Could not copy {src}: {e}")

    for frame_info in bad_frames:
        src = Path(frame_info['path'])
        if src.exists():
            try:
                shutil.copy2(src, bad_folder / src.name)
            except (OSError, IOError) as e:
                logger.warning(f"Could not copy {src}: {e}")
