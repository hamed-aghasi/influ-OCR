"""Frame classification: TF SavedModel labels frames as good/bad.

Returns lists of paths instead of copying files into good/bad folders.
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import settings
from .logger import job_logger


_cached_model: Any = None
_model_lock = threading.Lock()


def load_model() -> Optional[Any]:
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    with _model_lock:
        if _cached_model is not None:
            return _cached_model
        savedmodel_path = settings.model_dir / "frame_classifier_savedmodel"
        if not savedmodel_path.exists():
            return None
        try:
            import keras

            _cached_model = keras.layers.TFSMLayer(str(savedmodel_path), call_endpoint="serving_default")
            return _cached_model
        except Exception:  # noqa: BLE001 — model load is best-effort; caller checks None
            return None


def _preprocess(img: np.ndarray) -> Optional[np.ndarray]:
    if img is None:
        return None
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < settings.dark_frame_threshold:
            img = cv2.convertScaleAbs(img, alpha=1.5, beta=40)
        size = settings.classifier_image_size
        resized = cv2.resize(img, (size, size))
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)
    except cv2.error:
        return None


def _classify_one(frame_path: Path, model: Any) -> Tuple[Optional[str], float]:
    try:
        with open(frame_path, "rb") as f:
            arr = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except OSError:
        return None, 0.0

    processed = _preprocess(img)
    if processed is None:
        return None, 0.0

    try:
        prediction = model(processed)
    except Exception:  # noqa: BLE001 — TF inference can raise many things
        return None, 0.0

    if isinstance(prediction, dict):
        pred_value = next(iter(prediction.values())).numpy()
    else:
        pred_value = prediction.numpy() if hasattr(prediction, "numpy") else prediction

    confidence = float(pred_value[0][0]) if len(pred_value.shape) > 1 else float(pred_value[0])
    return ("GOOD" if confidence > settings.classifier_threshold else "BAD"), confidence


def classify_frames(
    frame_paths: List[Path],
    output_dir: Optional[Path] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    log = job_logger(__name__, job_id)
    start = datetime.now()

    model = load_model()
    if model is None:
        log.error("Classifier model could not be loaded")
        return {
            "error": "Model could not be loaded",
            "good_paths": [],
            "bad_paths": [],
            "good_frames": [],
            "bad_frames": [],
            "failed_frames": [],
            "total_frames": len(frame_paths),
        }

    log.info("Classifying %d frames", len(frame_paths))

    good_paths: List[Path] = []
    bad_paths: List[Path] = []
    good_frames: List[Dict] = []
    bad_frames: List[Dict] = []
    failed_frames: List[Dict] = []

    for i, frame_path in enumerate(frame_paths, 1):
        path = Path(frame_path)
        label, confidence = _classify_one(path, model)
        info = {"path": str(path), "filename": path.name, "confidence": confidence}

        if label == "GOOD":
            good_frames.append(info)
            good_paths.append(path)
        elif label == "BAD":
            bad_frames.append(info)
            bad_paths.append(path)
        else:
            failed_frames.append({**info, "error": "Processing failed"})

        if i % 50 == 0 or i == len(frame_paths):
            log.info("Progress: %d/%d (good=%d bad=%d)", i, len(frame_paths), len(good_frames), len(bad_frames))

    elapsed = (datetime.now() - start).total_seconds()
    stats = {
        "good_count": len(good_frames),
        "bad_count": len(bad_frames),
        "failed_count": len(failed_frames),
        "processing_time_seconds": elapsed,
        "processed_date": datetime.now().isoformat(),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(output_dir / "classification_summary.json", "w") as f:
                json.dump(stats, f, indent=2)
        except OSError as exc:
            log.warning("Could not write classification summary: %s", exc)

    log.info("Classification done: good=%d bad=%d failed=%d", len(good_frames), len(bad_frames), len(failed_frames))

    return {
        "good_paths": good_paths,
        "bad_paths": bad_paths,
        "good_frames": good_frames,
        "bad_frames": bad_frames,
        "failed_frames": failed_frames,
        "total_frames": len(frame_paths),
        "statistics": stats,
    }
