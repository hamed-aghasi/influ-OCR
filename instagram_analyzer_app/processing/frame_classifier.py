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
    """Load (and cache) the classifier. Returns None on any failure — callers
    must surface that as its own error, never as 'no frames were good'."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    log = job_logger(__name__)
    with _model_lock:
        if _cached_model is not None:
            return _cached_model
        savedmodel_path = settings.model_dir / "frame_classifier_savedmodel"
        if not savedmodel_path.exists():
            log.error("Classifier SavedModel not found at %s", savedmodel_path)
            return None
        try:
            import keras

            _cached_model = keras.layers.TFSMLayer(str(savedmodel_path), call_endpoint="serving_default")
            return _cached_model
        except Exception as exc:  # noqa: BLE001 — caller checks None; reason must be logged
            log.exception("Classifier model failed to load from %s: %s", savedmodel_path, exc)
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


_INFER_BATCH = 32  # ponytail: fixed chunk bounds memory; tune only if profiling says so


def _load_and_preprocess(frame_path: Path) -> Tuple[Optional[np.ndarray], str]:
    """Returns (image array without batch dim, reason). `reason` is non-empty
    only on failure, so the caller can report *why* a frame was dropped."""
    try:
        with open(frame_path, "rb") as f:
            arr = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except OSError as exc:
        return None, f"unreadable file: {exc}"

    processed = _preprocess(img)
    if processed is None:
        return None, "frame could not be decoded or preprocessed"
    return processed[0], ""


def _confidences(prediction: Any, n: int) -> np.ndarray:
    if isinstance(prediction, dict):
        prediction = next(iter(prediction.values()))
    arr = prediction.numpy() if hasattr(prediction, "numpy") else np.asarray(prediction)
    return arr.reshape(n, -1)[:, 0]


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

    def _fail(path: Path, reason: str) -> None:
        if not failed_frames:  # log the first failure only — the rest repeat it
            log.warning("Frame %s could not be classified: %s", path.name, reason)
        failed_frames.append({"path": str(path), "filename": path.name, "confidence": 0.0, "error": reason})

    # Batched inference: one model call per chunk instead of one per frame —
    # the per-frame loop was the classify stage's dominant cost.
    for offset in range(0, len(frame_paths), _INFER_BATCH):
        chunk = [Path(p) for p in frame_paths[offset:offset + _INFER_BATCH]]
        loaded = [_load_and_preprocess(p) for p in chunk]

        for path, (array, reason) in zip(chunk, loaded):
            if array is None:
                _fail(path, reason)

        valid = [(p, a) for p, (a, _) in zip(chunk, loaded) if a is not None]
        if valid:
            try:
                prediction = model(np.stack([a for _, a in valid]))
                confs = _confidences(prediction, len(valid))
            except Exception as exc:  # noqa: BLE001 — TF inference can raise many things
                for path, _ in valid:
                    _fail(path, f"inference failed: {exc}")
            else:
                for (path, _), confidence in zip(valid, confs):
                    info = {"path": str(path), "filename": path.name, "confidence": float(confidence)}
                    if confidence > settings.classifier_threshold:
                        good_frames.append(info)
                        good_paths.append(path)
                    else:
                        bad_frames.append(info)
                        bad_paths.append(path)

        done = min(offset + _INFER_BATCH, len(frame_paths))
        log.info("Progress: %d/%d (good=%d bad=%d)", done, len(frame_paths), len(good_frames), len(bad_frames))

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

    result = {
        "good_paths": good_paths,
        "bad_paths": bad_paths,
        "good_frames": good_frames,
        "bad_frames": bad_frames,
        "failed_frames": failed_frames,
        "total_frames": len(frame_paths),
        "statistics": stats,
    }

    if frame_paths and len(failed_frames) == len(frame_paths):
        # Every frame failed: the model is broken, not the upload. Without this
        # the caller sees good_paths == [] and blames the user's screenshots.
        result["error"] = f"classifier failed on all {len(frame_paths)} frames"
        log.error(result["error"])

    return result
