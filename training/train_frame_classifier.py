#!/usr/bin/env python3
"""Train the good/bad frame classifier and export it for the app.

Reads a folder of labelled frames::

    data/
      bad_frame/   <- blurred, mid-scroll, transition frames
      good_frame/  <- clean, readable Insights screens

and writes ``frame_classifier_savedmodel/`` + ``frame_classifier_v1.h5`` +
``model_metadata.json``, i.e. exactly what ``processing/frame_classifier.py``
loads at inference time.

The architecture and compile settings below were recovered from the shipped
v1 model (``frame_classifier_v1.h5``, keras 3.11.3), so a retrain is a drop-in
replacement rather than a new contract.

Usage::

    python training/train_frame_classifier.py --data-dir data/frames
    python training/train_frame_classifier.py --data-dir data/frames --dedup --epochs 30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Class index == position in this tuple. Keras' image_dataset_from_directory
# sorts class folders alphabetically, which is how v1 ended up with
# bad_frame=0 / good_frame=1 — keep the order or the sigmoid inverts.
CLASS_DIRS: Tuple[str, str] = ("bad_frame", "good_frame")
FALLBACK_DIRS: Tuple[str, str] = ("bad", "good")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Mirrors processing/config.py defaults. Overridable on the CLI, but changing
# one here without changing it there reintroduces train/serve skew.
DEFAULT_IMAGE_SIZE = 224
DEFAULT_DARK_THRESHOLD = 80
DEFAULT_PHASH_SIZE = 8      # config.perceptual_hash_size
DEFAULT_PHASH_DISTANCE = 4  # config.perceptual_hash_distance
DEFAULT_LR = 1e-3  # v1 was saved at 6.25e-05 == 1e-3 * 0.5**4 (ReduceLROnPlateau)


# ---------------------------------------------------------------- preprocessing


def preprocess_frame(
    path: Path,
    size: int = DEFAULT_IMAGE_SIZE,
    dark_threshold: int = DEFAULT_DARK_THRESHOLD,
    dark_boost: bool = True,
) -> Optional[np.ndarray]:
    """Byte-for-byte the same transform as ``frame_classifier._preprocess``.

    Kept in lockstep deliberately: cv2 decodes BGR and the app never converts
    to RGB, so training must see BGR too. Feeding RGB here would train the head
    on channel statistics the frozen backbone never sees in production.
    """
    try:
        with open(path, "rb") as f:
            arr = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except OSError:
        return None
    if img is None:
        return None
    try:
        if dark_boost:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if np.mean(gray) < dark_threshold:
                img = cv2.convertScaleAbs(img, alpha=1.5, beta=40)
        resized = cv2.resize(img, (size, size))
        return resized.astype(np.float32) / 255.0
    except cv2.error:
        return None


# ------------------------------------------------------------------- data


def resolve_class_dirs(data_dir: Path) -> Tuple[Path, Path]:
    """Accept bad_frame/good_frame, or plain bad/good as a convenience."""
    for names in (CLASS_DIRS, FALLBACK_DIRS):
        bad, good = data_dir / names[0], data_dir / names[1]
        if bad.is_dir() and good.is_dir():
            if names is FALLBACK_DIRS:
                print(f"[data] using {names[0]}/ and {names[1]}/ as bad_frame/good_frame")
            return bad, good
    sys.exit(
        f"error: {data_dir} must contain 'bad_frame/' and 'good_frame/' "
        f"(or 'bad/' and 'good/'); found: "
        f"{sorted(p.name for p in data_dir.iterdir() if p.is_dir()) if data_dir.is_dir() else 'nothing'}"
    )


def list_images(folder: Path) -> List[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def validate(paths: Sequence[Path], size: int, dark_threshold: int, dark_boost: bool) -> List[Path]:
    """Drop files that cannot be decoded, loudly. A corrupt frame that slips
    through would otherwise train the model on a zero tensor."""
    ok, bad = [], []
    for p in paths:
        if preprocess_frame(p, size, dark_threshold, dark_boost) is None:
            bad.append(p)
        else:
            ok.append(p)
    if bad:
        print(f"[data] skipped {len(bad)} undecodable file(s), e.g. {bad[0].name}")
    return ok


def dedupe(
    paths: List[Path],
    hash_size: int = DEFAULT_PHASH_SIZE,
    distance: int = DEFAULT_PHASH_DISTANCE,
) -> List[Path]:
    """Drop near-identical frames, keeping the first of each cluster.

    Near-duplicates are the main reason a run reports val_accuracy 1.0:
    consecutive frames from one video land in both splits, so the model
    recognises rather than generalises.

    Same phash logic and defaults as ``processing/dedup.py``, reimplemented
    rather than imported — that module pulls in ``config.settings``, which
    would make training depend on pydantic and on runtime secrets like
    OPENROUTER_API_KEY being set.
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        print("[dedup] needs ImageHash + Pillow (pip install -r training/requirements.txt); skipping")
        return paths

    kept: List[Tuple[Path, "imagehash.ImageHash"]] = []
    dupes = 0
    for path in paths:
        try:
            with Image.open(path) as img:
                current = imagehash.phash(img, hash_size=hash_size)
        except Exception as exc:  # noqa: BLE001 — a bad frame must not stop the run
            print(f"[dedup] phash failed for {path.name}: {exc}")
            continue
        if any((current - other) <= distance for _, other in kept):
            dupes += 1
        else:
            kept.append((path, current))
    print(f"[dedup] {len(paths)} -> {len(kept)} kept, {dupes} near-duplicates dropped")
    return [p for p, _ in kept]


def split(paths: List[Path], val_split: float, seed: int) -> Tuple[List[Path], List[Path]]:
    """Stratified per class: the caller splits each class separately."""
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_split))) if len(shuffled) > 1 else 0
    return shuffled[n_val:], shuffled[:n_val]


def build_dataset(tf, paths, labels, size, dark_threshold, dark_boost, batch_size, training, seed):
    def _load(path_bytes):
        img = preprocess_frame(Path(path_bytes.decode()), size, dark_threshold, dark_boost)
        # Validated up front, so None here means the file changed mid-run.
        return img if img is not None else np.zeros((size, size, 3), np.float32)

    def _map(path, label):
        img = tf.numpy_function(_load, [path], tf.float32)
        img.set_shape((size, size, 3))
        return img, label

    ds = tf.data.Dataset.from_tensor_slices(
        ([str(p) for p in paths], np.asarray(labels, dtype="float32"))
    )
    if training:
        ds = ds.shuffle(max(len(paths), 1), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (augment(tf, x, seed), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def augment(tf, img, seed: int):
    """Photometric only, plus a tiny crop-and-resize.

    No horizontal flip: these are screenshots of text, and a mirrored Insights
    panel is not an input the model will ever see.
    """
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    scale = tf.random.uniform([], 0.90, 1.0)
    size = tf.shape(img)[0]
    crop = tf.cast(tf.cast(size, tf.float32) * scale, tf.int32)
    img = tf.image.random_crop(img, [crop, crop, 3])
    img = tf.image.resize(img, [size, size])
    return tf.clip_by_value(img, 0.0, 1.0)


# ------------------------------------------------------------------ model


def build_model(keras, size: int, lr: float):
    """Reproduces v1 exactly: frozen MobileNetV2 -> GAP -> 64 -> 32 -> sigmoid.

    No Rescaling layer on purpose — the /255 happens in preprocess_frame, the
    same place it happens at inference.
    """
    base = keras.applications.MobileNetV2(
        input_shape=(size, size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(size, size, 3)),
            base,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def export(keras, model, out_dir: Path, size: int, epochs: int, history, val_ds, threshold: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    savedmodel_dir = out_dir / "frame_classifier_savedmodel"

    # TFSMLayer(call_endpoint="serving_default") is what the app uses, and only
    # model.export() produces that endpoint — model.save(dir) does not.
    model.export(str(savedmodel_dir))
    model.save(str(out_dir / "frame_classifier_v1.h5"))

    val_metrics = model.evaluate(val_ds, verbose=0, return_dict=True) if val_ds is not None else {}
    metadata = {
        "training_date": datetime.now().isoformat(),
        "model_version": "v1",
        "architecture": "MobileNetV2_transfer_learning",
        "input_shape": [size, size, 3],
        "validation_accuracy": val_metrics.get("accuracy"),
        "epochs_trained": len(history.epoch),
        "epochs_requested": epochs,
        "final_metrics": {
            "val_accuracy": val_metrics.get("accuracy"),
            "val_loss": val_metrics.get("loss"),
        },
        "labels": {"0": "bad_frame", "1": "good_frame"},
        "inference_contract": {
            "preprocess": "BGR, optional dark boost, resize, /255.0",
            "decision": f"sigmoid > {threshold} => GOOD",
        },
    }
    (out_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\n[export] wrote {savedmodel_dir}")
    print(f"[export] wrote {out_dir / 'frame_classifier_v1.h5'}")
    print(f"[export] wrote {out_dir / 'model_metadata.json'}")


def verify(keras, out_dir: Path, sample: Sequence[Path], size, dark_threshold, dark_boost) -> None:
    """Reload through the app's exact code path, not the training graph."""
    layer = keras.layers.TFSMLayer(
        str(out_dir / "frame_classifier_savedmodel"), call_endpoint="serving_default"
    )
    print("\n[verify] round-tripping through TFSMLayer:")
    for p in sample:
        img = preprocess_frame(p, size, dark_threshold, dark_boost)
        pred = layer(np.expand_dims(img, axis=0))
        if isinstance(pred, dict):
            pred = next(iter(pred.values()))
        value = float(np.asarray(pred).reshape(-1)[0])
        print(f"  {p.parent.name:11} {p.name[:40]:40} -> {value:.4f}")


# ------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, help="folder holding bad_frame/ and good_frame/")
    ap.add_argument("--output-dir", type=Path, default=Path("instagram_analyzer_app/models"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    ap.add_argument("--dark-threshold", type=int, default=DEFAULT_DARK_THRESHOLD)
    ap.add_argument("--threshold", type=float, default=0.65, help="GOOD cutoff, for metadata + report")
    ap.add_argument("--no-dark-boost", action="store_true", help="skip the low-light boost (must match config.py)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--dedup", action="store_true", help="drop near-duplicate frames before splitting")
    ap.add_argument("--dedup-hash-size", type=int, default=DEFAULT_PHASH_SIZE)
    ap.add_argument("--dedup-distance", type=int, default=DEFAULT_PHASH_DISTANCE)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    # Imported late: TF takes seconds to load, and --help should not pay for it.
    import tensorflow as tf
    import keras

    tf.keras.utils.set_random_seed(args.seed)
    dark_boost = not args.no_dark_boost

    bad_dir, good_dir = resolve_class_dirs(args.data_dir)
    per_class: List[List[Path]] = []
    for label, folder in enumerate((bad_dir, good_dir)):
        paths = list_images(folder)
        if not paths:
            sys.exit(f"error: no images in {folder}")
        paths = validate(paths, args.image_size, args.dark_threshold, dark_boost)
        if args.dedup:
            paths = dedupe(paths, args.dedup_hash_size, args.dedup_distance)
        print(f"[data] {CLASS_DIRS[label]:11} {len(paths)} frames")
        per_class.append(paths)

    train_paths, train_labels, val_paths, val_labels = [], [], [], []
    for label, paths in enumerate(per_class):
        tr, va = split(paths, args.val_split, args.seed + label)
        train_paths += tr
        train_labels += [label] * len(tr)
        val_paths += va
        val_labels += [label] * len(va)

    if not train_paths:
        sys.exit("error: no training frames left after filtering")
    print(f"[data] train={len(train_paths)} val={len(val_paths)}")

    counts = np.bincount(np.asarray(train_labels, dtype=int), minlength=2)
    class_weight = {i: float(len(train_labels)) / (2.0 * max(int(c), 1)) for i, c in enumerate(counts)}
    if counts.min() and counts.max() / counts.min() > 1.5:
        print(f"[data] imbalance {counts.tolist()} -> class_weight {class_weight}")

    train_ds = build_dataset(
        tf, train_paths, train_labels, args.image_size, args.dark_threshold,
        dark_boost, args.batch_size, not args.no_augment, args.seed,
    )
    val_ds = (
        build_dataset(
            tf, val_paths, val_labels, args.image_size, args.dark_threshold,
            dark_boost, args.batch_size, False, args.seed,
        )
        if val_paths
        else None
    )

    model = build_model(keras, args.image_size, args.lr)
    model.summary()

    callbacks = []
    if val_ds is not None:
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1
            ),
            # factor=0.5 is what produced v1's saved LR of 1e-3 * 0.5**4.
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
            ),
        ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    export(keras, model, args.output_dir, args.image_size, args.epochs, history, val_ds, args.threshold)

    sample = (val_paths or train_paths)[:3] + (val_paths or train_paths)[-3:]
    verify(keras, args.output_dir, sample, args.image_size, args.dark_threshold, dark_boost)

    if val_ds is not None:
        acc = model.evaluate(val_ds, verbose=0, return_dict=True).get("accuracy")
        if acc is not None and acc >= 0.999 and not args.dedup:
            print(
                "\n[warn] val_accuracy is ~1.0. With frames sampled from the same "
                "videos this usually means near-duplicates spanned the split. "
                "Re-run with --dedup before trusting it."
            )


if __name__ == "__main__":
    main()
