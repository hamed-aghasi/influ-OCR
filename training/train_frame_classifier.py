#!/usr/bin/env python3
"""Train the good/bad frame classifier and export it for the app.

Reads labelled frames in either layout::

    train/                        (per-video — preferred, produced by extract_frames.py)
      <video-name>/
        good/   <- frames carrying the numbers we need
        bad/    <- everything else

    data/frames/                  (class-first — legacy)
      bad_frame/
      good_frame/

and writes a timestamped run directory containing ``frame_classifier_savedmodel/``,
a Keras copy, ``model_metadata.json`` and a set of reports. Deploying that model
into the app is a separate, explicit step (``--deploy``).

The architecture and compile settings were recovered from the shipped v1 model,
so a retrain is a drop-in replacement rather than a new contract.

Usage::

    python training/train_frame_classifier.py --data-dir train
    python training/train_frame_classifier.py --data-dir train --holdout-videos "some video"
    python training/train_frame_classifier.py --data-dir train --deploy
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# Label index is pinned here, not inferred from directory ordering. v1 got
# bad=0/good=1 from alphabetical sorting; relying on that again would make an
# inverted sigmoid a silent, catastrophic possibility.
BAD, GOOD = 0, 1
CLASS_NAMES = {BAD: "bad", GOOD: "good"}
GOOD_DIRS = ("good", "good_frame")
BAD_DIRS = ("bad", "bad_frame")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Mirrors processing/config.py defaults. Overridable on the CLI, but changing
# one here without changing it there reintroduces train/serve skew.
DEFAULT_IMAGE_SIZE = 224
DEFAULT_DARK_THRESHOLD = 80
# --dedup settings. Deliberately coarser than production (config: 16/16): --dedup
# only thins near-identical training frames within a video, it does not decide
# what reaches OCR (that is processing/frame_selection/ now).
DEFAULT_PHASH_SIZE = 8
DEFAULT_PHASH_DISTANCE = 4
DEFAULT_THRESHOLD = 0.30    # config.classifier_threshold — what production decides at
DEFAULT_LR = 1e-3  # v1 was saved at 6.25e-05 == 1e-3 * 0.5**4 (ReduceLROnPlateau)


class FrameItem(NamedTuple):
    """One labelled frame, with the provenance every report needs."""

    path: Path
    label: int
    video: str
    luma: float

    @property
    def is_dark(self) -> bool:
        return self.luma < DEFAULT_DARK_THRESHOLD


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
    img = _decode(path)
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


def _decode(path: Path) -> Optional[np.ndarray]:
    try:
        with open(path, "rb") as f:
            arr = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except OSError:
        return None


def probe_frame(path: Path) -> Optional[float]:
    """Decode once and return mean luma, or None if the file is unusable.

    Doubles as the validity check, so discovery costs one decode per frame
    instead of the two it used to (validate + luma).
    """
    img = _decode(path)
    if img is None:
        return None
    try:
        return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
    except cv2.error:
        return None


# ------------------------------------------------------------------- data


def _images_in(folder: Path) -> List[Path]:
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and not p.name.startswith("._")   # macOS AppleDouble sidecars
        and p.name != ".DS_Store"
    )


def _class_subdirs(parent: Path) -> Dict[int, Path]:
    """Return {label: dir} for whichever class folders exist under `parent`."""
    found = {}
    for label, names in ((GOOD, GOOD_DIRS), (BAD, BAD_DIRS)):
        for name in names:
            d = parent / name
            if d.is_dir():
                found[label] = d
                break
    return found


def discover_dataset(root: Path) -> List[FrameItem]:
    """Find labelled frames in either layout, carrying the video identity.

    The video a frame came from is the single most important piece of metadata
    here: frames from one recording are near-duplicates of each other, so a
    split that ignores it scores the model on frames it has effectively already
    seen. That is how v1 shipped val_accuracy 1.0 while being blind in practice.
    """
    if not root.is_dir():
        sys.exit(f"error: {root} does not exist")

    pairs: List[Tuple[Path, int, str]] = []

    # Layout A: per-video folders, each with its own good/ and bad/.
    for child in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))):
        for label, class_dir in _class_subdirs(child).items():
            for p in _images_in(class_dir):
                pairs.append((p, label, child.name))

    # Layout B: class-first. A subfolder inside the class dir names the video.
    for label, class_dir in _class_subdirs(root).items():
        for p in _images_in(class_dir):
            rel = p.relative_to(class_dir).parts
            pairs.append((p, label, rel[0] if len(rel) > 1 else "ungrouped"))

    if not pairs:
        listing = sorted(p.name for p in root.iterdir()) if root.is_dir() else []
        sys.exit(
            f"error: no labelled frames under {root}\n"
            f"  expected either <video>/good|bad/ or good_frame|bad_frame/ at the top level\n"
            f"  found: {listing or 'nothing'}"
        )

    items, unreadable = [], []
    for path, label, video in pairs:
        luma = probe_frame(path)
        if luma is None:
            unreadable.append(path)
            continue
        items.append(FrameItem(path, label, video, luma))

    if unreadable:
        print(f"[data] skipped {len(unreadable)} undecodable file(s), e.g. {unreadable[0].name}")
    return items


def counts_matrix(items: Sequence[FrameItem]) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = {}
    for it in items:
        row = matrix.setdefault(it.video, {"good": 0, "bad": 0, "dark_good": 0})
        row[CLASS_NAMES[it.label]] += 1
        if it.label == GOOD and it.is_dark:
            row["dark_good"] += 1
    return matrix


def print_dataset_summary(items: Sequence[FrameItem]) -> None:
    matrix = counts_matrix(items)
    width = max((len(v) for v in matrix), default=5)
    print(f"\n[data] {len(items)} labelled frames from {len(matrix)} video(s)")
    print(f"       {'video'.ljust(width)}  {'good':>5} {'bad':>5} {'dark_good':>10}")
    for video, row in sorted(matrix.items()):
        print(f"       {video.ljust(width)}  {row['good']:>5} {row['bad']:>5} {row['dark_good']:>10}")

    good = [i for i in items if i.label == GOOD]
    dark_good = [i for i in good if i.is_dark]
    if good:
        share = len(dark_good) / len(good)
        if share < 0.10:
            print(
                f"\n[warn] only {len(dark_good)}/{len(good)} good frames ({share:.0%}) are dark-mode.\n"
                f"       The shipped model failed precisely because it never saw dark mode.\n"
                f"       Label some dark-mode videos before trusting this model on them."
            )


def group_split(
    items: Sequence[FrameItem],
    val_split: float,
    seed: int,
    holdout: Sequence[str] = (),
) -> Tuple[List[FrameItem], List[FrameItem]]:
    """Assign whole *videos* to train or val — never individual frames.

    Bucketing is a stable hash of the video name rather than a shuffle, so
    adding a video in round 3 does not reshuffle rounds 1-2 and make the
    numbers incomparable.
    """
    videos = sorted({i.video for i in items})

    if holdout:
        unknown = [v for v in holdout if v not in videos]
        if unknown:
            sys.exit(f"error: --holdout-videos names unknown video(s): {unknown}\n  known: {videos}")
        val_videos = set(holdout)
    elif val_split <= 0 or len(videos) < 2:
        val_videos = set()
    else:
        pct = int(round(val_split * 100))
        val_videos = {
            v for v in videos
            if int(hashlib.sha1(f"{seed}:{v}".encode()).hexdigest(), 16) % 100 < pct
        }
        # Hash bucketing can land on all-or-nothing with few videos; fall back
        # to a deterministic single holdout rather than training with no val.
        if not val_videos or len(val_videos) == len(videos):
            val_videos = {sorted(videos, key=lambda v: hashlib.sha1(f"{seed}:{v}".encode()).hexdigest())[0]}

    train = [i for i in items if i.video not in val_videos]
    val = [i for i in items if i.video in val_videos]

    if len(videos) < 2:
        print(
            f"\n[warn] only {len(videos)} video in the dataset, so train and val cannot be separated\n"
            f"       by video. Any validation score will be measured on near-duplicates of the\n"
            f"       training frames and will look far better than the model really is.\n"
            f"       Add a second video before believing the number."
        )
    else:
        print(f"\n[split] val videos: {sorted(val_videos) or 'none'}")
        print(f"[split] train={len(train)} frames / val={len(val)} frames")
    return train, val


def dedupe_within_videos(
    items: Sequence[FrameItem],
    hash_size: int = DEFAULT_PHASH_SIZE,
    distance: int = DEFAULT_PHASH_DISTANCE,
) -> List[FrameItem]:
    """Drop near-identical frames *within each video*, keeping the first seen.

    This is a data-volume reducer, not the leakage control — `group_split`
    handles leakage. Applied after the split so it can never move a frame's
    twin across the boundary. Same phash logic and defaults as
    ``processing/dedup.py``, reimplemented rather than imported because that
    module pulls in ``config.settings``, which would make training depend on
    pydantic and on runtime secrets being set.
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        print("[dedup] needs ImageHash + Pillow (pip install -r training/requirements.txt); skipping")
        return list(items)

    kept: List[FrameItem] = []
    dropped = 0
    by_video: Dict[Tuple[str, int], List] = {}
    for it in items:
        try:
            with Image.open(it.path) as img:
                current = imagehash.phash(img, hash_size=hash_size)
        except Exception as exc:  # noqa: BLE001 — a bad frame must not stop the run
            print(f"[dedup] phash failed for {it.path.name}: {exc}; keeping it")
            kept.append(it)
            continue
        bucket = by_video.setdefault((it.video, it.label), [])
        if any((current - other) <= distance for other in bucket):
            dropped += 1
        else:
            bucket.append(current)
            kept.append(it)
    print(f"[dedup] {len(items)} -> {len(kept)} kept, {dropped} near-duplicates dropped")
    return kept


def build_dataset(tf, items: Sequence[FrameItem], size, dark_threshold, dark_boost,
                  batch_size, training, seed, cache_file: str = ""):
    def _load(path_bytes):
        img = preprocess_frame(Path(path_bytes.decode()), size, dark_threshold, dark_boost)
        # Validated up front, so None here means the file changed mid-run.
        return img if img is not None else np.zeros((size, size, 3), np.float32)

    def _map(path, label):
        img = tf.numpy_function(_load, [path], tf.float32)
        img.set_shape((size, size, 3))
        return img, label

    ds = tf.data.Dataset.from_tensor_slices(
        ([str(i.path) for i in items], np.asarray([i.label for i in items], dtype="float32"))
    )
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    # Decode once, not once per epoch. numpy_function holds the GIL, so this is
    # the difference between seconds and minutes per epoch on a laptop.
    ds = ds.cache(cache_file)
    if training:
        ds = ds.shuffle(max(len(items), 1), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(lambda x, y: (augment(tf, x, size), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def augment(tf, img, size: int):
    """Photometric only, plus a tiny crop-and-resize.

    No horizontal flip: these are screenshots of text, and a mirrored Insights
    panel is not an input the model will ever see.

    Note this cannot manufacture dark mode from light mode — that is a polarity
    inversion, not a brightness nudge. Dark-mode coverage has to come from real
    labelled frames.
    """
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    scale = tf.random.uniform([], 0.90, 1.0)
    crop = tf.cast(tf.cast(size, tf.float32) * scale, tf.int32)
    img = tf.image.random_crop(img, [crop, crop, 3])
    img = tf.image.resize(img, [size, size])
    img = tf.clip_by_value(img, 0.0, 1.0)
    img.set_shape((size, size, 3))  # keep the static shape the Input layer expects
    return img


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
        # Threshold-free metrics alongside accuracy: accuracy is computed at 0.5
        # while production decides at config.classifier_threshold, so accuracy alone describes a model
        # nobody deploys.
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


# ------------------------------------------------------------------ scoring


def score_items(model, items: Sequence[FrameItem], size, dark_threshold, dark_boost,
                batch_size: int = 32) -> np.ndarray:
    """Raw sigmoid score per item, in order."""
    blank = np.zeros((size, size, 3), np.float32)
    scores = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        decoded = [preprocess_frame(i.path, size, dark_threshold, dark_boost) for i in chunk]
        batch = np.stack([d if d is not None else blank for d in decoded])
        preds = model.predict(batch, verbose=0)
        scores.extend(np.asarray(preds).reshape(-1).tolist())
    return np.asarray(scores, dtype="float64")


def confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = scores > threshold
    truth = labels == GOOD
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "kept": tp + fp,          # what the OpenRouter bill tracks
        "accuracy": round((tp + tn) / len(labels), 4) if len(labels) else 0.0,
    }


def threshold_report(items: Sequence[FrameItem], scores: np.ndarray, configured: float,
                     min_recall: float) -> Dict:
    """Sweep the decision threshold and name the operating points that matter.

    A dropped good frame is a metric nobody can recover; a kept bad frame only
    costs OCR tokens. So recall is the constraint and the number of kept frames
    is what we minimise subject to it.
    """
    labels = np.asarray([i.label for i in items])
    grid = sorted({round(t, 2) for t in np.arange(0.05, 1.0, 0.05)} | {round(configured, 4)})
    rows = [confusion(labels, scores, t) for t in grid]

    best_f1 = max(rows, key=lambda r: r["f1"])
    eligible = [r for r in rows if r["recall"] >= min_recall]
    cheapest = min(eligible, key=lambda r: (r["kept"], -r["threshold"])) if eligible else None
    current = next(r for r in rows if abs(r["threshold"] - configured) < 1e-9)

    print(f"\n[threshold] sweep on {len(items)} val frames "
          f"({int((labels == GOOD).sum())} good / {int((labels == BAD).sum())} bad)")
    print(f"  {'thr':>5} {'prec':>6} {'recall':>7} {'f1':>6} {'kept':>5} {'tp':>4} {'fp':>4} {'fn':>4}")
    for r in rows:
        mark = ""
        if r is current: mark += " <- config"
        if r is best_f1: mark += " <- best F1"
        if cheapest is not None and r is cheapest: mark += f" <- cheapest @recall>={min_recall}"
        print(f"  {r['threshold']:>5.2f} {r['precision']:>6.3f} {r['recall']:>7.3f} "
              f"{r['f1']:>6.3f} {r['kept']:>5} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4}{mark}")

    if cheapest is not None and abs(cheapest["threshold"] - configured) > 1e-9:
        print(f"\n  To adopt the cheapest safe operating point, set in .env or the environment:")
        print(f"      CLASSIFIER_THRESHOLD={cheapest['threshold']}")
        print(f"  (pydantic BaseSettings picks it up; no code change)")
    elif cheapest is None:
        print(f"\n  [warn] no threshold reaches recall >= {min_recall}. The model is not good "
              f"enough to filter safely yet — more labelled data is the fix, not a lower cutoff.")

    _print_histogram(items, scores)
    return {"sweep": rows, "configured": current, "best_f1": best_f1,
            "cheapest_at_min_recall": cheapest, "min_recall": min_recall}


def _print_histogram(items: Sequence[FrameItem], scores: np.ndarray, bins: int = 20) -> None:
    """Score distribution per class. Bimodality is the signature of
    'confidently wrong', which a single accuracy number hides completely."""
    print("\n[scores] val distribution (| = good, . = bad)")
    for label in (GOOD, BAD):
        sel = scores[np.asarray([i.label for i in items]) == label]
        if not len(sel):
            continue
        hist, _ = np.histogram(sel, bins=bins, range=(0.0, 1.0))
        peak = max(hist.max(), 1)
        char = "|" if label == GOOD else "."
        print(f"  {CLASS_NAMES[label]:>4}: " + "".join(
            char if h else " " for h in hist
        ) + f"   n={len(sel)} mean={sel.mean():.3f}")
    print("         " + "0" + " " * (bins - 2) + "1")


# ------------------------------------------------------------------ reports


def write_reports(run_dir: Path, train_items, train_scores, val_items, val_scores,
                  threshold: float, history) -> None:
    """Everything a human needs to decide what to label next round."""
    rows = []
    for split, items, scores in (("train", train_items, train_scores), ("val", val_items, val_scores)):
        for it, s in zip(items, scores):
            pred = GOOD if s > threshold else BAD
            rows.append({
                "path": str(it.path), "frame": it.path.name, "video": it.video,
                "split": split, "label": CLASS_NAMES[it.label], "score": round(float(s), 6),
                "pred": CLASS_NAMES[pred], "correct": int(pred == it.label),
                "mean_luma": round(it.luma, 2), "is_dark": int(it.is_dark),
            })
    with open(run_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    errors = [r for r in rows if not r["correct"]]
    errors.sort(key=lambda r: abs(r["score"] - threshold), reverse=True)
    lines = [
        "# Disagreements",
        "",
        f"Decision threshold {threshold}. Sorted by how *confident* the model was when it",
        "disagreed with your label — the top of this list is either the model's real blind",
        "spot or a frame whose label doesn't match the rule you've been applying.",
        "",
        f"{len(errors)} of {len(rows)} frames ({len(errors)/len(rows):.1%}).",
        "",
    ]
    val_errors = [r for r in errors if r["split"] == "val"]
    train_errors = [r for r in errors if r["split"] == "train"]
    for title, group, note in (
        ("## Validation errors", val_errors, "Frames from held-out videos. This is generalisation."),
        ("## Training errors", train_errors,
         "The model saw these and still got them wrong — a strong mislabel signal."),
    ):
        lines += [title, "", note, ""]
        if not group:
            lines += ["_none_", ""]
            continue
        lines += ["| score | your label | predicted | dark | video | frame |",
                  "|---|---|---|---|---|---|"]
        lines += [
            f"| {r['score']:.4f} | {r['label']} | {r['pred']} | {'yes' if r['is_dark'] else 'no'} "
            f"| {r['video']} | `{r['frame']}` |"
            for r in group[:60]
        ]
        if len(group) > 60:
            lines.append(f"\n_… and {len(group) - 60} more; see predictions.csv_")
        lines.append("")
    (run_dir / "errors.md").write_text("\n".join(lines), encoding="utf-8")

    _per_video_table(run_dir, rows, threshold)

    if history is not None:
        (run_dir / "history.json").write_text(
            json.dumps({k: [float(x) for x in v] for k, v in history.history.items()}, indent=1),
            encoding="utf-8",
        )
    print(f"[report] predictions.csv, errors.md, per_video.md -> {run_dir}")


def _per_video_table(run_dir: Path, rows: List[Dict], threshold: float) -> None:
    """Which video is dragging — the diagnosis that otherwise costs an afternoon."""
    videos: Dict[str, List[Dict]] = {}
    for r in rows:
        videos.setdefault(r["video"], []).append(r)

    lines = ["# Per-video results", "", f"Threshold {threshold}.", "",
             "| video | split | good | bad | dark | accuracy | recall (good) | mean good score |",
             "|---|---|---|---|---|---|---|---|"]
    print(f"\n[per-video] {'video':<28} {'split':>5} {'acc':>6} {'recall':>7} {'mean good':>10}")
    for video, group in sorted(videos.items()):
        good = [r for r in group if r["label"] == "good"]
        bad = [r for r in group if r["label"] == "bad"]
        dark = sum(r["is_dark"] for r in group)
        acc = sum(r["correct"] for r in group) / len(group)
        recall = sum(1 for r in good if r["pred"] == "good") / len(good) if good else float("nan")
        mean_good = sum(r["score"] for r in good) / len(good) if good else float("nan")
        split = group[0]["split"]
        lines.append(
            f"| {video} | {split} | {len(good)} | {len(bad)} | {dark} | {acc:.3f} | "
            f"{recall:.3f} | {mean_good:.3f} |"
        )
        print(f"            {video[:28]:<28} {split:>5} {acc:>6.3f} {recall:>7.3f} {mean_good:>10.3f}")
    (run_dir / "per_video.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ export


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def export(keras, model, run_dir: Path, args, items, train_items, val_items,
           history, val_metrics: Dict, sweep: Optional[Dict]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    savedmodel_dir = run_dir / "frame_classifier_savedmodel"

    # TFSMLayer(call_endpoint="serving_default") is what the app uses, and only
    # model.export() produces that endpoint — model.save(dir) does not.
    model.export(str(savedmodel_dir))
    model.save(str(run_dir / "frame_classifier.keras"))   # reloadable for --init-from later
    model.save(str(run_dir / "frame_classifier_v1.h5"))   # legacy name kept for the README contract

    best_epoch = None
    if history is not None and "val_loss" in history.history:
        best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    metadata = {
        "run_id": run_dir.name,
        "training_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": args.run_name or run_dir.name,
        "architecture": "MobileNetV2_transfer_learning",
        "input_shape": [args.image_size, args.image_size, 3],
        "labels": {"0": "bad_frame", "1": "good_frame"},
        "validation_accuracy": val_metrics.get("accuracy"),
        "epochs_requested": args.epochs,
        "epochs_ran": len(history.epoch) if history is not None else 0,
        # restore_best_weights=True means the exported weights come from here,
        # not from the last epoch.
        "best_epoch": best_epoch,
        "final_metrics": val_metrics,
        "threshold_sweep": {
            "configured": sweep["configured"] if sweep else None,
            "best_f1": sweep["best_f1"] if sweep else None,
            "cheapest_at_min_recall": sweep["cheapest_at_min_recall"] if sweep else None,
        } if sweep else None,
        "dataset": {
            "root": str(args.data_dir),
            "frames_total": len(items),
            "frames_train": len(train_items),
            "frames_val": len(val_items),
            "train_videos": sorted({i.video for i in train_items}),
            "val_videos": sorted({i.video for i in val_items}),
            "per_video": counts_matrix(items),
        },
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "git_sha": _git_sha(),
        "inference_contract": {
            "preprocess": "BGR, optional dark boost, resize, /255.0",
            "decision": f"sigmoid > {args.threshold} => GOOD",
            "dark_boost": not args.no_dark_boost,
        },
    }
    (run_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\n[export] {run_dir}")
    return savedmodel_dir


def verify(keras, savedmodel_dir: Path, items: Sequence[FrameItem], args) -> bool:
    """Reload through the app's exact code path, not the training graph.

    A broken export or an inverted sigmoid must fail here, at train time,
    rather than silently at upload time.
    """
    layer = keras.layers.TFSMLayer(str(savedmodel_dir), call_endpoint="serving_default")
    sample = ([i for i in items if i.label == BAD][:3] + [i for i in items if i.label == GOOD][:3])
    if not sample:
        return True

    print("\n[verify] round-tripping through TFSMLayer:")
    good_scores, bad_scores = [], []
    for it in sample:
        img = preprocess_frame(it.path, args.image_size, args.dark_threshold, not args.no_dark_boost)
        if img is None:
            print(f"  {CLASS_NAMES[it.label]:>4}  {it.path.name[:40]:40} -> UNREADABLE")
            continue
        pred = layer(np.expand_dims(img, axis=0))
        if isinstance(pred, dict):
            pred = next(iter(pred.values()))
        value = float(np.asarray(pred).reshape(-1)[0])
        (good_scores if it.label == GOOD else bad_scores).append(value)
        print(f"  {CLASS_NAMES[it.label]:>4}  {it.path.name[:40]:40} -> {value:.4f}")

    if good_scores and bad_scores and np.mean(good_scores) < np.mean(bad_scores):
        print("\n[error] good frames score BELOW bad frames — the sigmoid is inverted.\n"
              "        Do not deploy this model; check the class directory mapping.")
        return False
    return True


def deploy(run_dir: Path, models_dir: Path) -> None:
    """Swap the new model into the app, after backing up what is there.

    Export-to-temp-then-replace rather than writing over the live directory, so
    a crash cannot leave the app pointing at a half-written SavedModel.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    live = models_dir / "frame_classifier_savedmodel"

    if live.exists():
        backup = models_dir / f"backup_{stamp}"
        backup.mkdir()
        shutil.copytree(live, backup / "frame_classifier_savedmodel")
        meta = models_dir / "model_metadata.json"
        if meta.exists():
            shutil.copy2(meta, backup / "model_metadata.json")
        print(f"[deploy] backed up current model -> {backup}")

    staged = models_dir / f".staging_{stamp}"
    shutil.copytree(run_dir / "frame_classifier_savedmodel", staged)
    if live.exists():
        shutil.rmtree(live.with_name(f".old_{stamp}"), ignore_errors=True)
        os.replace(live, live.with_name(f".old_{stamp}"))
    os.replace(staged, live)
    shutil.rmtree(live.with_name(f".old_{stamp}"), ignore_errors=True)
    shutil.copy2(run_dir / "model_metadata.json", models_dir / "model_metadata.json")
    print(f"[deploy] {models_dir / 'frame_classifier_savedmodel'} updated")


# ------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "train",
                    help="labelled frames: <video>/good|bad/ or good_frame|bad_frame/")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="run directory (default: training/runs/<UTC timestamp>)")
    ap.add_argument("--run-name", type=str, default="", help="label for this run, recorded in metadata")
    ap.add_argument("--deploy", action="store_true",
                    help="after verification passes, install into instagram_analyzer_app/models/ (backs up first)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-split", type=float, default=0.2, help="fraction of VIDEOS held out")
    ap.add_argument("--holdout-videos", type=str, default="",
                    help="comma-separated video folder names to use as validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    ap.add_argument("--dark-threshold", type=int, default=DEFAULT_DARK_THRESHOLD)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="GOOD cutoff used for all reporting (config.classifier_threshold)")
    ap.add_argument("--min-recall", type=float, default=0.90,
                    help="recall floor when picking the cheapest safe threshold")
    ap.add_argument("--no-dark-boost", action="store_true", help="skip the low-light boost")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--dedup", action="store_true",
                    help="drop near-duplicate frames within each video, after the split")
    ap.add_argument("--dedup-hash-size", type=int, default=DEFAULT_PHASH_SIZE)
    ap.add_argument("--dedup-distance", type=int, default=DEFAULT_PHASH_DISTANCE)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--cache-file", type=str, default="",
                    help="spill the decoded-image cache to this path instead of RAM")
    args = ap.parse_args()

    if sys.version_info >= (3, 13):
        sys.exit(f"error: TensorFlow has no wheels for Python {sys.version_info.major}."
                 f"{sys.version_info.minor}; use Python <= 3.12 (see training/requirements.txt)")

    if args.no_dark_boost:
        print("[warn] --no-dark-boost trains a model the app cannot serve correctly:\n"
              "       processing/frame_classifier._preprocess applies the boost unconditionally\n"
              "       and config.py has no switch. Patch both before deploying such a model.")

    # Imported late: TF takes seconds to load, and --help should not pay for it.
    import tensorflow as tf
    import keras

    tf.keras.utils.set_random_seed(args.seed)
    random.seed(args.seed)
    dark_boost = not args.no_dark_boost

    items = discover_dataset(args.data_dir)
    print_dataset_summary(items)

    holdout = [v.strip() for v in args.holdout_videos.split(",") if v.strip()]
    train_items, val_items = group_split(items, args.val_split, args.seed, holdout)
    if not train_items:
        sys.exit("error: no training frames left after the split")

    if args.dedup:
        train_items = dedupe_within_videos(train_items, args.dedup_hash_size, args.dedup_distance)

    counts = np.bincount([i.label for i in train_items], minlength=2)
    class_weight = {i: float(len(train_items)) / (2.0 * max(int(c), 1)) for i, c in enumerate(counts)}
    if counts.min() and counts.max() / counts.min() > 1.5:
        print(f"[data] imbalance {counts.tolist()} -> class_weight {class_weight}")

    train_ds = build_dataset(tf, train_items, args.image_size, args.dark_threshold, dark_boost,
                             args.batch_size, not args.no_augment, args.seed, args.cache_file)
    val_ds = (build_dataset(tf, val_items, args.image_size, args.dark_threshold, dark_boost,
                            args.batch_size, False, args.seed,
                            args.cache_file + "_val" if args.cache_file else "")
              if val_items else None)

    model = build_model(keras, args.image_size, args.lr)
    model.summary()

    callbacks = []
    if val_ds is not None:
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                          restore_best_weights=True, verbose=1),
            # factor=0.5 is what produced v1's saved LR of 1e-3 * 0.5**4.
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                              min_lr=1e-6, verbose=1),
        ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        class_weight=class_weight, callbacks=callbacks)

    val_metrics = model.evaluate(val_ds, verbose=0, return_dict=True) if val_ds is not None else {}
    val_metrics = {k: round(float(v), 6) for k, v in val_metrics.items()}

    train_scores = score_items(model, train_items, args.image_size, args.dark_threshold, dark_boost)
    val_scores = (score_items(model, val_items, args.image_size, args.dark_threshold, dark_boost)
                  if val_items else np.asarray([]))

    sweep = threshold_report(val_items, val_scores, args.threshold, args.min_recall) if val_items else None
    if sweep is None:
        print("\n[warn] no validation videos — no threshold sweep, and no honest accuracy number.")

    run_dir = args.output_dir or (REPO_ROOT / "training" / "runs"
                                  / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    savedmodel_dir = export(keras, model, run_dir, args, items, train_items, val_items,
                            history, val_metrics, sweep)
    write_reports(run_dir, train_items, train_scores, val_items, val_scores, args.threshold, history)

    ok = verify(keras, savedmodel_dir, val_items or train_items, args)
    if sweep:
        _write_json(run_dir / "threshold_sweep.json", sweep)

    if args.deploy:
        if not ok:
            sys.exit("\nerror: refusing to deploy — verification failed")
        deploy(run_dir, REPO_ROOT / "instagram_analyzer_app" / "models")
    else:
        print(f"\nNot deployed. To install this model into the app:\n"
              f"    python training/train_frame_classifier.py --data-dir {args.data_dir} --deploy\n"
              f"or copy {savedmodel_dir} over instagram_analyzer_app/models/frame_classifier_savedmodel/")


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
