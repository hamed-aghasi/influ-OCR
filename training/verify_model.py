#!/usr/bin/env python3
"""Verify a deployed classifier: file inventory, hashes, and a live prediction.

Run it after --deploy, and again inside the container, to prove the model that
the app loads is byte-for-byte the one that was trained.

    .venv-train/bin/python training/verify_model.py
    .venv-train/bin/python training/verify_model.py --expect <sha256>
"""
import argparse, hashlib, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
from _bootstrap import bootstrap  # placeholder SECRET_KEY, LOG_DIR, app on sys.path — before processing.*

bootstrap(secret="verify-only-not-a-real-secret", log_dir="/tmp/verify-logs")
from processing.config import settings  # noqa: E402


def tree_digest(d: Path):
    """One digest over every file's path and bytes, and the per-file list."""
    files, h = [], hashlib.sha256()
    for f in sorted(d.rglob("*")):
        if f.is_file():
            b = f.read_bytes()
            files.append((f.relative_to(d).as_posix(), len(b), hashlib.sha256(b).hexdigest()[:16]))
            h.update(f.relative_to(d).as_posix().encode()); h.update(b)
    return h.hexdigest(), files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", help="fail unless the model digest equals this")
    ap.add_argument("--predict", action="store_true", help="also load it and score a frame")
    a = ap.parse_args()

    mdir = settings.model_dir / "frame_classifier_savedmodel"
    meta_path = settings.model_dir / "model_metadata.json"
    if not mdir.is_dir():
        sys.exit(f"FAIL: no SavedModel at {mdir}")
    digest, files = tree_digest(mdir)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    print(f"model dir : {mdir}")
    print(f"run_id    : {meta.get('run_id')}  ({meta.get('model_version')})")
    print(f"threshold : {settings.classifier_threshold}")
    print(f"digest    : sha256 {digest}")
    for rel, size, h in files:
        print(f"   {size:>10,} B  {h}  {rel}")
    expected = {"fingerprint.pb", "saved_model.pb",
                "variables/variables.data-00000-of-00001", "variables/variables.index"}
    missing = expected - {rel for rel, _, _ in files}
    if missing:
        sys.exit(f"FAIL: missing from the SavedModel: {sorted(missing)}")
    if any(size == 0 for _, size, _ in files):
        sys.exit("FAIL: a file in the SavedModel is empty")

    if a.predict:
        import numpy as np
        from processing.frame_classifier import load_model
        m = load_model()
        if m is None:
            sys.exit("FAIL: the app could not load this model")
        out = m(np.zeros((1, settings.classifier_image_size, settings.classifier_image_size, 3), "float32"))
        v = next(iter(out.values())) if isinstance(out, dict) else out
        print(f"loads OK  : predicted {float(np.asarray(v).reshape(-1)[0]):.4f} on a blank frame")

    if a.expect and a.expect != digest:
        sys.exit(f"FAIL: digest {digest} != expected {a.expect}")
    print("OK")


if __name__ == "__main__":
    main()
