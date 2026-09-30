#!/usr/bin/env python
"""Read-only self-check of everything a run needs. Prints one line per check
and exits 1 if any failed. Never changes anything; never imported by the app.

    tools/doctor.py [--offline]        # on the host (.venv-train)
    docker compose run --rm -v ./tools:/tools worker python /tools/doctor.py   # inside the worker image

Checks: SECRET_KEY, OPENROUTER_API_KEY, ffmpeg, Tesseract + fas/eng, the
classifier loads and its digest, Postgres (if DATABASE_URL), Redis (if
reachable), and — unless --offline — that OpenRouter answers from HERE and
that OPENROUTER_MODEL is in its catalogue.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# The app dir: a checkout (tools/ beside instagram_analyzer_app/), the Docker
# image (/app, with tools/ bind-mounted anywhere), or the current directory.
APP = next((p for p in (REPO / "instagram_analyzer_app", REPO, Path("/app"), Path.cwd())
            if (p / "processing" / "config.py").exists()), REPO / "instagram_analyzer_app")

results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'OK  ' if ok else 'FAIL'}  {name:28} {detail}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="skip the OpenRouter checks")
    a = ap.parse_args(argv)

    os.environ.setdefault("SECRET_KEY", "doctor-placeholder-secret-key")
    sys.path.insert(0, str(APP))
    from processing.config import settings

    print(f"doctor: app at {APP}")
    real_secret = os.environ.get("SECRET_KEY") != "doctor-placeholder-secret-key"
    check("SECRET_KEY", real_secret and len(os.environ["SECRET_KEY"]) >= 16,
          "set, >= 16 chars" if real_secret else "not set in this environment (compose sets it for the containers)")
    check("OPENROUTER_API_KEY", bool(settings.openrouter_api_key), "set" if settings.openrouter_api_key else "missing: every job would fail at OCR")

    check("ffmpeg", shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "not on PATH (720p downscale would be skipped)")
    from processing.metric_coverage import tesseract_available
    try:
        ver = subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:  # noqa: BLE001
        ver = "not on PATH"
    check("tesseract + fas/eng", tesseract_available(), f"{ver}; coverage {'on' if tesseract_available() else 'would be skipped'}")

    model_dir = settings.model_dir / "frame_classifier_savedmodel"
    files = sorted(p for p in model_dir.rglob("*") if p.is_file() and p.name != ".DS_Store") if model_dir.exists() else []
    digest = hashlib.sha256()
    for p in files:
        digest.update(str(p.relative_to(model_dir)).encode() + b"\0" + p.read_bytes())
    meta = settings.model_dir / "model_metadata.json"
    run_id = ""
    if meta.exists():
        import json
        run_id = json.loads(meta.read_text()).get("run_id", "")
    check("classifier files", bool(files), f"{len(files)} files, run {run_id or '?'}, tree sha256 {digest.hexdigest()[:12]}" if files else f"{model_dir} missing")
    try:
        from processing.frame_classifier import load_model
        check("classifier loads", load_model() is not None, f"threshold {settings.classifier_threshold}")
    except Exception as exc:  # noqa: BLE001
        check("classifier loads", False, f"{type(exc).__name__}: {exc}")

    if settings.database_url:
        from processing.db_client import is_database_available
        check("postgres", is_database_available(), settings.database_url.split("@")[-1])
    else:
        check("postgres", True, "DATABASE_URL not set here (in-memory store; compose sets it)")
    try:
        import redis
    except ImportError:
        check("redis", True, "redis client not installed here (the containers have it; only the queue needs it)")
    else:
        try:
            r = redis.Redis.from_url(settings.broker_url, socket_connect_timeout=3)
            check("redis", bool(r.ping()), settings.broker_url)
        except Exception as exc:  # noqa: BLE001
            check("redis", False, f"{settings.broker_url}: {type(exc).__name__}")

    if not a.offline:
        import requests
        s = requests.Session()
        s.trust_env = False
        try:
            r = s.get("https://openrouter.ai/api/v1/models", timeout=20)
            if r.status_code == 403:
                check("openrouter route", False, "HTTP 403 'Access denied by security policy': this address is blocked (VPN?)")
            else:
                check("openrouter route", r.status_code == 200, f"HTTP {r.status_code}")
                ids = {m.get("id") for m in r.json().get("data", [])} if r.status_code == 200 else set()
                check("model in catalogue", settings.openrouter_model in ids,
                      settings.openrouter_model + ("" if settings.openrouter_model in ids else " NOT FOUND (aliases drift; use a catalogue id)"))
        except Exception as exc:  # noqa: BLE001
            check("openrouter route", False, f"{type(exc).__name__}: {exc}")

    failed = [n for n, ok, _ in results if not ok]
    print("all checks passed" if not failed else f"FAILED: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
