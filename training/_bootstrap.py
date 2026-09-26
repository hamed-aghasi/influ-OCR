"""Shared bootstrap for the training and evaluation scripts.

processing.config declares secret_key as required, so importing any processing
module without it raises. These scripts never touch the web layer; a placeholder
keeps them runnable with no .env (setdefault, so a real value wins). Settings are
read at import time, so everything here must run *before* the first
`from processing... import`, exactly as the inline blocks it replaces did.

Each script passes the values it used before, so where its logs and frames land
is unchanged.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "instagram_analyzer_app"


def bootstrap(*, secret: str, log_dir, processing_dir=None) -> None:
    """Placeholder SECRET_KEY, the script's LOG_DIR (and PROCESSING_DIR when it
    extracts frames), then put the app on sys.path."""
    os.environ.setdefault("SECRET_KEY", secret)
    os.environ.setdefault("LOG_DIR", str(log_dir))
    if processing_dir is not None:
        os.environ.setdefault("PROCESSING_DIR", str(processing_dir))
    sys.path.insert(0, str(APP))


def scrape_model_overrides() -> None:
    """--model-dir and --threshold must be in the environment before
    processing.* is imported, because Settings is built at import."""
    for flag, env in (("--model-dir", "MODEL_DIR"), ("--threshold", "CLASSIFIER_THRESHOLD")):
        if flag in sys.argv:
            os.environ[env] = sys.argv[sys.argv.index(flag) + 1]
