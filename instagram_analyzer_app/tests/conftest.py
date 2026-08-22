"""Pytest fixtures.

Smoke-test mode: clear DATABASE_URL so db_client uses its in-memory store,
and provide a SECRET_KEY so settings doesn't fail to load.
"""

import os
import sys
from pathlib import Path


# Make the app package importable when pytest is run from the repo root or
# from inside instagram_analyzer_app/.
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

os.environ.setdefault("SECRET_KEY", "x" * 32)
os.environ.setdefault("OPENROUTER_API_KEY", "")
os.environ.pop("DATABASE_URL", None)
