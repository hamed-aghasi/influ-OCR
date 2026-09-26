"""The in-memory store used when DATABASE_URL is not set. One lock, three dicts;
every other submodule imports these by name and mutates them in place."""

import threading
from typing import Dict

_memory_lock = threading.RLock()
_memory_jobs: Dict[str, Dict] = {}
_memory_metrics: Dict[str, Dict] = {}
_memory_users: Dict[str, Dict] = {}
