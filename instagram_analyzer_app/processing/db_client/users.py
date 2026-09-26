"""Users and password hashing (bcrypt, with transparent upgrade of legacy SHA-256)."""

from datetime import datetime

import bcrypt

from ..logger import get_logger
from .memory import _memory_lock, _memory_users
from .pool import _conn, is_database_available

logger = get_logger("processing.db_client")   # same name as the flat module: same Logger

def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, stored_hash: str) -> bool:
    if not stored_hash:
        return False
    if stored_hash.startswith("$2"):
        try:
            return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
        except ValueError:
            return False
    # Legacy SHA-256 hash with hardcoded salt — verify and rehash on next login.
    import hashlib

    legacy = hashlib.sha256(f"instagram_analyzer_salt_2024{password}".encode()).hexdigest()
    return legacy == stored_hash


def _is_legacy_hash(stored_hash: str) -> bool:
    return bool(stored_hash) and not stored_hash.startswith("$2")


def create_user(username: str, password: str) -> bool:
    password_hash = _hash_password(password)

    if not is_database_available():
        with _memory_lock:
            if username in _memory_users:
                return False
            _memory_users[username] = {
                "username": username,
                "password_hash": password_hash,
                "created_at": datetime.now(),
            }
            return True

    try:
        with _conn() as c, c.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (username, password_hash),
            )
            return True
    except Exception as exc:  # noqa: BLE001
        logger.error("create_user failed: %s", exc)
        return False


def _update_user_password(username: str, password: str) -> None:
    new_hash = _hash_password(password)
    if not is_database_available():
        with _memory_lock:
            user = _memory_users.get(username)
            if user:
                user["password_hash"] = new_hash
        return
    try:
        with _conn() as c, c.cursor() as cur:
            cur.execute(
                "UPDATE users SET password_hash=%s WHERE username=%s",
                (new_hash, username),
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("rehash failed for %s: %s", username, exc)


def verify_user(username: str, password: str) -> bool:
    if not is_database_available():
        with _memory_lock:
            user = _memory_users.get(username)
        if not user:
            return False
        stored = user["password_hash"]
    else:
        try:
            with _conn() as c, c.cursor() as cur:
                cur.execute("SELECT password_hash FROM users WHERE username=%s", (username,))
                row = cur.fetchone()
                if not row:
                    return False
                stored = row[0]
        except Exception as exc:  # noqa: BLE001
            logger.error("verify_user failed: %s", exc)
            return False

    if not _verify_password(password, stored):
        return False

    if _is_legacy_hash(stored):
        logger.info("Upgrading legacy password hash for %s", username)
        _update_user_password(username, password)
    return True


def get_user_count() -> int:
    if not is_database_available():
        with _memory_lock:
            return len(_memory_users)
    try:
        with _conn() as c, c.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM users")
            row = cur.fetchone()
            return int(row[0]) if row else 0
    except Exception as exc:  # noqa: BLE001
        logger.error("get_user_count failed: %s", exc)
        return 0
