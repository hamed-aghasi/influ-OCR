def test_bcrypt_roundtrip():
    from processing.db_client import _hash_password, _verify_password

    h = _hash_password("hunter2")
    assert h.startswith("$2")
    assert _verify_password("hunter2", h)
    assert not _verify_password("wrong", h)


def test_legacy_sha256_still_verifies():
    """A user created under the old SHA-256 scheme should still be able to
    log in once; we rely on this to transparently migrate them."""
    import hashlib

    from processing.db_client import _is_legacy_hash, _verify_password

    legacy_hash = hashlib.sha256(b"instagram_analyzer_salt_2024hunter2").hexdigest()
    assert _is_legacy_hash(legacy_hash)
    assert _verify_password("hunter2", legacy_hash)
    assert not _verify_password("wrong", legacy_hash)


def test_create_and_verify_user_in_memory():
    from processing.db_client import (
        _memory_users,
        create_user,
        get_user_count,
        verify_user,
    )

    _memory_users.clear()
    assert get_user_count() == 0

    assert create_user("alice", "secret-pass-123") is True
    assert get_user_count() == 1

    # Duplicate insert should fail.
    assert create_user("alice", "other") is False

    assert verify_user("alice", "secret-pass-123") is True
    assert verify_user("alice", "wrong") is False
    assert verify_user("ghost", "anything") is False
