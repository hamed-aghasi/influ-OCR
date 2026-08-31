"""Upload ingress: extension allow-list, magic-byte sniffing, size cap,
ZIP-bomb guard, and what happens to the file when any of them reject.

These run against the real /upload route with an authenticated session; only
the Celery broker is stubbed, since enqueueing is not what is under test.
"""

import io
import zipfile

import pytest

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff" + b"\x00" * 64
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64


def _zip_bytes(members: dict) -> bytes:
    """Deflated, not stored — a bomb fixture is only meaningful if the
    compressed size is much smaller than the declared uncompressed size."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return buf.getvalue()


class _FakeResult:
    id = "task-123"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Logged-in client with uploads landing in tmp_path and no live broker."""
    from fastapi.testclient import TestClient

    import celery_app
    import main
    from processing.config import settings
    from processing.db_client import _memory_users, create_user

    monkeypatch.setattr(settings, "upload_dir", tmp_path)
    sent = []
    monkeypatch.setattr(
        celery_app.celery, "send_task", lambda *a, **k: sent.append((a, k)) or _FakeResult()
    )

    # Own the user store rather than relying on the startup hook: another test
    # module clears it, and whether admin exists would depend on file order.
    _memory_users.clear()
    create_user(settings.admin_username, settings.admin_password)

    with TestClient(main.app) as c:
        r = c.post(
            "/login",
            data={"username": settings.admin_username, "password": settings.admin_password},
        )
        assert r.status_code == 200, "fixture could not authenticate"
        c.sent_tasks = sent
        c.upload_dir = tmp_path
        yield c


def _post(client, filename, content, **overrides):
    form = {
        "campaign_date": "2026-08-31",
        "campaign_name": "spring",
        "product_name": "widget",
        "company": "acme",
    }
    form.update(overrides)
    return client.post(
        "/upload",
        files={"file": (filename, content, "application/octet-stream")},
        data=form,
        follow_redirects=False,
    )


# ----- happy path -----

def test_valid_png_upload_is_accepted_and_enqueued(client):
    r = _post(client, "shot.png", PNG)

    assert r.status_code == 200
    assert r.json()["job_id"]
    assert r.json()["task_id"] == "task-123"
    assert len(client.sent_tasks) == 1
    args = client.sent_tasks[0][1]["args"]
    assert args[2] == "image"                       # file_type
    assert list(client.upload_dir.glob("*.png"))    # file kept for the worker


def test_file_type_is_derived_from_extension(client):
    assert _post(client, "clip.mp4", MP4).status_code == 200
    assert client.sent_tasks[0][1]["args"][2] == "video"


def test_zip_upload_is_accepted(client):
    payload = _zip_bytes({"a.mp4": MP4})
    assert _post(client, "campaign.zip", payload).status_code == 200
    assert client.sent_tasks[0][1]["args"][2] == "zip"


# ----- auth -----

def test_upload_requires_authentication(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import main

    with TestClient(main.app) as anon:
        r = _post(anon, "shot.png", PNG)
    assert r.status_code == 401


def test_invalid_credentials_rejected(client):
    from processing.config import settings

    r = client.post(
        "/login",
        data={"username": settings.admin_username, "password": "definitely-wrong"},
    )
    assert r.status_code == 401
    assert "Invalid credentials" in r.text


# ----- extension allow-list -----

@pytest.mark.parametrize("name", ["payload.exe", "script.sh", "doc.pdf", "noext"])
def test_disallowed_extensions_rejected(client, name):
    r = _post(client, name, PNG)
    assert r.status_code == 400
    assert not client.sent_tasks


# ----- magic-byte sniffing -----

def test_extension_content_mismatch_rejected(client):
    """An executable renamed to .png must not reach the pipeline."""
    r = _post(client, "evil.png", b"MZ\x90\x00" + b"\x00" * 64)
    assert r.status_code == 400
    assert not client.sent_tasks


def test_zip_renamed_as_image_rejected(client):
    r = _post(client, "actually.png", _zip_bytes({"a.mp4": MP4}))
    assert r.status_code == 400


def test_image_renamed_as_zip_rejected(client):
    r = _post(client, "actually.zip", PNG)
    assert r.status_code == 400


def test_jpeg_and_png_magic_are_distinguished(client):
    assert _post(client, "shot.jpg", JPEG).status_code == 200
    assert _post(client, "shot2.jpg", PNG).status_code == 400


def test_sniff_accepts_containers_without_ftyp():
    """mkv/avi/mov have no shared magic at byte 4, so the sniffer trusts the
    extension for them — pinned so nobody 'tightens' it into a false reject."""
    from main import _sniff_ok

    assert _sniff_ok(b"anything at all", ".mkv") is True
    assert _sniff_ok(b"\x00\x00\x00\x18ftypqt  ", ".mp4") is True
    assert _sniff_ok(b"no magic here", ".mp4") is False
    assert _sniff_ok(PNG, ".png") is True
    assert _sniff_ok(b"random", ".txt") is False


# ----- rejected uploads leave nothing behind -----

def test_rejected_upload_leaves_no_file_on_disk(client):
    _post(client, "evil.png", b"MZ\x90\x00" + b"\x00" * 64)
    assert list(client.upload_dir.iterdir()) == []


def test_oversized_upload_is_capped_and_cleaned_up(client, monkeypatch):
    from processing.config import settings

    monkeypatch.setattr(settings, "max_upload_bytes", 1024)
    r = _post(client, "big.png", PNG + b"\x00" * 4096)

    assert r.status_code == 413
    assert list(client.upload_dir.iterdir()) == []   # partial write removed
    assert not client.sent_tasks


# ----- ZIP bomb guard -----

def test_zip_with_too_many_entries_rejected(client, monkeypatch):
    from processing.config import settings

    monkeypatch.setattr(settings, "max_zip_entries", 3)
    payload = _zip_bytes({f"f{i}.mp4": MP4 for i in range(5)})

    r = _post(client, "many.zip", payload)
    assert r.status_code == 400
    assert "too many entries" in r.text.lower()
    assert list(client.upload_dir.iterdir()) == []


def test_zip_bomb_by_uncompressed_size_rejected(client, monkeypatch):
    """The classic bomb: tiny compressed, enormous expanded."""
    from processing.config import settings

    monkeypatch.setattr(settings, "max_zip_uncompressed_bytes", 4096)
    payload = _zip_bytes({"huge.mp4": b"\x00" * 200_000})
    assert len(payload) < 4096, "fixture should be small compressed"

    r = _post(client, "bomb.zip", payload)
    assert r.status_code == 400
    assert list(client.upload_dir.iterdir()) == []
    assert not client.sent_tasks


def test_corrupt_zip_rejected(client):
    r = _post(client, "broken.zip", b"PK\x03\x04" + b"garbage" * 10)
    assert r.status_code == 400
    assert list(client.upload_dir.iterdir()) == []


def test_validate_zip_accepts_a_normal_archive(tmp_path):
    from main import _validate_zip

    z = tmp_path / "ok.zip"
    z.write_bytes(_zip_bytes({"a.mp4": MP4, "b.mp4": MP4}))
    _validate_zip(z)  # must not raise


# ----- form validation -----

def test_bad_date_format_rejected(client):
    r = _post(client, "shot.png", PNG, campaign_date="31-08-2026")
    assert r.status_code == 400
    assert not client.sent_tasks


def test_broker_failure_does_not_leave_a_phantom_job(client, monkeypatch):
    import celery_app

    def _broker_down(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(celery_app.celery, "send_task", _broker_down)

    r = _post(client, "shot.png", PNG)
    assert r.status_code == 503
    assert list(client.upload_dir.iterdir()) == []  # upload removed


# ----- ingress load -----

def test_500_video_uploads_within_one_minute(client):
    """Ingress load: 500 concurrent video uploads must all be accepted,
    enqueued exactly once each, with unique job ids, in under 60s.
    Processing itself is async (Celery) and out of scope here."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=32) as pool:
        responses = list(pool.map(lambda i: _post(client, f"clip_{i}.mp4", MP4), range(500)))
    elapsed = time.monotonic() - start

    assert elapsed < 60, f"500 uploads took {elapsed:.1f}s"
    assert all(r.status_code == 200 for r in responses)
    job_ids = [r.json()["job_id"] for r in responses]
    assert len(set(job_ids)) == 500                 # no job-id collisions
    assert len(client.sent_tasks) == 500            # every job enqueued exactly once
    assert all(k["args"][2] == "video" for _, k in client.sent_tasks)
    assert len(list(client.upload_dir.iterdir())) == 500  # every upload on disk
