"""Guards on an uploaded file before a job is created — the content sniff, the
ZIP limits, and discarding an upload the route has rejected. Shared by the web
route; the MinIO poller does not use them (see PIPELINE.md §1)."""

import zipfile
from pathlib import Path

from .config import settings
from .errors import APIError
from .filetypes import VIDEO_EXTS, ZIP_EXTS


VIDEO_MAGIC = b"ftyp"  # appears at byte 4 in MP4 family
ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


def _sniff_ok(header: bytes, ext: str) -> bool:
    if ext in ZIP_EXTS:
        return any(header.startswith(m) for m in ZIP_MAGIC)
    if ext in {".jpg", ".jpeg"}:
        return header.startswith(JPEG_MAGIC)
    if ext == ".png":
        return header.startswith(PNG_MAGIC)
    if ext in VIDEO_EXTS:
        return VIDEO_MAGIC in header[:32] or ext in {".mkv", ".avi", ".mov"}
    return False


def _validate_zip(zip_path: Path) -> None:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            entries = zf.infolist()
    except zipfile.BadZipFile as exc:
        raise APIError(400, "Invalid ZIP file") from exc

    if len(entries) > settings.max_zip_entries:
        raise APIError(400, f"ZIP has too many entries ({len(entries)})")

    total_uncompressed = sum(e.file_size for e in entries)
    if total_uncompressed > settings.max_zip_uncompressed_bytes:
        raise APIError(
            400,
            f"ZIP would expand to {total_uncompressed} bytes (max {settings.max_zip_uncompressed_bytes})",
        )


def discard_upload(path: Path) -> None:
    """Remove an upload the route has rejected; one already gone is fine."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
