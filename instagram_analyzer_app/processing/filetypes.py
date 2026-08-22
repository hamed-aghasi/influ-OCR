"""Allowed upload extensions and file-type mapping, shared by web + poller."""

from typing import Optional


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
ZIP_EXTS = {".zip"}
ALLOWED_EXTS = VIDEO_EXTS | IMAGE_EXTS | ZIP_EXTS


def file_type_for(ext: str) -> Optional[str]:
    ext = ext.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in ZIP_EXTS:
        return "zip"
    if ext in IMAGE_EXTS:
        return "image"
    return None
