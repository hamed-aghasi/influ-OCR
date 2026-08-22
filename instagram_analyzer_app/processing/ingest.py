"""MinIO/S3 auto-ingest.

Watches `incoming/<company>/<campaign>/<product>/<file>` in the configured
bucket. The poller CLAIMS an object by moving it to `ingesting/…` (so
overlapping polls can't double-ingest), downloads it, and enqueues a job.
When the job finishes, `settle_object` moves it to `processed/…` or
`failed/…` — `failed/` objects can be retried by moving them back to
`incoming/`.

Known v1 limitation: if a job dies so hard the reaper has to fail it, its
object stays in `ingesting/` and must be moved back to `incoming/` by hand.

All functions take an optional `client` so tests can inject a fake.
"""

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import settings
from .filetypes import ALLOWED_EXTS, file_type_for
from .logger import get_logger


logger = get_logger(__name__)

_client = None
_client_lock = threading.Lock()


def is_ingest_configured() -> bool:
    return settings.minio_ingest_configured


def get_client():
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        if not is_ingest_configured():
            return None
        import boto3

        _client = boto3.client(
            "s3",
            endpoint_url=settings.minio_endpoint,
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key,
        )
        logger.info("MinIO ingest client initialized for bucket: %s", settings.minio_bucket)
        return _client


def parse_object_key(key: str) -> Optional[Dict[str, str]]:
    """incoming/<company>/<campaign>/<product>/<file.ext> → metadata dict.

    Returns None for anything that doesn't match the convention.
    """
    if not key.startswith(settings.minio_incoming_prefix):
        return None
    rest = key[len(settings.minio_incoming_prefix):]
    parts = rest.split("/")
    if len(parts) != 4 or not all(parts):
        return None
    company, campaign, product, filename = parts
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return None
    return {
        "company": company,
        "campaign_name": campaign,
        "product_name": product,
        "filename": filename,
        "ext": ext,
        "file_type": file_type_for(ext) or "",
    }


def list_incoming(client=None) -> List[Dict[str, Any]]:
    """List objects under incoming/ (key + size), skipping folder markers."""
    client = client or get_client()
    objects: List[Dict[str, Any]] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.minio_bucket, Prefix=settings.minio_incoming_prefix):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("/"):
                objects.append({"key": obj["Key"], "size": obj["Size"]})
    return objects


def _move(client, src_key: str, dest_key: str) -> None:
    client.copy_object(
        Bucket=settings.minio_bucket,
        Key=dest_key,
        CopySource={"Bucket": settings.minio_bucket, "Key": src_key},
    )
    client.delete_object(Bucket=settings.minio_bucket, Key=src_key)


def claim_object(key: str, client=None) -> Optional[str]:
    """Move incoming/… → ingesting/…; returns the new key, or None if the
    claim failed (e.g. another poller got there first)."""
    client = client or get_client()
    dest = settings.minio_ingesting_prefix + key[len(settings.minio_incoming_prefix):]
    try:
        _move(client, key, dest)
        return dest
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not claim %s: %s", key, exc)
        return None


def reject_object(key: str, client=None) -> None:
    """Move an unusable incoming/ object straight to failed/."""
    client = client or get_client()
    dest = settings.minio_failed_prefix + key[len(settings.minio_incoming_prefix):]
    try:
        _move(client, key, dest)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not move rejected object %s: %s", key, exc)


def settle_object(ingesting_key: str, success: bool, client=None) -> None:
    """Move ingesting/… → processed/… (job ok) or failed/… (job failed)."""
    client = client or get_client()
    if client is None:
        return
    prefix = settings.minio_processed_prefix if success else settings.minio_failed_prefix
    dest = prefix + ingesting_key[len(settings.minio_ingesting_prefix):]
    try:
        _move(client, ingesting_key, dest)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not settle %s → %s: %s", ingesting_key, dest, exc)


def download_object(key: str, dest_path: Path, client=None) -> None:
    client = client or get_client()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(settings.minio_bucket, key, str(dest_path))
