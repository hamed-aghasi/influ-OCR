"""S3-compatible storage (Liara).

All four LIARA_* env vars must be set for `is_s3_configured()` to return True.
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from .config import settings
from .logger import get_logger


logger = get_logger(__name__)

_s3_client = None
_client_lock = threading.Lock()


def is_s3_configured() -> bool:
    return settings.s3_configured


def get_s3_client():
    global _s3_client
    if _s3_client is not None:
        return _s3_client
    with _client_lock:
        if _s3_client is not None:
            return _s3_client
        if not is_s3_configured():
            return None
        try:
            import boto3

            _s3_client = boto3.client(
                "s3",
                endpoint_url=settings.liara_endpoint,
                aws_access_key_id=settings.liara_access_key,
                aws_secret_access_key=settings.liara_secret_key,
            )
            logger.info("S3 client initialized for bucket: %s", settings.liara_bucket_name)
            return _s3_client
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to create S3 client: %s", exc)
            return None


def upload_json(job_id: str, data: Dict[str, Any], filename: str = "instagram_metrics.json") -> bool:
    client = get_s3_client()
    if not client:
        return False
    try:
        client.put_object(
            Bucket=settings.liara_bucket_name,
            Key=f"{job_id}/{filename}",
            Body=json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("S3 upload_json failed: %s", exc)
        return False


def download_json(job_id: str, filename: str = "instagram_metrics.json") -> Optional[Dict[str, Any]]:
    client = get_s3_client()
    if not client:
        return None
    key = f"{job_id}/{filename}"
    try:
        response = client.get_object(Bucket=settings.liara_bucket_name, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))
    except client.exceptions.NoSuchKey:
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("S3 download_json failed: %s", exc)
        return None


def upload_file(job_id: str, file_path: Path, s3_filename: Optional[str] = None) -> bool:
    client = get_s3_client()
    if not client:
        return False
    try:
        with open(file_path, "rb") as f:
            client.put_object(
                Bucket=settings.liara_bucket_name,
                Key=f"{job_id}/{s3_filename or file_path.name}",
                Body=f.read(),
            )
        return True
    except (OSError, IOError) as exc:
        logger.error("S3 upload_file local read failed: %s", exc)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("S3 upload_file remote failed: %s", exc)
        return False


def get_file_url(job_id: str, filename: str = "instagram_metrics.json", expires_in: int = 3600) -> Optional[str]:
    client = get_s3_client()
    if not client:
        return None
    try:
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.liara_bucket_name, "Key": f"{job_id}/{filename}"},
            ExpiresIn=expires_in,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("S3 presign failed: %s", exc)
        return None


def delete_job_files(job_id: str) -> bool:
    client = get_s3_client()
    if not client:
        return False
    try:
        response = client.list_objects_v2(Bucket=settings.liara_bucket_name, Prefix=f"{job_id}/")
        for obj in response.get("Contents", []):
            client.delete_object(Bucket=settings.liara_bucket_name, Key=obj["Key"])
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("S3 delete_job_files failed: %s", exc)
        return False
