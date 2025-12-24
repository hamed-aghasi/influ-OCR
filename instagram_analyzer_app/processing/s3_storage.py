"""
S3 Storage Module

Handles file uploads and downloads to Liara S3-compatible storage.
"""

import os
import json
from typing import Optional, Dict, Any
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)

# S3 Configuration
S3_ENDPOINT = os.getenv('LIARA_ENDPOINT')
S3_ACCESS_KEY = os.getenv('LIARA_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('LIARA_SECRET_KEY')
S3_BUCKET = os.getenv('LIARA_BUCKET_NAME')

_s3_client = None


def is_s3_configured() -> bool:
    """Check if S3 credentials are configured."""
    return all([S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET])


def get_s3_client():
    """Get or create S3 client."""
    global _s3_client

    if _s3_client is not None:
        return _s3_client

    if not is_s3_configured():
        logger.warning("S3 not configured - missing environment variables")
        return None

    try:
        import boto3
        _s3_client = boto3.client(
            's3',
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY
        )
        logger.info(f"S3 client initialized for bucket: {S3_BUCKET}")
        return _s3_client
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}")
        return None


def upload_json(job_id: str, data: Dict[str, Any], filename: str = "instagram_metrics.json") -> bool:
    """
    Upload JSON data to S3.

    Args:
        job_id: Job ID (used as folder name)
        data: Dictionary to save as JSON
        filename: Name of the file

    Returns:
        True if successful, False otherwise
    """
    client = get_s3_client()
    if not client:
        logger.warning(f"S3 not available, skipping upload for {job_id}")
        return False

    try:
        key = f"{job_id}/{filename}"
        json_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')

        client.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json_bytes,
            ContentType='application/json'
        )

        logger.info(f"Uploaded {key} to S3")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        return False


def download_json(job_id: str, filename: str = "instagram_metrics.json") -> Optional[Dict[str, Any]]:
    """
    Download JSON data from S3.

    Args:
        job_id: Job ID (folder name)
        filename: Name of the file

    Returns:
        Dictionary with JSON data, or None if not found
    """
    client = get_s3_client()
    if not client:
        return None

    try:
        key = f"{job_id}/{filename}"
        response = client.get_object(Bucket=S3_BUCKET, Key=key)
        json_bytes = response['Body'].read()
        return json.loads(json_bytes.decode('utf-8'))
    except client.exceptions.NoSuchKey:
        logger.warning(f"File not found in S3: {key}")
        return None
    except Exception as e:
        logger.error(f"Failed to download from S3: {e}")
        return None


def upload_file(job_id: str, file_path: Path, s3_filename: Optional[str] = None) -> bool:
    """
    Upload a file to S3.

    Args:
        job_id: Job ID (used as folder name)
        file_path: Local path to the file
        s3_filename: Optional custom filename for S3

    Returns:
        True if successful, False otherwise
    """
    client = get_s3_client()
    if not client:
        return False

    try:
        filename = s3_filename or file_path.name
        key = f"{job_id}/{filename}"

        with open(file_path, 'rb') as f:
            client.put_object(
                Bucket=S3_BUCKET,
                Key=key,
                Body=f.read()
            )

        logger.info(f"Uploaded file {key} to S3")
        return True
    except Exception as e:
        logger.error(f"Failed to upload file to S3: {e}")
        return False


def get_file_url(job_id: str, filename: str = "instagram_metrics.json", expires_in: int = 3600) -> Optional[str]:
    """
    Generate a presigned URL for downloading a file.

    Args:
        job_id: Job ID (folder name)
        filename: Name of the file
        expires_in: URL expiration time in seconds (default 1 hour)

    Returns:
        Presigned URL string, or None if failed
    """
    client = get_s3_client()
    if not client:
        return None

    try:
        key = f"{job_id}/{filename}"
        url = client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': key},
            ExpiresIn=expires_in
        )
        return url
    except Exception as e:
        logger.error(f"Failed to generate presigned URL: {e}")
        return None


def delete_job_files(job_id: str) -> bool:
    """
    Delete all files for a job from S3.

    Args:
        job_id: Job ID (folder name)

    Returns:
        True if successful, False otherwise
    """
    client = get_s3_client()
    if not client:
        return False

    try:
        # List all objects with the job_id prefix
        response = client.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"{job_id}/")

        if 'Contents' not in response:
            return True  # No files to delete

        # Delete each object
        for obj in response['Contents']:
            client.delete_object(Bucket=S3_BUCKET, Key=obj['Key'])
            logger.info(f"Deleted {obj['Key']} from S3")

        return True
    except Exception as e:
        logger.error(f"Failed to delete job files from S3: {e}")
        return False
