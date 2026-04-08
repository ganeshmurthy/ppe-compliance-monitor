"""MinIO client helper for downloading model and video files."""

import io
import os
import time
from minio import Minio
from minio.commonconfig import CopySource
from minio.error import S3Error
from urllib.parse import urlparse
from logger import get_logger

log = get_logger(__name__)

# Config bucket for user uploads and thumbnails (enables horizontal scaling)
CONFIG_BUCKET = os.getenv("CONFIG_BUCKET", "config")


def get_config_bucket():
    """Return the MinIO bucket name for config uploads and thumbnails."""
    return CONFIG_BUCKET


def get_minio_client():
    """Create and return a MinIO client from environment variables."""
    endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
    # Minio() expects bare host:port; strip scheme if a full URL was provided
    parsed = urlparse(endpoint)
    if parsed.scheme in ("http", "https"):
        endpoint = parsed.netloc or parsed.path
        if parsed.scheme == "https":
            secure = True

    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )


def download_file(
    bucket: str,
    object_name: str,
    local_path: str,
    max_retries: int = 5,
    retry_delay: int = 3,
) -> str:
    """
    Download a file from MinIO to a local path.

    Args:
        bucket: MinIO bucket name
        object_name: Object key/path in the bucket
        local_path: Local filesystem path to save the file
        max_retries: Maximum number of retry attempts
        retry_delay: Seconds to wait between retries

    Returns:
        The local path where the file was saved

    Raises:
        S3Error: If download fails after all retries
    """
    client = get_minio_client()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    for attempt in range(max_retries):
        try:
            log.info(
                f"Downloading {bucket}/{object_name} to {local_path} (attempt {attempt + 1}/{max_retries})"
            )
            client.fget_object(bucket, object_name, local_path)
            log.info(f"Successfully downloaded {bucket}/{object_name}")
            return local_path
        except S3Error as e:
            if attempt < max_retries - 1:
                log.warning(
                    f"Download failed: {e}. Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
            else:
                log.error(f"Download failed after {max_retries} attempts: {e}")
                raise


def upload_file(
    bucket: str,
    object_name: str,
    file_path: str,
    content_type: str | None = None,
) -> None:
    """
    Upload a file from the local filesystem to MinIO.

    Args:
        bucket: MinIO bucket name
        object_name: Object key/path in the bucket
        file_path: Local filesystem path to the file
        content_type: Optional MIME type (e.g. 'video/mp4', 'image/jpeg')
    """
    client = get_minio_client()
    client.fput_object(bucket, object_name, file_path, content_type=content_type)
    log.info(f"Uploaded {file_path} to {bucket}/{object_name}")


def copy_object(
    dest_bucket: str,
    dest_key: str,
    src_bucket: str,
    src_key: str,
) -> None:
    """Server-side copy within or across buckets (same MinIO)."""
    client = get_minio_client()
    client.copy_object(dest_bucket, dest_key, CopySource(src_bucket, src_key))
    log.info(f"Copied s3://{src_bucket}/{src_key} to s3://{dest_bucket}/{dest_key}")


def upload_bytes(
    bucket: str,
    object_name: str,
    data: bytes,
    content_type: str | None = None,
) -> None:
    """
    Upload bytes to MinIO.

    Args:
        bucket: MinIO bucket name
        object_name: Object key/path in the bucket
        data: Raw bytes to upload
        content_type: Optional MIME type
    """
    client = get_minio_client()
    client.put_object(
        bucket,
        object_name,
        io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )
    log.info(f"Uploaded {len(data)} bytes to {bucket}/{object_name}")


def get_object_stream(bucket: str, object_name: str):
    """
    Get an object from MinIO as a stream.

    Returns:
        Response object with .read() and .stream() - use iter_content or read().
    """
    client = get_minio_client()
    return client.get_object(bucket, object_name)


def object_exists(bucket: str, object_name: str) -> bool:
    """Check if an object exists in MinIO."""
    try:
        client = get_minio_client()
        client.stat_object(bucket, object_name)
        return True
    except S3Error:
        return False
