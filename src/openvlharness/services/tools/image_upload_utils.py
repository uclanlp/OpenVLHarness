"""Upload local images to cloud storage (GCS / S3) and return public URLs.

Uses content-hash-based blob names so that identical images are never uploaded
twice — even across server restarts or experiment runs. Before uploading, a
HEAD request checks whether the object already exists on the remote.

An in-memory cache (keyed by SHA-256 of the image bytes) further avoids the
HEAD round-trip for repeated calls within the same process.
"""

import hashlib
import os
import time
import mimetypes
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Provider selection via environment variable
# ---------------------------------------------------------------------------
#   IMAGE_UPLOAD_PROVIDER = "gcs" | "s3"
#
# GCS env vars:
#   GCS_BUCKET_NAME                   (required)
#   GCS_PROJECT_ID                    (optional, usually inferred from key)
#   GOOGLE_APPLICATION_CREDENTIALS    (path to service account JSON key)
#
# S3 env vars:
#   AWS_S3_BUCKET_NAME                (required)
#   AWS_ACCESS_KEY_ID                 (required)
#   AWS_SECRET_ACCESS_KEY             (required)
#   AWS_S3_REGION                     (default: us-east-1)
# ---------------------------------------------------------------------------


# In-memory cache: content hash -> (url, timestamp)
# Signed URLs expire after 1 hour, so cache entries older than 50 min are evicted.
_upload_cache: Dict[str, Tuple[str, float]] = {}
_CACHE_TTL_SECONDS = 50 * 60  # 50 minutes (signed URLs valid for 60 min)


def _content_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def _get_cached_url(content_hash: str) -> Optional[str]:
    entry = _upload_cache.get(content_hash)
    if entry is None:
        return None
    url, ts = entry
    if time.time() - ts > _CACHE_TTL_SECONDS:
        del _upload_cache[content_hash]
        return None
    return url


def _set_cached_url(content_hash: str, url: str) -> None:
    _upload_cache[content_hash] = (url, time.time())


def _blob_name(content_hash: str, extension: str = ".jpg") -> str:
    """Deterministic blob name based on content hash."""
    return f"tool-uploads/{content_hash}{extension}"


def _guess_content_type(path: str) -> str:
    ct, _ = mimetypes.guess_type(path)
    return ct or "application/octet-stream"


# ---- GCS ------------------------------------------------------------------

def _upload_gcs(image_bytes: bytes, content_type: str, extension: str, content_hash: str) -> str:
    from google.cloud import storage as gcs_storage
    from datetime import timedelta

    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set. "
            "Point it to the service account JSON key file."
        )
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    if not bucket_name:
        raise RuntimeError("GCS_BUCKET_NAME is not set.")

    project_id = os.getenv("GCS_PROJECT_ID")
    client = gcs_storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    key = _blob_name(content_hash, extension)
    blob = bucket.blob(key)

    # Skip upload if object already exists
    if not blob.exists():
        blob.upload_from_string(image_bytes, content_type=content_type)

    url = blob.generate_signed_url(expiration=timedelta(hours=1), method="GET")
    return url


# ---- S3 -------------------------------------------------------------------

def _upload_s3(image_bytes: bytes, content_type: str, extension: str, content_hash: str) -> str:
    import boto3
    from botocore.exceptions import ClientError

    bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
    if not bucket_name:
        raise RuntimeError("AWS_S3_BUCKET_NAME is not set.")

    region = os.getenv("AWS_S3_REGION", "us-east-1")
    client = boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    key = _blob_name(content_hash, extension)

    # Skip upload if object already exists
    try:
        client.head_object(Bucket=bucket_name, Key=key)
    except ClientError:
        client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=image_bytes,
            ContentType=content_type,
        )

    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": key},
        ExpiresIn=3600,
    )
    return url


# ---- Public API -----------------------------------------------------------

def upload_image_bytes(
    image_bytes: bytes,
    content_type: str = "image/jpeg",
    extension: str = ".jpg",
    provider: Optional[str] = None,
) -> str:
    """Upload image bytes to cloud storage and return a public URL.

    Uses content-hash-based blob names for deduplication:
    - In-memory cache avoids repeated calls within the same process.
    - Remote HEAD check avoids re-uploading across server restarts.

    Args:
        image_bytes: Raw image data.
        content_type: MIME type of the image.
        extension: File extension (e.g. ".jpg", ".png").
        provider: "gcs" or "s3". Defaults to IMAGE_UPLOAD_PROVIDER env var.

    Returns:
        A publicly accessible (signed) URL.
    """
    ch = _content_hash(image_bytes)

    # Level 1: in-memory cache (avoids even a HEAD request)
    cached = _get_cached_url(ch)
    if cached:
        return cached

    # Level 2: remote check + conditional upload
    provider = (provider or os.getenv("IMAGE_UPLOAD_PROVIDER", "gcs")).lower()
    if provider == "gcs":
        url = _upload_gcs(image_bytes, content_type, extension, ch)
    elif provider == "s3":
        url = _upload_s3(image_bytes, content_type, extension, ch)
    else:
        raise ValueError(f"Unknown IMAGE_UPLOAD_PROVIDER: {provider!r}. Use 'gcs' or 's3'.")

    _set_cached_url(ch, url)
    return url


def upload_image_file(
    image_path: str,
    provider: Optional[str] = None,
) -> str:
    """Upload a local image file and return a public URL."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    content_type = _guess_content_type(image_path)
    ext = os.path.splitext(image_path)[1] or ".jpg"
    with open(image_path, "rb") as f:
        return upload_image_bytes(f.read(), content_type, ext, provider)
