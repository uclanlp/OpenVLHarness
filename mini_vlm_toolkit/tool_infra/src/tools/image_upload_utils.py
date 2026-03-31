"""Upload local images to cloud storage (GCS / S3) and return public URLs."""

import os
import uuid
import mimetypes
from typing import Optional, Tuple

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


def _generate_blob_name(extension: str = ".jpg") -> str:
    return f"tool-uploads/{uuid.uuid4().hex}{extension}"


def _guess_content_type(path: str) -> str:
    ct, _ = mimetypes.guess_type(path)
    return ct or "application/octet-stream"


# ---- GCS ------------------------------------------------------------------

def _upload_gcs(image_bytes: bytes, content_type: str, extension: str) -> str:
    from google.cloud import storage as gcs_storage

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
    blob_name = _generate_blob_name(extension)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(image_bytes, content_type=content_type)

    # Generate a signed URL valid for 1 hour
    from datetime import timedelta
    url = blob.generate_signed_url(expiration=timedelta(hours=1), method="GET")
    return url


# ---- S3 -------------------------------------------------------------------

def _upload_s3(image_bytes: bytes, content_type: str, extension: str) -> str:
    import boto3

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
    blob_name = _generate_blob_name(extension)
    client.put_object(
        Bucket=bucket_name,
        Key=blob_name,
        Body=image_bytes,
        ContentType=content_type,
    )

    # Generate a presigned URL valid for 1 hour
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": blob_name},
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

    Args:
        image_bytes: Raw image data.
        content_type: MIME type of the image.
        extension: File extension (e.g. ".jpg", ".png").
        provider: "gcs" or "s3". Defaults to IMAGE_UPLOAD_PROVIDER env var.

    Returns:
        A publicly accessible (signed) URL.
    """
    provider = (provider or os.getenv("IMAGE_UPLOAD_PROVIDER", "gcs")).lower()
    if provider == "gcs":
        return _upload_gcs(image_bytes, content_type, extension)
    elif provider == "s3":
        return _upload_s3(image_bytes, content_type, extension)
    else:
        raise ValueError(f"Unknown IMAGE_UPLOAD_PROVIDER: {provider!r}. Use 'gcs' or 's3'.")


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
