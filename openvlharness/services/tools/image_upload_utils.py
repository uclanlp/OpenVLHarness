"""Publish image bytes for Serper Lens: GCS or S3 (time-limited signed URLs) or a local directory
served by ``openvlharness.services.image_host``.

SHA-256 object names and remote existence checks avoid redundant uploads.
A process-local cache reuses URLs for 50 minutes; signed URLs last one hour.
"""

import hashlib
import ipaddress
import logging
import os
import re
import stat
import tempfile
import time
from typing import Dict, Optional, Set, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider selection via environment variable
# ---------------------------------------------------------------------------
#   IMAGE_UPLOAD_PROVIDER = "gcs" | "s3" | "local"
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
    """Return the SHA-256 hex digest used to identify image content."""
    return hashlib.sha256(image_bytes).hexdigest()


def _get_cached_url(content_hash: str) -> Optional[str]:
    """Return an unexpired cached URL, evicting expired entries."""
    entry = _upload_cache.get(content_hash)
    if entry is None:
        return None
    url, ts = entry
    if time.time() - ts > _CACHE_TTL_SECONDS:
        del _upload_cache[content_hash]
        return None
    return url


def _set_cached_url(content_hash: str, url: str) -> None:
    """Cache a signed URL with its creation time."""
    _upload_cache[content_hash] = (url, time.time())


def _blob_name(content_hash: str, extension: str = ".jpg") -> str:
    """Deterministic blob name based on content hash."""
    return f"tool-uploads/{content_hash}{extension}"


# ---- GCS ------------------------------------------------------------------

def _upload_gcs(image_bytes: bytes, content_type: str, extension: str, content_hash: str) -> str:
    """Upload missing content to the configured GCS bucket and sign a one-hour GET URL.

    Requires GOOGLE_APPLICATION_CREDENTIALS and GCS_BUCKET_NAME; GCS_PROJECT_ID
    is optional. Missing required settings raise RuntimeError.
    """
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
    """Upload missing content to the configured S3 bucket and sign a one-hour GET URL.

    Uses AWS_S3_BUCKET_NAME, AWS_S3_REGION and configured AWS credentials.
    Raises RuntimeError when the bucket setting is absent.
    """
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


# ---- Local directory served by openvlharness.services.image_host -----------

class ImageHostUnreachableError(RuntimeError):
    """A published image cannot be fetched at its public URL, so Serper Lens cannot see it."""


# Reverse image search silently returns no matches when Serper cannot fetch the
# query image (HTTP 200 with an empty result set), so the router verifies the
# public URL itself: once per base URL before the first search of a process, and
# again whenever Lens returns no matches (see the image search tool).
_VERIFY_TIMEOUT_SEC = 15.0
_VERIFY_MAX_BYTES = 64 * 1024 * 1024
_HASH_NAME = re.compile(r"/([0-9a-f]{64})\.[A-Za-z0-9]{1,8}$")
_verified_base_urls: Set[str] = set()


def _is_private_host(hostname: str) -> bool:
    """True for localhost and private, loopback or link-local addresses."""
    if not hostname or hostname.lower() == "localhost" or hostname.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local or address.is_unspecified


def content_hash_from_url(url: str) -> Optional[str]:
    """Return the SHA-256 content hash encoded in an uploaded image's URL, if any."""
    match = _HASH_NAME.search(urlparse(url).path)
    return match.group(1) if match else None


def public_url_problem(url: str, content_hash: str, timeout: float = _VERIFY_TIMEOUT_SEC) -> Optional[str]:
    """Fetch ``url`` from this process and report why it does not serve the uploaded image.

    Returns None when the response is HTTP 200 and its body hashes to
    ``content_hash``, otherwise a short reason without the URL. Never raises.
    """
    import requests

    try:
        with requests.get(url, timeout=timeout, stream=True,
                          headers={"User-Agent": "OpenVLHarness image-host check"}) as response:
            if response.status_code != 200:
                return f"HTTP {response.status_code}"
            digest = hashlib.sha256()
            size = 0
            for chunk in response.iter_content(64 * 1024):
                size += len(chunk)
                if size > _VERIFY_MAX_BYTES:
                    return "the response is larger than any uploaded image"
                digest.update(chunk)
            if digest.hexdigest() != content_hash:
                return ("the body served at the URL is not the uploaded image "
                        "(an error page or another server answered)")
            return None
    except Exception as exc:  # noqa: BLE001 - any failure is a reason, never an exception
        return f"{type(exc).__name__}: {str(exc)[:200]}"


def forget_verified_base_url(url: str) -> None:
    """Re-verify the base URL of ``url`` on the next local upload."""
    parsed = urlparse(url)
    _verified_base_urls.discard(f"{parsed.scheme}://{parsed.netloc}")


def _verify_public_url(base_url: str, url: str, content_hash: str) -> None:
    """Raise ImageHostUnreachableError unless ``url`` serves the uploaded image."""
    from openvlharness.services import degradation

    if base_url in _verified_base_urls:
        return
    problem = public_url_problem(url, content_hash)
    if problem is None:
        _verified_base_urls.add(base_url)
        logger.info("Image host verified: %s serves uploaded images.", base_url)
        return
    degradation.record("image_host", tool="local_upload", reason="unreachable")
    logger.error(
        "IMAGE_HOST_UNREACHABLE url=%s -- %s. Serper Lens cannot fetch images from this host; "
        "check the image host, the tunnel and IMAGE_UPLOAD_PUBLIC_BASE_URL.", url, problem,
    )
    raise ImageHostUnreachableError(problem)


def _upload_local(image_bytes: bytes, content_type: str, extension: str, content_hash: str) -> str:
    """Write the image under IMAGE_UPLOAD_LOCAL_DIR and return its public URL.

    The directory is served by ``python -m openvlharness.services.image_host``;
    IMAGE_UPLOAD_PUBLIC_BASE_URL is the address at which that server is reachable
    from the internet (for example a tunnel URL). No cloud credentials are needed.
    The first upload per base URL is fetched back through that address.

    Raises:
        RuntimeError: The public base URL is not set, or names a private address
            that Serper cannot reach.
        ImageHostUnreachableError: The image is not served at its public URL.
    """
    base_url = os.getenv("IMAGE_UPLOAD_PUBLIC_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        raise RuntimeError(
            "IMAGE_UPLOAD_PUBLIC_BASE_URL is not set; it must be the public address of the image host."
        )
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise RuntimeError(f"IMAGE_UPLOAD_PUBLIC_BASE_URL must be an http(s) URL, got {base_url!r}.")
    if _is_private_host(parsed.hostname):
        raise RuntimeError(
            f"IMAGE_UPLOAD_PUBLIC_BASE_URL={base_url} is a private address. Serper Lens fetches "
            "query images from the internet; publish the image host through a tunnel and set "
            "the tunnel URL instead."
        )
    directory = prepare_private_directory(local_upload_dir())
    name = f"{content_hash}{extension}"
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        # A unique temporary file per call, so concurrent uploads of the same
        # image cannot remove each other's temporary file before the rename.
        fd, tmp_path = tempfile.mkstemp(prefix=f"{name}.", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(image_bytes)
            os.chmod(tmp_path, 0o644)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    url = f"{base_url}/{name}"
    _verify_public_url(base_url, url, content_hash)
    return url


def local_upload_dir() -> str:
    """Return the directory shared by the local provider and the image host."""
    return os.getenv("IMAGE_UPLOAD_LOCAL_DIR") or os.path.join(tempfile.gettempdir(), "openvlharness-uploads")


def prepare_private_directory(path: str) -> str:
    """Create ``path`` if needed and make sure it is a private directory this process owns.

    Rejects symbolic links and directories owned by another user, and sets the
    mode to 0700 whenever it differs. Returns the resolved path.

    Raises:
        RuntimeError: The path is a symbolic link, not a directory, or owned by someone else.
    """
    os.makedirs(path, mode=0o700, exist_ok=True)
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode):
        raise RuntimeError(f"Image upload directory must not be a symbolic link: {path}")
    if not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"Image upload directory is not a directory: {path}")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise RuntimeError(f"Image upload directory is owned by another user: {path}")
    if stat.S_IMODE(info.st_mode) != 0o700:
        os.chmod(path, 0o700)
    return os.path.realpath(path)


# ---- Public API -----------------------------------------------------------

def upload_image_bytes(
    image_bytes: bytes,
    content_type: str = "image/jpeg",
    extension: str = ".jpg",
    provider: Optional[str] = None,
) -> str:
    """Upload image bytes when needed and return a URL that Serper can fetch.

    Args:
        image_bytes: Encoded image bytes used for content-hash deduplication.
        content_type: MIME type stored on the remote object.
        extension: Filename extension appended to the content hash.
        provider: gcs, s3 or local; defaults to IMAGE_UPLOAD_PROVIDER, then gcs.

    Returns:
        A URL usable without additional request authentication. Signed cloud
        URLs expire after one hour (cached ones may have less remaining); local
        URLs stay valid while the image host is running.

    Raises:
        ValueError: The provider is unsupported.
        RuntimeError: Required storage settings are missing.

    Storage-client errors propagate. The in-memory cache is keyed by image bytes,
    so repeated calls reuse the cached URL even when other arguments differ.
    """
    ch = _content_hash(image_bytes)
    provider = (provider or os.getenv("IMAGE_UPLOAD_PROVIDER", "gcs")).lower()

    # The local provider is cheap (a file write) and its public base URL may
    # change when a tunnel restarts, so it is never served from the URL cache.
    if provider == "local":
        return _upload_local(image_bytes, content_type, extension, ch)

    # Level 1: in-memory cache (avoids even a HEAD request)
    cached = _get_cached_url(ch)
    if cached:
        return cached

    # Level 2: remote check + conditional upload
    if provider == "gcs":
        url = _upload_gcs(image_bytes, content_type, extension, ch)
    elif provider == "s3":
        url = _upload_s3(image_bytes, content_type, extension, ch)
    else:
        raise ValueError(f"Unknown IMAGE_UPLOAD_PROVIDER: {provider!r}. Use 'gcs', 's3' or 'local'.")

    _set_cached_url(ch, url)
    return url

