"""HTTP status checks for standalone tool clients."""
from openvlharness.services.http import DEFAULT_ERROR_BODY_LIMIT, extract_error_detail


def raise_for_status_with_detail(response, prefix=None, max_chars=DEFAULT_ERROR_BODY_LIMIT):
    """Raise RuntimeError with status and server detail for an unsuccessful HTTP response.

    Unlike openvlharness.services.http.raise_for_status_with_detail, the error
    carries no status_code, so retry_with_exponential_backoff never retries it.
    """
    if response.ok:
        return
    detail, _payload = extract_error_detail(response, max_chars=max_chars)
    message = f"{response.status_code} {detail or response.reason}"
    if prefix:
        message = f"{prefix}: {message}"
    raise RuntimeError(message)
