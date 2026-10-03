"""HTTP error extraction for standalone tool clients."""
import json


DEFAULT_ERROR_BODY_LIMIT = 2048


def _stringify_detail(detail):
    """Convert structured HTTP error detail to readable text, preserving None."""
    if detail is None:
        return None
    if isinstance(detail, (dict, list)):
        try:
            return json.dumps(detail, ensure_ascii=True)
        except Exception:
            return str(detail)
    return str(detail)


def extract_error_detail(response, max_chars=DEFAULT_ERROR_BODY_LIMIT):
    """Return bounded server error text and decoded JSON, falling back to the response body."""
    payload = None
    detail = None
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if detail is None:
            error = payload.get("error")
            if isinstance(error, dict):
                detail = error.get("message") or error.get("detail")
        if detail is None:
            detail = payload.get("message")

    if detail is None:
        text = response.text.strip()
        if text:
            detail = text

    detail = _stringify_detail(detail)
    if detail and max_chars and len(detail) > max_chars:
        detail = f"{detail[:max_chars]}... (truncated)"
    return detail, payload


def raise_for_status_with_detail(response, prefix=None, max_chars=DEFAULT_ERROR_BODY_LIMIT):
    """Raise RuntimeError with status and server detail for an unsuccessful HTTP response."""
    if response.ok:
        return
    detail, _payload = extract_error_detail(response, max_chars=max_chars)
    message = f"{response.status_code} {detail or response.reason}"
    if prefix:
        message = f"{prefix}: {message}"
    raise RuntimeError(message)
