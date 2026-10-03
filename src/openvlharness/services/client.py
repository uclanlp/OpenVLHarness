"""HTTP tool transport. Capabilities select concrete names and supply call IDs."""
from urllib.parse import urlparse
from .http import _tool_execute_once, retry_with_exponential_backoff


class ToolClient:
    """HTTP client for the tool router's structured execute endpoint."""
    def __init__(self, base_url, *, timeout_seconds=600, retry=8):
        """Validate the router connection and configure transport retries.

        Args:
            base_url: HTTP(S) router base without an endpoint suffix.
            timeout_seconds: Positive timeout for each HTTP request.
            retry: Nonnegative number of additional attempts for transient failures.

        Raises:
            ValueError: Connection settings or retry limits are invalid.
        """
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment or parsed.username is not None:
            raise ValueError("services.client.base_url must be an HTTP(S) router base URL")
        if parsed.path.rstrip("/").endswith(("/execute", "/v1", "/chat/completions", "/responses")):
            raise ValueError("services.client.base_url must be the router base, without an endpoint suffix")
        if type(timeout_seconds) not in (int, float) or timeout_seconds <= 0:
            raise ValueError("services.client.timeout_seconds must be positive")
        if type(retry) is not int or retry < 0:
            raise ValueError("services.client.retry must be a nonnegative integer (additional attempts)")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retry = retry

    def execute(self, tool_name, args, *, tool_call_id):
        """Invoke a concrete tool and return its structured execute envelope.

        Args:
            tool_name: Concrete service tool name registered by the router.
            args: Resolved task arguments, without deployment settings.
            tool_call_id: Identifier correlating the request and result.

        Returns:
            A success or failure envelope. Invalid router envelopes become explicit
            invalid_tool_envelope failures; structured tool failures are preserved.

        Raises:
            RuntimeError: An unstructured HTTP error exhausts applicable retries.
            requests.RequestException: Transport attempts fail.
        """
        request = {"tool_name": tool_name, "args": args, "tool_call_id": tool_call_id}
        invoke = retry_with_exponential_backoff(_tool_execute_once, max_retries=self.retry)
        payload = invoke(base_url=self.base_url, tool_call=request, timeout=self.timeout_seconds)
        if not isinstance(payload, dict) or "ok" not in payload:
            return {"ok": False, "tool_name": tool_name, "tool_call_id": tool_call_id,
                    "result": None, "error": {"code": "invalid_tool_envelope",
                    "message": "The router did not return a structured execute envelope."}, "meta": {}}
        return payload
