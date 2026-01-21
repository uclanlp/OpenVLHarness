import requests


class ToolClientError(RuntimeError):
    def __init__(self, message, status_code=None, detail=None, payload=None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail
        self.payload = payload


def _extract_error_detail(response):
    payload = None
    try:
        payload = response.json()
    except ValueError:
        payload = None
    detail = None
    if isinstance(payload, dict):
        if payload.get("detail"):
            detail = payload.get("detail")
        elif isinstance(payload.get("error"), dict):
            error = payload.get("error")
            detail = error.get("message") or error.get("detail")
        elif payload.get("message"):
            detail = payload.get("message")
    if detail is None:
        text = response.text.strip()
        if text:
            detail = text
    return detail, payload


def _raise_for_status_with_detail(response):
    if response.ok:
        return
    detail, payload = _extract_error_detail(response)
    if detail:
        message = f"{response.status_code} {detail}"
    else:
        message = f"{response.status_code} {response.reason}"
    raise ToolClientError(
        message,
        status_code=response.status_code,
        detail=detail,
        payload=payload,
    )


class ToolClient:
    def __init__(self, base_url):
        if not base_url:
            raise ValueError("base_url is required")
        self.base_url = base_url.rstrip("/")

    def list_tools(self):
        response = requests.get(f"{self.base_url}/tools", timeout=30)
        _raise_for_status_with_detail(response)
        return response.json()

    def execute(self, tool_name, args):
        payload = {"tool_name": tool_name, "args": args}
        response = requests.post(f"{self.base_url}/execute", json=payload, timeout=300)
        _raise_for_status_with_detail(response)
        return response.json()
