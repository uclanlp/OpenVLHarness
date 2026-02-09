from mini_vlm_toolkit.utils import api_utils


class _DummyResponse:
    def __init__(self, status_code, *, json_data=None, text="", reason="Error", json_raises=False):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self.reason = reason
        self._json_raises = json_raises

    @property
    def ok(self):
        return 200 <= int(self.status_code) < 300

    def json(self):
        if self._json_raises:
            raise ValueError("non-json")
        return self._json_data


def test_tool_execute_request_passthrough_structured_non_2xx():
    def fake_post(_url, json, timeout):
        assert timeout == 5
        assert json["tool_name"] == "Demo_Tool"
        return _DummyResponse(
            500,
            json_data={
                "ok": False,
                "result": None,
                "error": {"code": "tool_crash", "message": "worker crashed"},
                "tool_call_id": "call_1",
                "tool_name": "Demo_Tool",
                "meta": {"server": "tool-infra"},
            },
        )

    original_post = api_utils.requests.post
    api_utils.requests.post = fake_post
    try:
        payload = api_utils.tool_execute_request(
            base_url="http://tool-server",
            tool_call={"tool_name": "Demo_Tool", "args": {}, "tool_call_id": "call_1"},
            timeout=5,
        )
    finally:
        api_utils.requests.post = original_post

    assert payload["ok"] is False
    assert payload["error"]["code"] == "tool_crash"
    assert payload["tool_call_id"] == "call_1"
    assert payload["meta"]["http_status"] == 500


def test_tool_execute_request_fills_missing_ids_from_request():
    def fake_post(_url, json, timeout):
        assert timeout == 7
        assert json["tool_call_id"] == "call_2"
        return _DummyResponse(
            404,
            json_data={
                "ok": False,
                "result": None,
                "error": {"code": "tool_not_found", "message": "missing"},
                "meta": None,
            },
        )

    original_post = api_utils.requests.post
    api_utils.requests.post = fake_post
    try:
        payload = api_utils.tool_execute_request(
            base_url="http://tool-server",
            tool_call={"tool_name": "Lookup_Tool", "args": {"q": "x"}, "tool_call_id": "call_2"},
            timeout=7,
        )
    finally:
        api_utils.requests.post = original_post

    assert payload["ok"] is False
    assert payload["tool_name"] == "Lookup_Tool"
    assert payload["tool_call_id"] == "call_2"
    assert payload["meta"]["http_status"] == 404


def test_tool_execute_request_non_envelope_4xx_raises():
    def fake_post(_url, json, timeout):
        assert timeout == 3
        assert json["tool_call_id"] == "call_3"
        return _DummyResponse(
            400,
            json_data={"detail": "bad request"},
            text='{"detail":"bad request"}',
            reason="Bad Request",
        )

    original_post = api_utils.requests.post
    api_utils.requests.post = fake_post
    try:
        try:
            api_utils.tool_execute_request(
                base_url="http://tool-server",
                tool_call={"tool_name": "Demo_Tool", "args": {}, "tool_call_id": "call_3"},
                timeout=3,
            )
            raise AssertionError("Expected RuntimeError for non-envelope 400 response")
        except RuntimeError as exc:
            assert "Tool execution request failed" in str(exc)
            assert "400" in str(exc)
    finally:
        api_utils.requests.post = original_post


def test_tool_execute_request_ok_non_json_text_fallback():
    def fake_post(_url, json, timeout):
        assert timeout == 2
        assert json["tool_name"] == "Demo_Tool"
        return _DummyResponse(200, text="raw-text", json_raises=True)

    original_post = api_utils.requests.post
    api_utils.requests.post = fake_post
    try:
        payload = api_utils.tool_execute_request(
            base_url="http://tool-server",
            tool_call={"tool_name": "Demo_Tool", "args": {}, "tool_call_id": "call_4"},
            timeout=2,
        )
    finally:
        api_utils.requests.post = original_post

    assert payload["ok"] is True
    assert payload["result"] == "raw-text"
    assert payload["tool_name"] == "Demo_Tool"
    assert payload["tool_call_id"] == "call_4"
    assert payload["meta"]["warning"] == "non_json_response_from_execute"


if __name__ == "__main__":
    test_tool_execute_request_passthrough_structured_non_2xx()
    test_tool_execute_request_fills_missing_ids_from_request()
    test_tool_execute_request_non_envelope_4xx_raises()
    test_tool_execute_request_ok_non_json_text_fallback()
    print("PASS: test_tool_execute_request_passthrough_structured_non_2xx")
    print("PASS: test_tool_execute_request_fills_missing_ids_from_request")
    print("PASS: test_tool_execute_request_non_envelope_4xx_raises")
    print("PASS: test_tool_execute_request_ok_non_json_text_fallback")
