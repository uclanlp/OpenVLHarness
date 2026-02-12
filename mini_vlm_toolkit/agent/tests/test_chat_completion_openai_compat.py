from mini_vlm_toolkit.utils import api_utils


class _DummyResponse:
    def __init__(self, status_code=200, *, json_data=None, text="", reason="OK"):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {"choices": [{"message": {"content": "ok"}}]}
        self.text = text
        self.reason = reason

    @property
    def ok(self):
        return 200 <= int(self.status_code) < 300

    def json(self):
        return self._json_data


def test_chat_completion_openai_gpt5_payload_compat():
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _DummyResponse()

    original_post = api_utils.requests.post
    api_utils.requests.post = fake_post
    try:
        api_utils.chat_completion_request(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "hi"}],
            base_url="https://api.openai.com",
            api_key="KEY",
            temperature=0.0,
            max_tokens=128,
            reasoning_effort="medium",
            top_p=1.0,
            extra_body={"repetition_penalty": 1.2},
            timeout_seconds=17,
        )
    finally:
        api_utils.requests.post = original_post

    payload = captured["json"]
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer KEY"
    assert captured["timeout"] == 17
    assert payload["model"] == "gpt-5-mini"
    assert payload["max_completion_tokens"] == 128
    assert "max_tokens" not in payload
    assert "temperature" not in payload
    assert payload["top_p"] == 1.0
    assert payload["reasoning_effort"] == "medium"
    assert "repetition_penalty" not in payload


def test_chat_completion_openai_sanitizes_top_level_schema_combinators():
    captured = {}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Demo_Tool",
                "parameters": {
                    "type": "object",
                    "properties": {"image_path": {"type": "string"}},
                    "required": ["image_path"],
                    "oneOf": [
                        {"required": ["image_path"]},
                        {"required": ["image_base64"]},
                    ],
                },
            },
        }
    ]

    def fake_post(_url, json, headers, timeout):
        captured["json"] = json
        _ = headers, timeout
        return _DummyResponse()

    original_post = api_utils.requests.post
    api_utils.requests.post = fake_post
    try:
        api_utils.chat_completion_request(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "hi"}],
            base_url="https://api.openai.com",
            api_key="KEY",
            tools=tools,
        )
    finally:
        api_utils.requests.post = original_post

    sent_params = captured["json"]["tools"][0]["function"]["parameters"]
    assert "oneOf" not in sent_params
    # Ensure the caller-provided schema is not mutated.
    assert "oneOf" in tools[0]["function"]["parameters"]


def test_chat_completion_non_openai_keeps_legacy_payload_fields():
    captured = {}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Demo_Tool",
                "parameters": {
                    "type": "object",
                    "properties": {"image_path": {"type": "string"}},
                    "required": ["image_path"],
                    "oneOf": [
                        {"required": ["image_path"]},
                        {"required": ["image_base64"]},
                    ],
                },
            },
        }
    ]

    def fake_post(_url, json, headers, timeout):
        captured["json"] = json
        _ = headers, timeout
        return _DummyResponse()

    original_post = api_utils.requests.post
    api_utils.requests.post = fake_post
    try:
        api_utils.chat_completion_request(
            model="Qwen/Qwen3-VL-8B-Instruct",
            messages=[{"role": "user", "content": "hi"}],
            base_url="http://127.0.0.1:8222",
            api_key="EMPTY",
            temperature=0.0,
            max_tokens=96,
            reasoning_effort="medium",
            extra_body={"repetition_penalty": 1.1},
            tools=tools,
        )
    finally:
        api_utils.requests.post = original_post

    payload = captured["json"]
    assert payload["max_tokens"] == 96
    assert payload["temperature"] == 0.0
    assert payload["repetition_penalty"] == 1.1
    assert "max_completion_tokens" not in payload
    assert "reasoning_effort" not in payload
    assert "oneOf" in payload["tools"][0]["function"]["parameters"]


if __name__ == "__main__":
    test_chat_completion_openai_gpt5_payload_compat()
    test_chat_completion_openai_sanitizes_top_level_schema_combinators()
    test_chat_completion_non_openai_keeps_legacy_payload_fields()
    print("PASS: test_chat_completion_openai_gpt5_payload_compat")
    print("PASS: test_chat_completion_openai_sanitizes_top_level_schema_combinators")
    print("PASS: test_chat_completion_non_openai_keeps_legacy_payload_fields")
