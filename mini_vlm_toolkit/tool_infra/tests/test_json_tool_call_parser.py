import json

from client.json_tool_call_parser import parse_json_tool_call


def log_step(title):
    print(f"\n== {title} ==")


def log_output(label, payload):
    print(f"{label}:\n{json.dumps(payload, indent=2, sort_keys=True)}")


def test_parse_json_tool_call_pass_cases():
    log_step("Parse JSON tool call pass cases")
    payloads = [
        '{"tool_name": "DrawLine_Tool", "args": {"image_path": "/x", "box": [1, 2, 3, 4]}}',
        '{"name": "DrawLine_Tool", "arguments": {"image_path": "/x", "box": [1, 2, 3, 4]}}',
        (
            '{"name": "DrawLine_Tool", '
            '"arguments": "{\\"image_path\\": \\"/x\\", \\"box\\": [1, 2, 3, 4]}"}'
        ),
        (
            "Here is the call:\n"
            "```json\n"
            '{"tool_name": "DrawLine_Tool", "args": {"image_path": "/x"}}\n'
            "```\n"
        ),
        (
            '{"tool_calls": [{"type": "function", "function": '
            '{"name": "DrawLine_Tool", "arguments": {"image_path": "/x"}}}]}'
        ),
        '[{"name": "DrawLine_Tool", "arguments": {"image_path": "/x"}}]',
    ]

    for payload in payloads:
        parsed = parse_json_tool_call(payload)
        log_output("Parsed tool call", parsed)
        assert parsed["tool_name"] == "DrawLine_Tool"
        assert "image_path" in parsed["args"]


def test_parse_json_tool_call_fail_cases():
    log_step("Parse JSON tool call fail cases")
    payloads = [
        '{"tool_name": "DrawLine_Tool"}',
        '{"name": "DrawLine_Tool"}',
        '{"tool_name": "DrawLine_Tool", "args": []}',
        '{"name": "DrawLine_Tool", "arguments": 123}',
        '{"tool_calls": [{"name": "A_Tool", "arguments": {}}, {"name": "B_Tool", "arguments": {}}]}',
        (
            '{"tool_name": "A_Tool", "args": {}}'
            '{"tool_name": "B_Tool", "args": {}}'
        ),
    ]

    for payload in payloads:
        try:
            parse_json_tool_call(payload)
        except ValueError:
            continue
        raise AssertionError(f"Expected parse failure for: {payload}")


def main():
    test_parse_json_tool_call_pass_cases()
    test_parse_json_tool_call_fail_cases()
    print("JSON tool call parser tests completed.")


if __name__ == "__main__":
    main()
