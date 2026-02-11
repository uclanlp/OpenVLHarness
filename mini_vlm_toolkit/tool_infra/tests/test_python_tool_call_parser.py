import json

from client.python_tool_call_parser import parse_python_tool_call


def log_step(title):
    print(f"\n== {title} ==")


def log_output(label, payload):
    print(f"{label}:\n{json.dumps(payload, indent=2, sort_keys=True)}")


def test_parse_python_tool_call_pass_cases():
    log_step("Parse tool call pass cases")
    pass_cases = [
        "Python_Executor_Tool(code=\"print(12)\")",
        "execution = Python_Executor_Tool(code=\"print(13)\")",
        "```python\nPython_Executor_Tool(code=\"print(14)\")\n```",
        "Here:\nPython_Executor_Tool(code=\"print(15)\")\nDone.",
        "DrawLine_Tool(image_path=\"/tmp/a.png\", box=[1,2,3,4], corner_a=1, corner_b=3)",
        "<think>ignore</think>\nPython_Executor_Tool(code=\"print(16)\")",
    ]
    for payload in pass_cases:
        parsed = parse_python_tool_call(payload)
        log_output("Parsed tool call", parsed)
        assert parsed["tool_name"].endswith("_Tool")
        assert isinstance(parsed["args"], dict)


def test_parse_python_tool_call_fail_cases():
    log_step("Parse tool call fail cases")
    fail_cases = [
        "tool.execute(code=\"print(1)\")",
        "NotATool(code=\"print(2)\")",
        "Python_Executor_Tool(\"positional\")",
        "Python_Executor_Tool(code=\"print(3)\")\nDrawLine_Tool(image_path=\"/x\")",
    ]
    for payload in fail_cases:
        try:
            parse_python_tool_call(payload)
        except ValueError:
            continue
        raise AssertionError(f"Expected parse failure for: {payload}")


def main():
    test_parse_python_tool_call_pass_cases()
    test_parse_python_tool_call_fail_cases()
    print("Python tool-call parser tests completed.")


if __name__ == "__main__":
    main()
