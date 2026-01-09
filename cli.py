import argparse
import json

from mini_vlm_tool_kit.client.client import ToolClient
from mini_vlm_tool_kit.tools.python_executor.tool import Python_Executor_Tool


def main():
    parser = argparse.ArgumentParser(description="Mini VLM Tool Kit CLI")
    parser.add_argument("--server", help="Server base URL, e.g. http://127.0.0.1:8001")
    parser.add_argument("--tool", required=True, help="Tool name, e.g. Python_Executor_Tool")
    parser.add_argument("--code", required=True, help="Python code to execute")
    parser.add_argument("--timeout", type=int, default=5, help="Execution timeout in seconds")
    args = parser.parse_args()

    if args.server:
        client = ToolClient(args.server)
        result = client.execute(
            args.tool,
            {"code": args.code, "timeout_seconds": args.timeout},
        )
    else:
        if args.tool != "Python_Executor_Tool":
            raise ValueError("Only Python_Executor_Tool is implemented for local execution")
        tool = Python_Executor_Tool()
        result = tool.execute(code=args.code, timeout_seconds=args.timeout)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
