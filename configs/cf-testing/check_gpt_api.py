"""Quick availability check for the GPT API config in gpt_agent.yaml.

Usage:
    python configs/cf-testing/check_gpt_api.py [--config PATH] [--prompt TEXT]
"""

import argparse
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_vlm_toolkit.agent import build_agent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "gpt_agent_with_tools.yaml"),
        help="Path to the agent yaml config.",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with the single word: pong.",
        help="Prompt text to send.",
    )
    args = parser.parse_args()

    print(f"[check] loading agent from {args.config}")
    try:
        agent = build_agent(args.config)
    except Exception:
        print("[check] FAILED to build agent")
        traceback.print_exc()
        return 1

    print(f"[check] base={agent.base} model={agent.model}")
    print(f"[check] sending prompt: {args.prompt!r}")
    try:
        result = agent.generate([{"type": "text", "value": args.prompt}])
    except Exception:
        print("[check] FAILED during generate()")
        traceback.print_exc()
        return 2
    print(result)
    prediction = result.get("prediction") if isinstance(result, dict) else result
    print("[check] OK — API reachable")
    print(f"[check] prediction: {prediction!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
