import json
import os

import requests

from tests.utils import ensure_tool_server


SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8101")

CANONICAL_TOOLS = [
    "Visual_Grounding_Tool",
    "Depth_Estimation_Tool",
    "Zoom_In_Tool",
    "OCR_Tool",
    "Text_Search_Tool",
    "Image_Search_Tool",
    "Webpage_Visit_Tool",
    "Python_Coding_Agent_Tool",
]

ACCEPTED_CASES = {
    "Visual_Grounding_Tool": [
        "Visual_Grounding_Tool",
        "visual_grounding_tool",
        "VISUAL_GROUNDING_TOOL",
        "Visual Grounding Tool",
        "Visual-Grounding-Tool",
        "visualgroundingtool",
        "visualgrounding",
    ],
    "Depth_Estimation_Tool": [
        "Depth_Estimation_Tool",
        "depth_estimation_tool",
        "Depth Estimation Tool",
        "depthestimationtool",
        "depthestimation",
    ],
    "Zoom_In_Tool": [
        "Zoom_In_Tool",
        "zoom_in_tool",
        "Zoom In Tool",
        "zoomintool",
        "zoomin",
    ],
    "OCR_Tool": [
        "OCR_Tool",
        "ocr_tool",
        "OCR Tool",
        "ocrtool",
        "ocr",
    ],
    "Text_Search_Tool": [
        "Text_Search_Tool",
        "text_search_tool",
        "Text Search Tool",
        "textsearchtool",
        "textsearch",
    ],
    "Image_Search_Tool": [
        "Image_Search_Tool",
        "image_search_tool",
        "Image Search Tool",
        "imagesearchtool",
        "imagesearch",
    ],
    "Webpage_Visit_Tool": [
        "Webpage_Visit_Tool",
        "webpage_visit_tool",
        "Webpage Visit Tool",
        "webpagevisittool",
        "webpagevisit",
    ],
    "Python_Coding_Agent_Tool": [
        "Python_Coding_Agent_Tool",
        "python_coding_agent_tool",
        "Python Coding Agent Tool",
        "pythoncodingagenttool",
        "pythoncodingagent",
    ],
}

REJECTED_CASES = [
    "visual",
    "grounding",
    "search",
    "python",
    "visit",
    "agenttoolbox",
    "depthzoom",
]


def log_step(title):
    print(f"\n== {title} ==")


def log_output(label, payload):
    print(f"{label}:\n{json.dumps(payload, indent=2, sort_keys=True)}")


def _fetch_tools():
    response = requests.get(f"{SERVER_URL.rstrip('/')}/tools", timeout=10)
    response.raise_for_status()
    payload = response.json()
    assert isinstance(payload, dict), "Expected /tools to return a dict"
    return payload


def _execute_raw(tool_name, args=None):
    response = requests.post(
        f"{SERVER_URL.rstrip('/')}/execute",
        json={"tool_name": tool_name, "args": args or {}},
        timeout=60,
    )
    try:
        payload = response.json()
    except Exception:
        payload = {"raw_text": response.text}
    return response.status_code, payload


def test_required_tools_present():
    log_step("Check required tools on live router")
    tools = _fetch_tools()
    missing = [tool_name for tool_name in CANONICAL_TOOLS if tool_name not in tools]
    log_output("Available tools subset", {name: tools.get(name) for name in CANONICAL_TOOLS if name in tools})
    assert not missing, f"Live router is missing required tools: {missing}"


def test_accepted_name_variants_route_to_live_router():
    log_step("Accepted tool-name variants")
    for canonical_name, requested_names in ACCEPTED_CASES.items():
        for requested_name in requested_names:
            status_code, payload = _execute_raw(requested_name)
            print(f"{canonical_name} <- {requested_name}: status={status_code}")
            assert isinstance(payload, dict), f"Expected dict payload for {requested_name}, got {type(payload).__name__}"
            assert payload.get("tool_name") == requested_name, f"Expected echoed requested tool_name for {requested_name}"

            error = payload.get("error")
            error_code = error.get("code") if isinstance(error, dict) else None
            assert error_code not in {
                "tool_not_found",
                "tool_name_ambiguous_case",
                "tool_name_ambiguous_normalized",
            }, f"{requested_name} failed routing with error {error_code}: {payload}"


def test_rejected_name_variants_fail_cleanly():
    log_step("Rejected tool-name variants")
    for requested_name in REJECTED_CASES:
        status_code, payload = _execute_raw(requested_name)
        print(f"{requested_name}: status={status_code}")
        assert status_code == 404, f"Expected 404 for {requested_name}, got {status_code} payload={payload}"
        assert payload.get("tool_name") == requested_name
        error = payload.get("error")
        assert isinstance(error, dict), f"Expected structured error payload for {requested_name}"
        assert error.get("code") == "tool_not_found", f"Expected tool_not_found for {requested_name}, got {error}"


def main():
    if not ensure_tool_server(SERVER_URL, required_tools=CANONICAL_TOOLS):
        return
    test_required_tools_present()
    test_accepted_name_variants_route_to_live_router()
    test_rejected_name_variants_fail_cleanly()
    print("test_tool_name_matching: OK")


if __name__ == "__main__":
    main()
