"""
Tests that the hermes tool-call format round-trips correctly through the verl pipeline:

  model output  ->  HermesToolParser  ->  MiniVLMHTTPTool.execute  ->  ToolResponse
                                                                              |
                                                                 agent loop builds
                                                                 {"role": "tool", ...}

All tests mock the HTTP server so they run without a live SAM3 instance.
To test against a live server, set TOOL_SERVER_URL=http://127.0.0.1:8101.

Key concern this test addresses:
  The old mini_vlm_toolkit agent used OpenAI-style structured tool_calls objects,
  but verl uses the *hermes* format where tool calls appear as plain text inside
  <tool_call>{"name": ..., "arguments": ...}</tool_call> XML tags.
  This test verifies the hermes parser, tool adapter, and message format all work
  together correctly so the multi-turn rollout can proceed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import asyncio
import base64
import io
import json
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tiny_png_b64(color=(255, 0, 0), size=(8, 8)) -> str:
    """Return a base64-encoded PNG (no data-URL prefix)."""
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


TINY_PNG_B64 = _tiny_png_b64()


class _StringTokenizer:
    """Fake tokenizer whose decode() returns its input as-is (for parser tests)."""
    def decode(self, text):
        return text


def _load_tool():
    """Load MiniVLMHTTPTool from the project tool_config.yaml."""
    from verl.tools.utils.tool_registry import initialize_tools_from_config
    config_path = os.path.join(os.path.dirname(__file__), "../../rl/configs/tool_config.yaml")
    tools = initialize_tools_from_config(config_path)
    assert tools, "No tools loaded from tool_config.yaml"
    return tools[0]


def _ok_http_response(result: dict, tool_name="Visual_Grounding_Tool") -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "ok": True,
        "tool_name": tool_name,
        "result": result,
        "error": None,
        "tool_call_id": "test_call",
        "meta": {},
    }
    return mock_resp


def _err_http_response(code: str, message: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.json.return_value = {
        "ok": False,
        "tool_name": "Visual_Grounding_Tool",
        "result": None,
        "error": {"code": code, "message": message},
        "tool_call_id": "test_call",
        "meta": {},
    }
    return mock_resp


# ---------------------------------------------------------------------------
# 1.  HermesToolParser — unit tests
# ---------------------------------------------------------------------------

class TestHermesToolParser(unittest.IsolatedAsyncioTestCase):
    """HermesToolParser correctly extracts tool calls from model output text."""

    def _parser(self):
        from verl.experimental.agent_loop.tool_parser import HermesToolParser
        return HermesToolParser(_StringTokenizer())

    async def test_valid_tool_call_extracted(self):
        """A well-formed <tool_call> block yields a FunctionCall with correct name/args."""
        parser = self._parser()
        model_output = (
            "<tool_call>\n"
            '{"name": "Visual_Grounding_Tool", "arguments": {"query": "people"}}\n'
            "</tool_call>"
        )
        content, calls = await parser.extract_tool_calls(model_output)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "Visual_Grounding_Tool")
        args = json.loads(calls[0].arguments)
        self.assertEqual(args["query"], "people")

    async def test_no_tool_call_returns_empty_list(self):
        """Plain text output (no <tool_call> tags) returns an empty call list."""
        parser = self._parser()
        content, calls = await parser.extract_tool_calls("There are 5 people in the image.")
        self.assertEqual(calls, [])
        self.assertIn("5 people", content)

    async def test_malformed_json_skipped_gracefully(self):
        """Malformed JSON inside <tool_call> is skipped; no exception raised."""
        parser = self._parser()
        model_output = "<tool_call>{not: valid json}</tool_call>"
        content, calls = await parser.extract_tool_calls(model_output)
        self.assertEqual(calls, [])

    async def test_tool_call_tag_removed_from_content(self):
        """The <tool_call>...</tool_call> block is stripped from the returned content."""
        parser = self._parser()
        model_output = (
            "Let me check.\n"
            "<tool_call>\n"
            '{"name": "Visual_Grounding_Tool", "arguments": {"query": "car"}}\n'
            "</tool_call>"
        )
        content, calls = await parser.extract_tool_calls(model_output)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("<tool_call>", content)
        self.assertNotIn("</tool_call>", content)

    async def test_arguments_are_json_string(self):
        """FunctionCall.arguments is a JSON string (not a dict), matching verl contract."""
        parser = self._parser()
        model_output = (
            '<tool_call>{"name": "Visual_Grounding_Tool", '
            '"arguments": {"query": "blue basket", "image_path": "image_1"}}</tool_call>'
        )
        _, calls = await parser.extract_tool_calls(model_output)
        self.assertEqual(len(calls), 1)
        # arguments must be a str that parses as JSON
        self.assertIsInstance(calls[0].arguments, str)
        args = json.loads(calls[0].arguments)
        self.assertEqual(args["query"], "blue basket")


# ---------------------------------------------------------------------------
# 2.  MiniVLMHTTPTool — unit tests
# ---------------------------------------------------------------------------

class TestMiniVLMHTTPTool(unittest.IsolatedAsyncioTestCase):
    """MiniVLMHTTPTool correctly POSTs to /execute and converts the response."""

    async def test_posts_to_execute_endpoint(self):
        """execute() POSTs to <tool_base>/execute with the right JSON envelope."""
        tool = _load_tool()
        with patch("requests.post", return_value=_ok_http_response({"pred_scores": [0.9]})) as mock_post:
            iid, _ = await tool.create()
            await tool.execute(iid, {"query": "people", "image_base64": f"data:image/png;base64,{TINY_PNG_B64}"})

        mock_post.assert_called_once()
        url = mock_post.call_args[0][0]
        self.assertTrue(url.endswith("/execute"), f"Expected POST to /execute, got: {url}")
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["tool_name"], "Visual_Grounding_Tool")
        self.assertEqual(payload["args"]["query"], "people")

    async def test_ok_response_returns_text(self):
        """A successful server response produces a ToolResponse with valid JSON text."""
        tool = _load_tool()
        with patch("requests.post", return_value=_ok_http_response({"pred_scores": [0.9], "pred_boxes": [[0, 0, 10, 10]]})):
            iid, _ = await tool.create()
            response, reward, metrics = await tool.execute(
                iid, {"query": "people", "image_base64": f"data:image/png;base64,{TINY_PNG_B64}"}
            )

        self.assertIsNotNone(response.text)
        parsed = json.loads(response.text)
        self.assertTrue(parsed["ok"])
        self.assertEqual(metrics["exec_ok"], 1.0)

    async def test_rendered_image_extracted_into_tool_response(self):
        """rendered_image_b64 in the result is promoted to ToolResponse.image and scrubbed from text."""
        tool = _load_tool()
        result = {"pred_scores": [0.9], "rendered_image_b64": TINY_PNG_B64}
        with patch("requests.post", return_value=_ok_http_response(result)):
            iid, _ = await tool.create()
            response, _, _ = await tool.execute(
                iid, {"query": "people", "image_base64": f"data:image/png;base64,{TINY_PNG_B64}"}
            )

        self.assertIsNotNone(response.image)
        self.assertGreater(len(response.image), 0, "Expected at least one PIL image in ToolResponse.image")
        # The base64 blob must be scrubbed from the text to keep the token count low
        text_payload = json.loads(response.text)
        self.assertNotIn("rendered_image_b64", text_payload.get("result", {}),
                         "rendered_image_b64 should be removed from the text payload")

    async def test_server_error_returns_error_text(self):
        """A server ok=False response produces an error ToolResponse (no exception raised)."""
        tool = _load_tool()
        with patch("requests.post", return_value=_err_http_response("sam3_unavailable", "SAM3 not running")):
            iid, _ = await tool.create()
            response, _, metrics = await tool.execute(
                iid, {"query": "people", "image_base64": f"data:image/png;base64,{TINY_PNG_B64}"}
            )

        self.assertIn("Error", response.text)
        self.assertEqual(metrics["exec_ok"], 0.0)

    async def test_connection_error_returns_error_text(self):
        """A network-level failure produces an error ToolResponse (no exception raised)."""
        tool = _load_tool()
        with patch("requests.post", side_effect=ConnectionError("Connection refused")):
            iid, _ = await tool.create()
            response, _, metrics = await tool.execute(
                iid, {"query": "people", "image_base64": f"data:image/png;base64,{TINY_PNG_B64}"}
            )

        self.assertIn("Error", response.text)
        self.assertEqual(metrics["exec_ok"], 0.0)

    async def test_default_image_injected_from_agent_data(self):
        """When args have no image, the first image in agent_data.image_data is injected automatically."""
        tool = _load_tool()

        class FakeAgentData:
            image_data = [Image.new("RGB", (16, 16), (0, 200, 0))]

        with patch("requests.post", return_value=_ok_http_response({"pred_scores": [0.8]})) as mock_post:
            iid, _ = await tool.create()
            await tool.execute(iid, {"query": "green region"}, agent_data=FakeAgentData())

        payload = mock_post.call_args[1]["json"]
        self.assertIn("image_base64", payload["args"],
                      "Expected image_base64 to be injected from agent_data when not provided")

    async def test_image_path_ref_rewritten_to_base64(self):
        """image_path='image_1' is rewritten to image_base64 using agent_data.image_data."""
        tool = _load_tool()
        img = Image.new("RGB", (16, 16), (100, 100, 200))

        class FakeAgentData:
            image_data = [img]

        with patch("requests.post", return_value=_ok_http_response({"pred_scores": [0.7]})) as mock_post:
            iid, _ = await tool.create()
            await tool.execute(
                iid, {"query": "blue region", "image_path": "image_1"}, agent_data=FakeAgentData()
            )

        payload = mock_post.call_args[1]["json"]
        self.assertNotIn("image_path", payload["args"],
                         "image_path should be rewritten to image_base64")
        self.assertIn("image_base64", payload["args"])


# ---------------------------------------------------------------------------
# 3.  Full round-trip: parser -> tool adapter -> agent-loop message
# ---------------------------------------------------------------------------

class TestHermesToolRoundTrip(unittest.IsolatedAsyncioTestCase):
    """
    Simulates the verl multi-turn agent loop:
      1. Model generates hermes-format output containing <tool_call>
      2. HermesToolParser extracts the FunctionCall
      3. MiniVLMHTTPTool.execute() is called with the parsed arguments
      4. ToolResponse is converted to a 'tool' role message the way tool_agent_loop does it
    """

    async def test_text_only_round_trip(self):
        """Text-only tool response -> role=tool message with string content."""
        from verl.experimental.agent_loop.tool_parser import HermesToolParser

        parser = HermesToolParser(_StringTokenizer())
        model_output = (
            '<tool_call>{"name": "Visual_Grounding_Tool", "arguments": {"query": "people"}}</tool_call>'
        )
        _, calls = await parser.extract_tool_calls(model_output)
        self.assertEqual(len(calls), 1)

        tool = _load_tool()
        args = json.loads(calls[0].arguments)
        args["image_base64"] = f"data:image/png;base64,{TINY_PNG_B64}"

        with patch("requests.post", return_value=_ok_http_response({"count": 7})):
            iid, _ = await tool.create()
            tool_response, _, _ = await tool.execute(iid, args)

        # No image in result -> text-only tool message (agent loop line 306)
        self.assertIsNone(tool_response.image)
        tool_message = {"role": "tool", "content": tool_response.text or ""}
        self.assertEqual(tool_message["role"], "tool")
        self.assertIsInstance(tool_message["content"], str)
        parsed = json.loads(tool_message["content"])
        self.assertTrue(parsed["ok"])

    async def test_multimodal_round_trip(self):
        """Tool response with rendered image -> role=tool message with image+text content list."""
        from verl.experimental.agent_loop.tool_parser import HermesToolParser

        parser = HermesToolParser(_StringTokenizer())
        model_output = (
            '<tool_call>{"name": "Visual_Grounding_Tool", "arguments": {"query": "people"}}</tool_call>'
        )
        _, calls = await parser.extract_tool_calls(model_output)

        tool = _load_tool()
        args = json.loads(calls[0].arguments)
        args["image_base64"] = f"data:image/png;base64,{TINY_PNG_B64}"

        result = {"pred_scores": [0.92], "pred_boxes": [[10, 20, 50, 60]], "rendered_image_b64": TINY_PNG_B64}
        with patch("requests.post", return_value=_ok_http_response(result)):
            iid, _ = await tool.create()
            tool_response, _, _ = await tool.execute(iid, args)

        # Image present -> structured content list (agent loop lines 297-303)
        self.assertIsNotNone(tool_response.image)
        content = []
        if tool_response.image:
            content.append({"type": "image"})
        if tool_response.text:
            content.append({"type": "text", "text": tool_response.text})
        tool_message = {"role": "tool", "content": content}

        self.assertEqual(tool_message["role"], "tool")
        types = [c["type"] for c in tool_message["content"]]
        self.assertIn("image", types, "image token must come first in tool message")
        self.assertIn("text", types, "text payload must be present in tool message")
        # image before text (matches agent loop ordering)
        self.assertLess(types.index("image"), types.index("text"))

        # Sanity-check the text payload
        text_entry = next(c for c in content if c["type"] == "text")
        parsed = json.loads(text_entry["text"])
        self.assertTrue(parsed["ok"])
        # rendered_image_b64 scrubbed from text
        self.assertNotIn("rendered_image_b64", parsed.get("result", {}))


# ---------------------------------------------------------------------------
# 4.  Live server test (only runs when TOOL_SERVER_URL is set)
# ---------------------------------------------------------------------------

@unittest.skipUnless(os.environ.get("TOOL_SERVER_URL"), "Set TOOL_SERVER_URL=http://127.0.0.1:8101 to run")
class TestLiveServer(unittest.IsolatedAsyncioTestCase):
    """End-to-end test against a running SAM3 tool server."""

    async def test_live_visual_grounding(self):
        tool = _load_tool()
        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        iid, _ = await tool.create()
        response, _, metrics = await tool.execute(
            iid, {"query": "red region", "image_base64": f"data:image/png;base64,{b64}"}
        )
        print(f"\n[LIVE] text: {response.text[:300]}")
        print(f"[LIVE] images: {len(response.image or [])}")
        print(f"[LIVE] metrics: {metrics}")
        self.assertIsNotNone(response.text)


async def _run_live_call(image_path: str, query: str):
    """Call the live SAM3 server with a real image and print the full tool message."""
    import pprint

    print(f"\n{'='*60}")
    print(f"Live call: query={query!r}, image={image_path}")
    print(f"{'='*60}")

    tool = _load_tool()

    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    iid, _ = await tool.create()
    tool_response, _, metrics = await tool.execute(
        iid, {"query": query, "image_base64": f"data:image/png;base64,{b64}"}
    )

    # Build the tool message exactly as tool_agent_loop does
    if tool_response.image:
        content = []
        content.append({"type": "image"})
        if tool_response.text:
            content.append({"type": "text", "text": tool_response.text})
        tool_message = {"role": "tool", "content": content}
    else:
        tool_message = {"role": "tool", "content": tool_response.text or ""}

    print("\n--- tool_response.text ---")
    if tool_response.text:
        try:
            pprint.pprint(json.loads(tool_response.text))
        except Exception:
            print(tool_response.text)

    print(f"\n--- tool_response.image ---")
    print(f"  {len(tool_response.image or [])} PIL image(s) returned")
    if tool_response.image:
        for i, img_out in enumerate(tool_response.image):
            print(f"  [{i}] size={img_out.size}, mode={img_out.mode}")

    print("\n--- tool_message (as fed back into the conversation) ---")
    pprint.pprint(tool_message)

    print(f"\n--- metrics ---")
    pprint.pprint(metrics)
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run hermes tool round-trip tests or a live server call.")
    parser.add_argument(
        "--live",
        metavar="IMAGE_PATH",
        default=None,
        help="Skip unit tests and run a live SAM3 call with the given image path.",
    )
    parser.add_argument("--query", default="people", help="Query for the live call (default: people)")
    args, remaining = parser.parse_known_args()

    if args.live:
        asyncio.run(_run_live_call(args.live, args.query))
    else:
        import sys
        sys.argv = [sys.argv[0]] + remaining
        unittest.main(verbosity=2)
