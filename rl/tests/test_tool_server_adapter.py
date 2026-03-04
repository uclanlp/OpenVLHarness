import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

try:
    from rl.tools.tool_server_adapter import MiniVLMHTTPTool
    from verl.tools.schemas import OpenAIFunctionToolSchema

    _VERL_IMPORT_OK = True
except Exception:  # pragma: no cover - test guard for environments without verl installed
    _VERL_IMPORT_OK = False


_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2fF6UAAAAASUVORK5CYII="
)


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


@unittest.skipUnless(_VERL_IMPORT_OK, "verl package is required for adapter tests")
class ToolServerAdapterTest(unittest.TestCase):
    def _build_tool(self):
        schema = OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "OCR_Tool",
                    "description": "desc",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_path": {"type": "string", "description": "path"},
                            "image_base64": {"type": "string", "description": "base64"},
                        },
                        "required": [],
                    },
                },
            }
        )
        return MiniVLMHTTPTool(
            config={
                "type": "native",
                "tool_name": "OCR_Tool",
                "tool_base": "http://127.0.0.1:8101",
                "request_timeout_sec": 10,
            },
            tool_schema=schema,
        )

    @patch("rl.tools.tool_server_adapter.requests.post")
    def test_execute_rewrites_image_ref_and_extracts_image(self, mock_post):
        payload = {
            "ok": True,
            "tool_name": "OCR_Tool",
            "result": {
                "texts": ["hello"],
                "cropped_image_base64": _TINY_PNG_B64,
            },
            "error": None,
            "tool_call_id": "x",
            "meta": {},
        }
        mock_post.return_value = _FakeResponse(payload)

        tool = self._build_tool()
        image = Image.new("RGB", (2, 2), color="white")
        agent_data = SimpleNamespace(image_data=[image])
        response, reward, metrics = asyncio.run(
            tool.execute("instance_1", {"image_path": "image_1"}, agent_data=agent_data)
        )

        self.assertEqual(reward, 0.0)
        self.assertEqual(metrics["exec_ok"], 1.0)
        self.assertIsNotNone(response.image)
        self.assertEqual(len(response.image), 1)

        sent_args = mock_post.call_args.kwargs["json"]["args"]
        self.assertNotIn("image_path", sent_args)
        self.assertIn("image_base64", sent_args)


if __name__ == "__main__":
    unittest.main()
