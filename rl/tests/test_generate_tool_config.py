import unittest
from unittest.mock import patch

from rl.tools.generate_tool_config import generate_tool_config


class GenerateToolConfigTest(unittest.TestCase):
    @patch("rl.tools.generate_tool_config._request_json")
    def test_generate_and_sanitize_required(self, mock_request_json):
        mock_request_json.side_effect = [
            {
                "OCR_Tool": {
                    "tool_name": "OCR_Tool",
                    "is_alias": False,
                    "canonical_tool_name": "OCR_Tool",
                }
            },
            [
                {
                    "type": "function",
                    "function": {
                        "name": "OCR_Tool",
                        "description": "ocr",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "image_base64": {"type": "string"},
                            },
                        },
                    },
                }
            ],
        ]

        out = generate_tool_config("http://127.0.0.1:8101", ["OCR_Tool"], 10.0)
        self.assertIn("tools", out)
        self.assertEqual(len(out["tools"]), 1)
        params = out["tools"][0]["tool_schema"]["function"]["parameters"]
        self.assertEqual(params["required"], [])
        self.assertEqual(params["type"], "object")


if __name__ == "__main__":
    unittest.main()
