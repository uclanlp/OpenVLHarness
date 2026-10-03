"""Regression cases discovered by the independent release review."""
import concurrent.futures
import contextlib
import json
from pathlib import Path
import signal
import sys
import threading
import unittest
from unittest.mock import Mock, patch

from openvlharness.bootstrap import load_agent_config
from openvlharness.capabilities.grounding import Capability as Grounding
from openvlharness.contracts import VirtualCall
from openvlharness.memory.store import MemoryStore
from openvlharness.services.client import ToolClient
from openvlharness.services.models import ModelAdapter
from openvlharness.services.tools.python_executor_local.executor import (
    CodeExecutionTimeoutError, execute_python_code, prepare_execution_context,
)


class ReleaseReviewRegressions(unittest.TestCase):
    @unittest.skipUnless(hasattr(signal, "setitimer"), "POSIX timers required")
    def test_fractional_coding_timeout_works_and_restores_previous_alarm(self):
        original_handler = signal.getsignal(signal.SIGALRM)
        original_timer = signal.getitimer(signal.ITIMER_REAL)
        previous_handler = lambda signum, frame: None
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, 10)
        try:
            result = execute_python_code("print(7)", prepare_execution_context({}), timeout_seconds=0.5)
            self.assertEqual(result["printed_output"], "7")
            self.assertIs(signal.getsignal(signal.SIGALRM), previous_handler)
            self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0], 0)
            with self.assertRaises(CodeExecutionTimeoutError):
                execute_python_code("while True:\n    pass", prepare_execution_context({}), timeout_seconds=0.05)
            self.assertIs(signal.getsignal(signal.SIGALRM), previous_handler)
            self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0], 0)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, original_handler)
            signal.setitimer(signal.ITIMER_REAL, *original_timer)

    def test_parallel_coding_keeps_stdout_per_question_and_restores_process_stdout(self):
        original_stdout = sys.stdout
        original_redirect = contextlib.redirect_stdout
        first_started = threading.Event()

        @contextlib.contextmanager
        def observed_redirect(stream):
            with original_redirect(stream):
                first_started.set()
                yield

        try:
            with patch("openvlharness.services.tools.python_executor_local.executor.contextlib.redirect_stdout", observed_redirect), \
                 concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                left = pool.submit(execute_python_code,
                    "import time\nprint('LEFT_START')\ntime.sleep(0.12)\nprint('LEFT_END')",
                    prepare_execution_context({}), 2)
                self.assertTrue(first_started.wait(2))
                right = pool.submit(execute_python_code,
                    "import time\nprint('RIGHT_START')\ntime.sleep(0.20)\nprint('RIGHT_END')",
                    prepare_execution_context({}), 2)
                left_result, right_result = left.result(timeout=5), right.result(timeout=5)
            restored = sys.stdout is original_stdout
        finally:
            # Even a regression must not corrupt pytest's own output capture.
            sys.stdout = original_stdout
        self.assertEqual(left_result["printed_output"], "LEFT_START\nLEFT_END")
        self.assertEqual(right_result["printed_output"], "RIGHT_START\nRIGHT_END")
        self.assertTrue(restored)

    def test_credentials_in_endpoint_userinfo_are_rejected_before_logging_or_requests(self):
        secret_url = "http://fixture-user:fixture-password@fixture.invalid:8222"
        with self.assertRaises(ValueError):
            ModelAdapter({"alias": "qwen3-vl-8b-instruct", "base": secret_url})
        with self.assertRaises(ValueError):
            ToolClient(secret_url)
        config = {"agent_name": "MainAgent", "agent": {"model_service": "orchestrator"},
            "memory": {"modules": [{"name": "image"}, {"name": "tool_context"}, {"name": "url"}]},
            "capabilities": {"registries": ["paper"], "tools": {"grounding": {"backend": "sam3_segment"}}},
            "services": {"models": {"orchestrator": {"alias": "qwen3-vl-8b-instruct", "base": "http://fixture.invalid/v1"}},
                "client": {"base_url": "http://fixture.invalid"}, "server": {},
                "tools": {"sam3_segment": {"server_url": secret_url}}}}
        with self.assertRaises(ValueError):
            load_agent_config(config)

    def test_invalid_grounding_result_fails_before_registering_images_or_bindings(self):
        image = json.loads((Path(__file__).parent / "fixtures/model_protocol/old_release.json").read_text())["image_data_url"]
        cases = [
            ("sam3_segment", {"orig_img_h": 100, "orig_img_w": 100, "pred_boxes": [["bad", 0, 1, 1]], "pred_scores": [1]}),
            ("sam3_segment", {"orig_img_h": 100, "orig_img_w": 100, "pred_boxes": [[0, 0, float("nan"), 1]], "pred_scores": [1]}),
            ("grounding_dino", {"size": [100], "boxes": [[0, 0, 1, 1]], "scores": [1]}),
        ]
        for backend, raw in cases:
            with self.subTest(backend=backend, raw=raw):
                raw["rendered_image_b64"] = image.split(",", 1)[1]
                client = Mock()
                client.execute.return_value = {"ok": True, "result": raw}
                memory = MemoryStore()
                memory.start([])
                capability = Grounding(backend=backend, client_config={"base_url": "http://fixture.invalid"}, client=client)
                result = capability.invoke(VirtualCall("retained-id", "Visual_Grounding_Tool", {"query": "object"}), memory)
                self.assertFalse(result.ok)
                self.assertEqual(result.payload["tool_call_id"], "retained-id")
                self.assertEqual(result.error["code"], "invalid_tool_result")
                self.assertEqual(memory.state.images.registry, {})
                self.assertEqual(memory.state.bindings.bindings, {})
                self.assertEqual(memory.state.images.tool_generated_counter, 0)


if __name__ == "__main__":
    unittest.main()
