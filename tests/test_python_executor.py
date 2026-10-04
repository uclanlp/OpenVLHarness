"""Real subprocess checks for termination, output bounds and copied array bindings."""
import concurrent.futures
import subprocess
import time
import unittest
from unittest.mock import patch

import numpy as np

from openvlharness.services.tools.python_executor_local.executor import (
    CodeExecutionTimeoutError, MAX_OUTPUT_BYTES, MAX_VARIABLE_PREVIEW,
    MAX_VARIABLE_TOTAL, MAX_VARIABLES, execute_python_code, prepare_execution_context,
)


class PythonExecutorTests(unittest.TestCase):
    def test_deadline_kills_blocking_and_exception_catching_code_and_preview_rendering(self):
        programs = [
            "import time\nwhile True:\n    try:\n        time.sleep(10)\n    except BaseException:\n        continue",
            "while True:\n    try:\n        raise ValueError('again')\n    except BaseException:\n        continue",
            "class BadRepr:\n    def __repr__(self):\n        while True: pass\nvalue = BadRepr()",
        ]
        real_popen, processes = subprocess.Popen, []

        def record_process(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process

        with patch("openvlharness.services.tools.python_executor_local.executor.subprocess.Popen", record_process), \
             concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            for code in programs:
                with self.subTest(code=code):
                    started = time.monotonic()
                    future = pool.submit(execute_python_code, code, prepare_execution_context({}), 0.4)
                    with self.assertRaises(CodeExecutionTimeoutError):
                        future.result(timeout=3)
                    self.assertLess(time.monotonic() - started, 3)
                    self.assertIsNotNone(processes[-1].poll())

    def test_native_stdout_and_variable_previews_are_bounded(self):
        result = execute_python_code(
            "import os\nos.write(1, b'x' * 1000000)\n"
            "for i in range(200): globals()['variable_' + str(i)] = 'v' * 100000",
            prepare_execution_context({}), 3)
        self.assertLessEqual(len(result["printed_output"]), MAX_OUTPUT_BYTES)
        self.assertIn("truncated", result["printed_output"])
        self.assertLessEqual(len(result["variables"]), MAX_VARIABLES + 1)
        self.assertLessEqual(sum(len(k) + len(v) for k, v in result["variables"].items()), MAX_VARIABLE_TOTAL)
        self.assertTrue(all(len(v) <= MAX_VARIABLE_PREVIEW for v in result["variables"].values()))
        self.assertIn("__truncated__", result["variables"])

    def test_numpy_bindings_keep_type_shape_and_are_copied(self):
        array = np.arange(6, dtype=np.float32).reshape(2, 3)
        result = execute_python_code(
            "print(binding['depth'].dtype, binding['depth'].shape)\nbinding['depth'][:] = 99",
            prepare_execution_context({"binding": {"depth": array}}), 3)
        self.assertEqual(result["printed_output"], "float32 (2, 3)")
        np.testing.assert_array_equal(array, np.arange(6, dtype=np.float32).reshape(2, 3))

    def test_execution_exception_and_abrupt_child_exit_are_explicit(self):
        for code, message in [("1 / 0", "division by zero"), ("import os\ngetattr(os, '_exit')(3)", "without a result")]:
            with self.subTest(code=code), self.assertRaisesRegex(RuntimeError, message):
                execute_python_code(code, prepare_execution_context({}), 3)

    def test_invalid_deadlines_are_rejected(self):
        for timeout in (0, -1, None, True, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, "positive and finite"):
                execute_python_code("print(1)", prepare_execution_context({}), timeout)
