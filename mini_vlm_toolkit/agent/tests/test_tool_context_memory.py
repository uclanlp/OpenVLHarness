import json
from pathlib import Path
import sys
import types

import numpy as np

repo_root = Path(__file__).resolve().parents[3]
pkg_root = repo_root / "mini_vlm_toolkit"

# Avoid importing mini_vlm_toolkit/__init__.py, which can pull optional deps.
mini_pkg = types.ModuleType("mini_vlm_toolkit")
mini_pkg.__path__ = [str(pkg_root)]
sys.modules.setdefault("mini_vlm_toolkit", mini_pkg)

memory_pkg = types.ModuleType("mini_vlm_toolkit.memory")
memory_pkg.__path__ = [str(pkg_root / "memory")]
sys.modules.setdefault("mini_vlm_toolkit.memory", memory_pkg)

utils_pkg = types.ModuleType("mini_vlm_toolkit.utils")
utils_pkg.__path__ = [str(pkg_root / "utils")]
sys.modules.setdefault("mini_vlm_toolkit.utils", utils_pkg)

api_utils_stub = types.ModuleType("mini_vlm_toolkit.utils.api_utils")
api_utils_stub.local_image_to_data_url = lambda _path: "data:image/png;base64,"
sys.modules.setdefault("mini_vlm_toolkit.utils.api_utils", api_utils_stub)

from mini_vlm_toolkit.memory.tool_context_memory import ToolContextMemory


def test_coding_context_ndarrays_are_json_safe_transport_values():
    memory = ToolContextMemory()
    depth = np.array([[1.25, 2.5], [3.75, 4.0]], dtype=np.float32)
    memory._tool_result_bindings = {
        "depth_scene": {
            "depth_meters": depth,
            "label": "scene",
        }
    }

    resolved, _info = memory.resolve_tool_call_args(
        agent=None,
        args={"code": "print(depth_scene['depth_meters'].shape)"},
        tool_name="Python_Executor_Tool",
    )

    json.dumps({"tool_name": "Python_Executor_Tool", "args": resolved})
    encoded = resolved["context_bindings"]["depth_scene"]["depth_meters"]
    assert encoded["__toolvision_type__"] == "ndarray"
    assert encoded["dtype"] == "float32"
    assert encoded["shape"] == [2, 2]
    assert isinstance(encoded["data_b64"], str)
    np.testing.assert_array_equal(
        memory._tool_result_bindings["depth_scene"]["depth_meters"],
        depth,
    )


def test_non_coding_tools_do_not_receive_transport_encoded_bindings():
    memory = ToolContextMemory()
    memory._tool_result_bindings = {
        "depth_scene": {
            "depth_meters": np.array([[1.0]], dtype=np.float32),
        }
    }

    resolved, _info = memory.resolve_tool_call_args(
        agent=None,
        args={"query": "read text"},
        tool_name="OCR_Tool",
    )

    assert resolved == {"query": "read text"}


if __name__ == "__main__":
    test_coding_context_ndarrays_are_json_safe_transport_values()
    test_non_coding_tools_do_not_receive_transport_encoded_bindings()
    print("Tool context memory tests passed.")
