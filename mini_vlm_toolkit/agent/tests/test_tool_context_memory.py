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

from mini_vlm_toolkit.memory import build_persistent_state_memory
from mini_vlm_toolkit.memory.run_state import BindingDescriptor, FieldDescriptor


def _seed_descriptors(memory):
    memory.state.bindings.bindings = {
        "depth_scene": {
            "depth_meters": np.array([[1.25, 2.5], [3.75, 4.0]], dtype=np.float32),
            "label": "scene",
        }
    }
    memory.state.bindings.descriptors = {
        "depth_scene": BindingDescriptor(
            key="depth_scene",
            source_tool="Depth_Estimation_Tool",
            source_args={},
            fields={
                "depth_meters": FieldDescriptor(
                    dtype="ndarray shape=(2, 2) dtype=float32",
                    description="per-pixel metric depth in meters",
                    shape_hint="",
                    preview="range=[1.25, 4]",
                ),
            },
        )
    }


def test_python_coding_agent_tool_receives_binding_descriptors():
    memory = build_persistent_state_memory(modules=[{"name": "image"}, {"name": "tool_context"}])
    _seed_descriptors(memory)

    resolved, _info = memory.resolve_tool_call_args(
        agent=None,
        args={"task": "compute the mean depth"},
        tool_name="Python_Coding_Agent_Tool",
    )

    assert "binding_descriptors" in resolved
    descriptors = resolved["binding_descriptors"]
    assert isinstance(descriptors, dict)
    assert "depth_scene" in descriptors
    depth_desc = descriptors["depth_scene"]
    assert depth_desc["source_tool"] == "Depth_Estimation_Tool"
    assert "depth_meters" in depth_desc["fields"]
    # Descriptors are JSON-safe metadata, not raw values.
    assert "depth_meters" not in resolved
    # Old context_bindings injection mechanism is gone.
    assert "context_bindings" not in resolved


def test_non_coding_agent_tools_receive_no_binding_descriptors():
    memory = build_persistent_state_memory(modules=[{"name": "image"}, {"name": "tool_context"}])
    _seed_descriptors(memory)

    resolved, _info = memory.resolve_tool_call_args(
        agent=None,
        args={"query": "read text"},
        tool_name="OCR_Tool",
    )

    assert "binding_descriptors" not in resolved
    assert "context_bindings" not in resolved
    assert resolved == {"query": "read text"}


def test_binding_descriptors_omitted_when_no_bindings_accumulated():
    memory = build_persistent_state_memory(modules=[{"name": "image"}, {"name": "tool_context"}])

    resolved, _info = memory.resolve_tool_call_args(
        agent=None,
        args={"task": "anything"},
        tool_name="Python_Coding_Agent_Tool",
    )

    assert "binding_descriptors" not in resolved


if __name__ == "__main__":
    test_python_coding_agent_tool_receives_binding_descriptors()
    test_non_coding_agent_tools_receive_no_binding_descriptors()
    test_binding_descriptors_omitted_when_no_bindings_accumulated()
    print("Tool context memory tests passed.")
