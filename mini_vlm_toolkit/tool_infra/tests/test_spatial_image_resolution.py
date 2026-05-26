from pathlib import Path
import sys
import types


repo_root = Path(__file__).resolve().parents[3]
pkg_root = repo_root / "mini_vlm_toolkit"

mini_pkg = types.ModuleType("mini_vlm_toolkit")
mini_pkg.__path__ = [str(pkg_root)]
sys.modules.setdefault("mini_vlm_toolkit", mini_pkg)

utils_pkg = types.ModuleType("mini_vlm_toolkit.utils")
utils_pkg.__path__ = [str(pkg_root / "utils")]
sys.modules.setdefault("mini_vlm_toolkit.utils", utils_pkg)

api_utils_stub = types.ModuleType("mini_vlm_toolkit.utils.api_utils")
api_utils_stub.local_image_to_data_url = lambda _path: "data:image/png;base64,"
sys.modules.setdefault("mini_vlm_toolkit.utils.api_utils", api_utils_stub)

from mini_vlm_toolkit.memory.modules.image import ImageModule, register_image_data_url
from mini_vlm_toolkit.memory.run_state import ImagesState, MemoryRunState


def test_image_ids_list_resolves_to_base64s_and_preserves_refs():
    state = MemoryRunState()
    register_image_data_url(
        state.images,
        "input_image_1",
        "data:image/png;base64," + ("A" * 80),
        source="test",
    )
    register_image_data_url(
        state.images,
        "input_image_2",
        "data:image/png;base64," + ("B" * 80),
        source="test",
    )

    args, resolution = ImageModule().before_tool_call(
        agent=None,
        tool_name="Tracking_Tool",
        args={"target": "ball", "image_ids": ["input_image_1", "input_image_2"]},
        state=state,
    )

    assert resolution["errors"] == []
    assert "image_ids" not in args
    assert args["_image_ids"] == ["input_image_1", "input_image_2"]
    assert args["image_base64s"] == [
        "data:image/png;base64," + ("A" * 80),
        "data:image/png;base64," + ("B" * 80),
    ]


def main():
    test_image_ids_list_resolves_to_base64s_and_preserves_refs()
    print("Spatial image resolution tests completed.")


if __name__ == "__main__":
    main()
