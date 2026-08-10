"""Contract tests for generated-image attribution.

Why this exists: with ``--save-images files`` the trajectory writer decodes EVERY
data-URI image in message order, including images returned by search/retrieval
tools. The original resolver took the highest numeric suffix, so an agent that
only searched -- and never generated anything -- had its last RETRIEVED image
scored as its generation. That fabricates a non-zero score on a sample the model
actually failed, and KnowGen / WISE / FactIP / SearchGen all share the resolver.

Attribution now comes from the provenance block the image memory module writes
into the trajectory (``- tool_generated_image_2: tool_output:Image_Generation_Tool``),
which is present in already-completed runs too, so old runs can be re-scored
without re-running inference.
"""

import json
import os
import pathlib
import tempfile
import types

# Load only the pure resolver helpers: importing the package would pull in pandas
# and the rest of the eval stack, which this contract does not need.
_SRC = (pathlib.Path(__file__).resolve().parents[1] / "knowgen.py").read_text()
_START = _SRC.index("_GEN_TOOLS = (")
_END = _SRC.index("\n\n\n", _SRC.index("# cannot attribute"))
_STUB = (
    "import glob, json, os, re\nimport os.path as osp\n"
    "from typing import Optional\n\n" + _SRC[_START:_END]
)
_kg = types.ModuleType("kg_resolver")
exec(compile(_STUB, "kg_resolver", "exec"), _kg.__dict__)
resolve = _kg._resolve_generated_image

GEN = "tool_output:Image_Generation_Tool"
EDIT = "tool_output:Image_Edit_Tool"
SEARCH = "tool_output:Text_Search_Tool"


def _run_dir(refs, n_files, with_trajectory=True):
    root = tempfile.mkdtemp()
    ft = os.path.join(root, "full_trajectories")
    imgs = os.path.join(ft, "images")
    os.makedirs(imgs)
    for i in range(n_files):
        with open(os.path.join(imgs, f"T1_{i}.png"), "wb") as f:
            f.write(b"x")
    if with_trajectory:
        block = "Currently available image IDs for tool calls:\n" + "\n".join(
            f"- {r}: {s}" for r, s in refs
        )
        with open(os.path.join(ft, "T1.json"), "w") as f:
            json.dump(
                {"task_id": "T1",
                 "messages": [{"role": "user",
                               "content": [{"type": "text", "text": block}]}]},
                f,
            )
    return root


def test_search_only_trajectory_has_no_generated_image():
    """The bug: the last retrieved image used to be scored as the generation."""
    root = _run_dir([("tool_generated_image_1", SEARCH),
                     ("tool_generated_image_2", SEARCH)], 2)
    assert resolve(root, "T1") is None


def test_generation_as_last_tool_output_is_attributed():
    root = _run_dir([("tool_generated_image_1", SEARCH),
                     ("tool_generated_image_2", GEN)], 2)
    assert resolve(root, "T1") is not None


def test_edit_tool_counts_as_generation():
    root = _run_dir([("tool_generated_image_1", EDIT)], 1)
    assert resolve(root, "T1") is not None


def test_search_after_generation_is_not_attributed():
    """The last saved file is a search result, so it must not be scored."""
    root = _run_dir([("tool_generated_image_1", GEN),
                     ("tool_generated_image_2", SEARCH)], 2)
    assert resolve(root, "T1") is None


def test_missing_provenance_refuses_to_guess():
    root = _run_dir([], 2, with_trajectory=False)
    assert resolve(root, "T1") is None


def test_legacy_escape_hatch_restores_the_old_behaviour():
    """Kept only to reproduce pre-fix numbers on purpose."""
    root = _run_dir([("tool_generated_image_1", SEARCH)], 2)
    os.environ["MVTK_GEN_IMAGE_LEGACY_LASTFILE"] = "1"
    try:
        assert resolve(root, "T1") is not None
    finally:
        del os.environ["MVTK_GEN_IMAGE_LEGACY_LASTFILE"]
