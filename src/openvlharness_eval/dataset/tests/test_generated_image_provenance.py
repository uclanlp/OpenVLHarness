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

# ⛔ Import the REAL module, do not exec an extracted slice.
# The first version of this file pulled the resolver's source out of knowgen.py and
# exec'd it with its own injected `import re`. That masked a missing `import re` in
# knowgen.py itself, which made the entire openvlharness_eval.dataset package fail to
# import -- every benchmark dead, and this suite still green. A test that supplies the
# imports the module forgot cannot catch that class of bug.
#
# The package pulls in pandas and the rest of the eval stack, so skip cleanly when those
# are absent rather than silently falling back to a slice.
import pytest

knowgen = pytest.importorskip(
    "openvlharness_eval.dataset.knowgen",
    reason="needs the eval stack (pandas, validators, ...) installed",
)
resolve = knowgen._resolve_generated_image


def test_dataset_package_imports():
    """Guards the failure the slice-based version hid: a missing module-level import."""
    import openvlharness_eval.dataset  # noqa: F401


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


def test_search_after_generation_still_finds_the_generation():
    """Our prompts encourage verifying by search AFTER generating.

    Refusing whenever the newest ref is not a generation punished exactly those
    better-behaved trajectories. The k-th tool ref maps to the k-th saved file, so the
    generation is still locatable.
    """
    root = _run_dir([("tool_generated_image_1", GEN),
                     ("tool_generated_image_2", SEARCH)], 2)
    got = resolve(root, "T1")
    assert got is not None and got.endswith("T1_0.png")


def test_stale_files_from_an_earlier_run_are_refused():
    """Provenance lists 2 images but the directory holds 5 -- leftovers from a previous
    run. Filename-max would score the stale T1_4."""
    root = _run_dir([("tool_generated_image_1", SEARCH),
                     ("tool_generated_image_2", GEN)], 5)
    assert resolve(root, "T1") is None


def test_input_images_are_counted_before_tool_images():
    root = _run_dir([("input_image_1", "user_input"),
                     ("tool_generated_image_1", GEN)], 2)
    got = resolve(root, "T1")
    assert got is not None and got.endswith("T1_1.png")


def test_last_generation_wins_when_there_are_several():
    root = _run_dir([("tool_generated_image_1", GEN),
                     ("tool_generated_image_2", SEARCH),
                     ("tool_generated_image_3", EDIT)], 3)
    got = resolve(root, "T1")
    assert got is not None and got.endswith("T1_2.png")


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
