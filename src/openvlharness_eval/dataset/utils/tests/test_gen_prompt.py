"""The generation benchmarks must all state the task the same way.

Asking for an image is part of stating a text-to-image task correctly, so the thing worth
pinning is UNIFORMITY: KnowGen and WISE were fixed on 2026-07-15 while FactIP, SearchGen
and T2I-FactualBench kept bare prompts, which left three of the five under-measured by a
harness defect while their scores sat in the same table. Also pin the escape hatch, which
exists to reproduce pre-fix numbers -- not because bare prompts are the neutral baseline.
"""

import importlib.util
import os
import pathlib

_SPEC = importlib.util.spec_from_file_location(
    "gen_prompt", pathlib.Path(__file__).resolve().parents[1] / "gen_prompt.py"
)
gp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gp)


def test_wraps_by_default():
    out = gp.wrap_generation_prompt("a red apple")
    assert "MUST produce an image" in out
    assert out.endswith("a red apple")
    assert gp.generation_prompt_mode() == "wrapped"


def test_bare_mode_returns_the_prompt_untouched():
    os.environ["MVTK_GEN_BARE_PROMPT"] = "1"
    try:
        assert gp.wrap_generation_prompt("a red apple") == "a red apple"
        assert gp.generation_prompt_mode() == "bare"
    finally:
        del os.environ["MVTK_GEN_BARE_PROMPT"]


def test_all_five_generation_datasets_use_the_shared_wrapper():
    """Uniformity is the point -- a benchmark that forgets it is measured differently."""
    root = pathlib.Path(__file__).resolve().parents[2]
    for name in ("knowgen", "wise", "factip", "searchgen", "t2i_factualbench"):
        src = (root / f"{name}.py").read_text()
        assert "wrap_generation_prompt" in src, f"{name}.py does not use the shared wrapper"
        assert "MUST produce an image" not in src, (
            f"{name}.py carries its own copy of the instruction; use the shared wrapper "
            "so the protocol cannot drift per benchmark"
        )
