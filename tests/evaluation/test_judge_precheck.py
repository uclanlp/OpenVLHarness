"""Benchmarks scored by an LLM judge are rejected before inference when no judge is configured."""
import inspect

import pytest

from evaluation_kit import cli
from evaluation_kit import dataset as datasets


NO_JUDGE = {"model": None}
EXACT = {"model": "exact_matching"}
JUDGE = {"model": "gpt-4o", "api_base": "https://judge.invalid/v1/chat/completions"}


@pytest.mark.parametrize("name", ["MMSearch", "hle", "RealXBench", "BrowseCompVL", "VDRBench"])
def test_judge_only_benchmark_without_judge_exits_before_inference(name):
    with pytest.raises(SystemExit, match="scored by an LLM judge.*--mode infer"):
        cli._check_judge([name], NO_JUDGE)


@pytest.mark.parametrize("name", ["FREAK_VQA", "SimpleVQA", "SimpleVQA_EN", "hle", "FVQA_test"])
def test_strict_benchmarks_also_reject_exact_matching(name):
    with pytest.raises(SystemExit, match="scored by an LLM judge"):
        cli._check_judge([name], EXACT)
    cli._check_judge([name], JUDGE)


@pytest.mark.parametrize("name", ["FSC147_val", "VStarBench", "ERQA", "ODinW13_bbox_val_qwen3"])
def test_judge_free_benchmarks_pass_without_judge(name):
    cli._check_judge([name], NO_JUDGE)


def test_flag_matches_the_evaluator_check():
    # Every class flagged for the pre-inference check calls require_judge in evaluate, and vice versa.
    for cls in datasets.IMAGE_DATASET:
        calls = "require_judge(" in inspect.getsource(cls.evaluate)
        with_flag = bool(getattr(cls, "REQUIRES_LLM_JUDGE", False))
        assert calls == with_flag, cls.__name__


def test_cli_exits_before_building_the_agent(monkeypatch, tmp_path):
    for name in ("TOOL_SERVER_URL", "SAM3_SERVER_URL", "OCR_SERVER_URL", "DEPTH_ESTIMATION_SERVER_URL",
                 "MODEL_BASE_URL", "AUXILIARY_MODEL_BASE_URL"):
        monkeypatch.setenv(name, "http://fixture.invalid:8080")
    monkeypatch.setenv("MODEL_API_KEY", "fixture-key")
    monkeypatch.setenv("AUXILIARY_MODEL_API_KEY", "fixture-key")
    monkeypatch.setenv("LMUData", str(tmp_path))

    def forbidden(*args, **kwargs):
        raise AssertionError("the agent must not be built")

    monkeypatch.setattr(cli.bootstrap, "build_agent", forbidden)
    with pytest.raises(SystemExit, match="MMSearch is scored by an LLM judge"):
        cli.main(["--data", "FSC147_val", "MMSearch", "--config", "configs/agent/qwen3_8b_paper.yaml",
                  "--judge-config", "configs/judge_no_judge.yaml", "--work-dir", str(tmp_path / "runs")])


def test_mmbrowsecomp_exact_match_mode_needs_no_judge():
    cli._check_judge(["MMBrowseComp"], {"model": None, "use_exact_match": True})
    with pytest.raises(SystemExit, match="scored by an LLM judge"):
        cli._check_judge(["MMBrowseComp"], NO_JUDGE)


def test_unresponsive_judge_is_an_error_not_a_key_error(monkeypatch, tmp_path):
    import pandas as pd
    from evaluation_kit.dataset import image_shortqa

    class Silent:
        def working(self):
            return False

    monkeypatch.setattr(image_shortqa, "build_judge", lambda **kwargs: Silent())
    cls = next(c for c in datasets.IMAGE_DATASET if "hle" in c.supported_datasets())
    obj = cls.__new__(cls)
    obj.dataset_name = "hle"
    path = tmp_path / "Model.xlsx"
    pd.DataFrame({"index": [0], "question": ["q"], "answer": ["a"], "prediction": ["a"]}).to_excel(path, index=False)
    with pytest.raises(RuntimeError, match="did not answer a test request"):
        obj.evaluate(str(path), **JUDGE)
