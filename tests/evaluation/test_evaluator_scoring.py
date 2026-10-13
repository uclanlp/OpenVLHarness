"""The counting, CountQA, image-MCQ and spatial-MCQ evaluators each score a 3-row
prediction file (two right, one wrong) to known metrics; the strict count parse
and integer extraction are checked on fixed tables.

Evaluators are instantiated without loading their benchmark (no download); the
prediction file carries the columns the real TSVs provide.
"""
import pandas as pd
import pytest

from evaluation_kit.dataset import DATASET_CLASSES
from evaluation_kit.dataset.utils.fsc147 import extract_integer_from_prediction, parse_count


def evaluator(name, rows=None):
    cls = next(c for c in DATASET_CLASSES if name in c.supported_datasets())
    obj = cls.__new__(cls)
    obj.dataset_name = name
    if rows is not None:  # loaders that re-read benchmark columns from self.data
        obj.data = pd.DataFrame(rows).drop(columns="prediction")
    return obj


def prediction_file(tmp_path, name, rows):
    path = tmp_path / name / "Model.xlsx"
    path.parent.mkdir()
    pd.DataFrame(rows).to_excel(path, index=False)
    return str(path)


COUNT = {"index": [0, 1, 2], "question": ["q"] * 3, "image_path": ["0.jpg", "1.jpg", "2.jpg"],
         "answer": [3, 5, 7], "prediction": ["3", "There are 6.", "<answer>7</answer>"]}
MCQ = {"index": [0, 1, 2], "question": ["q"] * 3, "image_path": ["0.jpg", "1.jpg", "2.jpg"],
       "A": ["cat", "dog", "cow"], "B": ["red", "blue", "green"], "C": ["x", "y", "z"], "D": ["p", "q", "r"],
       "answer": ["A", "B", "C"], "prediction": ["A", "(B) blue", "D"], "category": ["c"] * 3}


@pytest.mark.parametrize("name", ["FSC147_val", "PixmoCount_val"])
def test_counting_mae_rmse_accuracy(tmp_path, name):
    result = evaluator(name).evaluate(prediction_file(tmp_path, name, COUNT), model=None)
    assert dict(result) == {"mae": 0.33, "rmse": 0.58, "acc": 66.67}


def test_countqa_accuracy_and_relative_accuracy(tmp_path):
    result = evaluator("CountQA").evaluate(prediction_file(tmp_path, "CountQA", COUNT), model=None)
    assert {k: round(v, 2) for k, v in dict(result).items()} == {"acc": 66.67, "RA@5": 66.67, "RA@10": 66.67}


@pytest.mark.parametrize("name", ["FSC147_val", "PixmoCount_val", "CountQA"])
def test_counting_never_calls_a_judge_and_scores_unparseable_as_wrong(tmp_path, monkeypatch, name):
    obj = evaluator(name)
    monkeypatch.setattr(type(obj), "_extract_with_judge",
                        lambda *args, **kwargs: pytest.fail("counting must not call a language model"), raising=False)
    rows = {"index": [0, 1, 2], "question": ["q"] * 3, "image_path": ["0.jpg", "1.jpg", "2.jpg"],
            "answer": [0, 5, 7], "prediction": ["none", "5", "I first counted 7, then 6."]}
    result = obj.evaluate(prediction_file(tmp_path, name, rows), model="gpt-4o",
                          api_base="http://judge.invalid/v1/chat/completions")
    # "none" is unparseable, so it is wrong even though the true count is 0; the last
    # number (6) is the answer in the third row.
    assert result["acc"] == pytest.approx(33.33)


@pytest.mark.parametrize("text,expected", [
    ("12", 12), ("I first counted 12, but the final count is 14.", 14), ("\\boxed{42} after counting 40", 42),
    ("I count 1,092 beads", 1092), ("5.0", 5), ("2.5", None), ("seven", None), ("", None), ("nan", None),
    (float("nan"), None), (3, 3), ("Error: Unable to get response from the server.", None),
    ("Error: Reached the iteration limit without a final answer.", None),
    ("-5", None), ("The count is -5.", None), ("\\boxed{-3}", None), (-5, None), (-5.0, None),
    ("between 3-5 apples", 5), ("x-ray shows 4", 4),
])
def test_strict_count_parse(text, expected):
    assert parse_count(text) == expected


@pytest.mark.parametrize("name", ["FSC147_val", "PixmoCount_val", "CountQA"])
def test_negative_count_is_wrong(tmp_path, name):
    rows = {"index": [0], "question": ["q"], "image_path": ["0.jpg"], "answer": [5], "prediction": ["-5"]}
    result = evaluator(name).evaluate(prediction_file(tmp_path, name, rows), model=None)
    assert result["acc"] == 0


def test_iteration_limit_rows_are_failed_rows():
    from evaluation_kit.runner import _is_failed_prediction
    from openvlharness.agent.agent import MainAgent
    assert _is_failed_prediction(MainAgent.limit_msg) and _is_failed_prediction(MainAgent.fail_msg)


@pytest.mark.parametrize("name", ["RealWorldQA", "VStarBench"])
def test_image_mcq_exact_matching(tmp_path, name):
    result = evaluator(name, MCQ).evaluate(prediction_file(tmp_path, name, MCQ), model="exact_matching")
    assert result["Overall"].iloc[0] == pytest.approx(2 / 3)


@pytest.mark.parametrize("name", ["ERQA", "EmbSpatialBench", "MMSIBench_wo_circular", "MindCubeBench_tiny_raw_qa"])
def test_spatial_mcq_overall(tmp_path, name):
    result = evaluator(name).evaluate(prediction_file(tmp_path, name, MCQ), model=None)
    assert result["overall"] == pytest.approx(200 / 3)


@pytest.mark.parametrize("text,expected", [
    ("12", 12), ("12.7", 12), ("-5", -5), ("1e3", 1000), ("\\boxed{42}", 42), ("none", None),
    (float("nan"), None), ("I see seven birds", 7),
])
def test_integer_extraction(text, expected):
    assert extract_integer_from_prediction(text) == expected


@pytest.mark.parametrize("api_base", ["http://judge.test/v1/chat/completions", "http://judge.test/v1"])
def test_mmbrowsecomp_judge_accepts_chat_completions_url(tmp_path, monkeypatch, api_base):
    import openai
    from types import SimpleNamespace

    base_urls = []

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None):
            base_urls.append(base_url)
            reply = SimpleNamespace(content="OVERALL_CORRECTNESS: YES\nCHECKLIST_SCORE: 1/1")
            create = lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=reply)])  # noqa: E731
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    rows = {"index": [0], "question": ["Question: q"], "answer": ["a"], "prediction": ["a"]}
    result = evaluator("MMBrowseComp").evaluate(
        prediction_file(tmp_path, "MMBrowseComp", rows), model="judge", key="k", api_base=api_base)
    assert base_urls == ["http://judge.test/v1"]
    assert result["Overall"].iloc[0] == 100.0


def test_mcq_rescores_a_rule_based_cache_without_fingerprint(tmp_path):
    from evaluation_kit.smp import dump

    rows = {**MCQ, "prediction": ["A", "B", "C"]}
    path = prediction_file(tmp_path, "RealWorldQA", rows)
    # A cache written before fingerprints existed, from when row 1 had failed.
    stale = {0: {"hit": 1, "log": ""}, 1: {"hit": 0, "log": "Failed in Prefetch"}, 2: {"hit": 1, "log": ""}}
    dump(stale, str(tmp_path / "RealWorldQA" / "Model_exact_matching_result.pkl"))
    result = evaluator("RealWorldQA", rows).evaluate(path, model="exact_matching")
    assert result["Overall"].iloc[0] == pytest.approx(1.0)
