"""Judge caches must follow the judge and the predictions."""

import importlib.util
import json
import logging
import os.path as osp
import re

import pytest

if importlib.util.find_spec("evaluation_kit") is None:
    pytest.skip("Retained evaluator is available only from the research checkout", allow_module_level=True)

for dependency in ("pandas", "validators"):
    pytest.importorskip(dependency, reason="Retained evaluator dependencies are optional")

import pandas as pd

from evaluation_kit.dataset import image_shortqa, image_yorn
from evaluation_kit.dataset.utils import judge_cache
from evaluation_kit.dataset.utils.judge_cache import fingerprint_path, guard_cache
from evaluation_kit.dataset.utils.spatial_bench import cal_scores
from evaluation_kit.smp import dump, load


def frame(predictions, start=1):
    return pd.DataFrame({
        "index": list(range(start, start + len(predictions))),
        "prediction": list(predictions),
    })


@pytest.fixture
def cache(tmp_path):
    return str(tmp_path / "Replay_judge_cache.pkl")


@pytest.fixture
def log_records(monkeypatch, caplog):
    # The evaluation logger does not propagate; route the guard's messages to caplog.
    monkeypatch.setattr(judge_cache, "get_logger", lambda name: logging.getLogger("test_judge_cache"))
    caplog.set_level(logging.INFO, logger="test_judge_cache")
    return caplog


def test_unchanged_run_keeps_cache(cache, log_records):
    data = frame(["a", "b"])
    guard_cache(cache, data, "judge-a", prompt="hle")
    dump({1: "verdict-a", 2: "verdict-b"}, cache)
    guard_cache(cache, data.copy(), "judge-a", prompt="hle")
    assert load(cache) == {1: "verdict-a", 2: "verdict-b"}
    sidecar = load(fingerprint_path(cache))
    assert sidecar["judge_model"] == "judge-a" and sidecar["judge_prompt"] == "hle"
    assert sorted(sidecar["predictions"]) == ["1", "2"]
    assert not log_records.records


@pytest.mark.parametrize("judge_model,prompt", [("judge-b", "hle"), ("judge-a", "judge_vdr")])
def test_judge_or_prompt_change_deletes_cache(cache, log_records, judge_model, prompt):
    data = frame(["a", "b"])
    guard_cache(cache, data, "judge-a", prompt="hle")
    dump({1: "verdict-a", 2: "verdict-b"}, cache)
    guard_cache(cache, data, judge_model, prompt=prompt)
    assert not osp.exists(cache)
    assert load(fingerprint_path(cache))["judge_model"] == judge_model


@pytest.mark.parametrize("to_key", [int, str])
def test_changed_prediction_drops_only_that_key(cache, to_key):
    guard_cache(cache, frame(["a", "b", "c"]), "judge-a")
    dump({to_key(1): "v1", to_key(2): "v2", to_key(3): "v3"}, cache)
    guard_cache(cache, frame(["a", "B", "c"]), "judge-a")
    assert load(cache) == {to_key(1): "v1", to_key(3): "v3"}


def test_rows_left_out_of_a_run_keep_their_fingerprint(cache):
    guard_cache(cache, frame(["a", "b", "c"]), "judge-a")
    dump({1: "v1", 2: "v2", 3: "v3"}, cache)
    guard_cache(cache, frame(["a"]), "judge-a")
    assert load(cache) == {1: "v1", 2: "v2", 3: "v3"}
    guard_cache(cache, frame(["a", "b", "C"]), "judge-a")
    assert load(cache) == {1: "v1", 2: "v2"}


def test_table_cache_is_deleted_when_the_predictions_change(tmp_path):
    storage = str(tmp_path / "Replay_judge.xlsx")
    guard_cache(storage, frame(["a", "b"]), "judge-a", keyed=False)
    dump(frame(["a", "b"]), storage)
    guard_cache(storage, frame(["a", "b"]), "judge-a", keyed=False)
    assert osp.exists(storage)
    guard_cache(storage, frame(["a", "b", "c"]), "judge-a", keyed=False)
    assert not osp.exists(storage)


def test_cache_without_fingerprint_is_kept_with_a_warning(cache, log_records):
    dump({1: "old verdict"}, cache)
    guard_cache(cache, frame(["changed"]), "judge-a")
    assert load(cache) == {1: "old verdict"}
    warnings = [r for r in log_records.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert cache in warnings[0].getMessage() and "Delete it to re-judge" in warnings[0].getMessage()
    assert osp.exists(fingerprint_path(cache))

    log_records.clear()
    guard_cache(cache, frame(["changed"]), "judge-a")
    assert load(cache) == {1: "old verdict"}
    assert not log_records.records


def test_spatial_llm_cache_follows_the_judge(tmp_path, monkeypatch):
    monkeypatch.setattr(cal_scores, "_build_llm_judge", lambda judge_kwargs, task_name: object())
    seen = []

    def llm_fn(df, model, mode, max_retry, nproc, cache_file, key_col):
        cached = load(cache_file) if osp.exists(cache_file) else {}
        seen.append(sorted(cached))
        cached.update({k: ("A", p) for k, p in zip(df[key_col], df["prediction"])})
        dump(cached, cache_file)
        return df

    eval_file = str(tmp_path / "Replay.xlsx")
    for model in ("judge-a", "judge-a", "judge-b"):
        score_fn = cal_scores._build_score_fn(
            task_name="MCQ", judge_kwargs={"model": model}, rule_fn=None, llm_fn=llm_fn, mode="mcq",
        )
        cal_scores.attach_score_cache(score_fn, eval_file, "tag")
        score_fn(frame(["A", "B"]))
    assert seen == [[], [1, 2], []]


class HLEJudge:
    """Stand-in HLE judge: says yes when the response equals the correct answer, or always/never."""

    def __init__(self, verdict="match"):
        self.verdict = verdict
        self.calls = 0

    def working(self):
        return True

    def generate(self, prompt, **kwargs):
        self.calls += 1
        if self.verdict == "match":
            response = re.search(r"\[response\]: (.*)", prompt).group(1)
            answer = re.search(r"\[correct_answer\]: (.*)", prompt).group(1)
            correct = response.strip() == answer.strip()
        else:
            correct = self.verdict == "yes"
        return json.dumps({
            "extracted_final_answer": "", "reasoning": "stand-in",
            "correct": "yes" if correct else "no", "confidence": 100,
        })


@pytest.fixture
def hle(tmp_path, monkeypatch):
    monkeypatch.delenv("MMDR_JUDGE_PROMPT", raising=False)
    dataset = image_shortqa.ImageShortQADataset.__new__(image_shortqa.ImageShortQADataset)
    dataset.dataset_name = "hle"
    eval_file = str(tmp_path / "Replay.xlsx")

    def evaluate(predictions, judge, model="judge-a"):
        dump(pd.DataFrame({
            "index": [1, 2, 3, 4],
            "question": ["q1", "q2", "q3", "q4"],
            "answer": ["a1", "a2", "a3", "a4"],
            "prediction": predictions,
        }), eval_file)
        monkeypatch.setattr(image_shortqa, "build_judge", lambda **kwargs: judge)
        return float(dataset.evaluate(eval_file, model=model, nproc=1)["Overall"][0])

    return evaluate


def test_hle_rejudges_when_the_judge_changes(hle):
    predictions = ["a1", "a2", "a3", "a4"]
    assert hle(predictions, HLEJudge("yes"), model="judge-a") == 1.0
    assert hle(predictions, HLEJudge("no"), model="judge-b") == 0.0


def test_hle_rejudges_only_changed_predictions(hle):
    judge = HLEJudge()
    assert hle(["wrong", "wrong", "a3", "a4"], judge) == 0.5
    assert judge.calls == 4
    assert hle(["wrong", "wrong", "a3", "a4"], judge) == 0.5
    assert judge.calls == 4
    assert hle(["a1", "wrong", "a3", "a4"], judge) == 0.75
    assert judge.calls == 5


def test_hallusionbench_storage_follows_the_predictions(tmp_path):
    dataset = image_yorn.ImageYORNDataset.__new__(image_yorn.ImageYORNDataset)
    dataset.dataset_name = "HallusionBench"
    eval_file = str(tmp_path / "Replay.xlsx")

    def evaluate(predictions):
        n = len(predictions)
        dump(pd.DataFrame({
            "index": [f"VD_illusion_1_{i}_0_0" for i in range(n)],
            "category": ["VD"] * n,
            "l2-category": ["VD_illusion"] * n,
            "answer": ["Yes", "No"] * (n // 2),
            "prediction": predictions,
        }), eval_file)
        return float(dataset.evaluate(eval_file, model=None)["Overall"][0])

    assert evaluate(["Yes", "No"]) == 100.0
    assert evaluate(["No", "Yes", "No", "Yes"]) == 0.0

