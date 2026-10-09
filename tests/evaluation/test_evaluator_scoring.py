"""The counting, CountQA, image-MCQ and spatial-MCQ evaluators each score a 3-row
prediction file (two right, one wrong) to known metrics; integer extraction is
checked on a fixed table.

Evaluators are instantiated without loading their benchmark (no download); the
prediction file carries the columns the real TSVs provide.
"""
import pandas as pd
import pytest

from evaluation_kit.dataset import DATASET_CLASSES
from evaluation_kit.dataset.utils.fsc147 import extract_integer_from_prediction


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
