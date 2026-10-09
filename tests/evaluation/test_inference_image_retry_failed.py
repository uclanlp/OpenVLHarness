import importlib.util
import pytest

if importlib.util.find_spec("evaluation_kit") is None:
    pytest.skip("Retained evaluator is available only from the research checkout", allow_module_level=True)

for dependency in ("pandas", "validators", "openpyxl", "xlsxwriter"):
    pytest.importorskip(dependency, reason="Retained evaluator dependencies are optional")

import contextlib
import os
from pathlib import Path
import tempfile

import pandas as pd

from evaluation_kit import runner
from evaluation_kit.utils import file_db

pytestmark = pytest.mark.usefixtures("synchronous_evaluation")


@contextlib.contextmanager
def _patch_env(updates):
    original = {
        key: os.environ.get(key)
        for key in updates
    }
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _Dataset:
    dataset_name = "DummySet"
    data = pd.DataFrame({"index": [1, 2]})

    def __len__(self):
        return len(self.data)

    def build_prompt(self, item):
        return [{"type": "text", "value": str(item["index"])}]

    def dump_image(self, item):
        return []


class _Model:
    def generate(self, message, dataset=None, task_id=None):
        del message, dataset
        return f"retried-{task_id}"


def _write_final_result_file(work_dir):
    dataset_dir = work_dir / "DummySet"
    dataset_dir.mkdir()
    result_file = runner.get_pred_file_path(
        str(dataset_dir),
        "Model",
        "DummySet",
        use_env_format=True,
    )
    runner.dump(
        pd.DataFrame(
            {
                "index": [1, 2],
                "prediction": [
                    "ok",
                    runner.FAIL_MSG,
                ],
            }
        ),
        result_file,
    )
    return dataset_dir, Path(result_file)


def test_final_result_file_is_skipped_by_default():
    with _patch_env({"PRED_FORMAT": "xlsx"}), tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        dataset_dir, _result_file = _write_final_result_file(work_dir)

        runner.infer_image_job(
            _Model(),
            str(work_dir),
            "Model",
            _Dataset(),
            ignore_failed=False,
        )

        assert not (dataset_dir / "Model_PREV.pkl").exists()


def test_ignore_failed_retries_failed_rows_from_final_result_file():
    with _patch_env({"PRED_FORMAT": "xlsx"}), tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        _dataset_dir, result_file = _write_final_result_file(work_dir)

        runner.infer_image_job(
            _Model(),
            str(work_dir),
            "Model",
            _Dataset(),
            ignore_failed=True,
        )

        refreshed = runner.load(str(result_file))
        predictions = {
            idx: pred
            for idx, pred in zip(refreshed["index"], refreshed["prediction"])
        }
        assert predictions == {1: "ok", 2: "retried-2"}
