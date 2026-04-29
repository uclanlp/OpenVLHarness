import contextlib
import os
from pathlib import Path
import sys
import tempfile
import types

import pandas as pd

repo_root = Path(__file__).resolve().parents[3]
pkg_root = repo_root / "mini_vlm_toolkit"

# Avoid importing mini_vlm_toolkit/__init__.py, which pulls eager dataset deps.
mini_pkg = types.ModuleType("mini_vlm_toolkit")
mini_pkg.__path__ = [str(pkg_root)]
sys.modules.setdefault("mini_vlm_toolkit", mini_pkg)

utils_pkg = types.ModuleType("mini_vlm_toolkit.utils")
utils_pkg.__path__ = [str(pkg_root / "utils")]


def _track_progress_rich(func, tasks, save=None, keys=None, **_kwargs):
    results = {}
    for key, task in zip(keys, tasks):
        results[key] = func(**task)
    if save is not None:
        from mini_vlm_toolkit.smp import dump

        dump(results, save)
    return [results[key] for key in keys]


utils_pkg.track_progress_rich = _track_progress_rich
sys.modules.setdefault("mini_vlm_toolkit.utils", utils_pkg)

reporting_stub = types.ModuleType("mini_vlm_toolkit.utils.reporting_utils")
reporting_stub.build_trajectory_records = lambda *args, **kwargs: []
reporting_stub.write_input_snapshot_json = lambda *args, **kwargs: None
reporting_stub.write_html_reports = lambda *args, **kwargs: None
reporting_stub.write_full_trajectory_jsons = lambda *args, **kwargs: None
reporting_stub.write_raw_response_jsons = lambda *args, **kwargs: None
reporting_stub.write_raw_trajectory_jsons = lambda *args, **kwargs: None
sys.modules.setdefault("mini_vlm_toolkit.utils.reporting_utils", reporting_stub)

from mini_vlm_toolkit import inference_image


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
    result_file = inference_image.get_pred_file_path(
        str(dataset_dir),
        "Model",
        "DummySet",
        use_env_format=True,
    )
    inference_image.dump(
        pd.DataFrame(
            {
                "index": [1, 2],
                "prediction": [
                    "ok",
                    inference_image.FAIL_MSG,
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

        inference_image.infer_image_job(
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

        inference_image.infer_image_job(
            _Model(),
            str(work_dir),
            "Model",
            _Dataset(),
            ignore_failed=True,
        )

        refreshed = inference_image.load(str(result_file))
        predictions = {
            idx: pred
            for idx, pred in zip(refreshed["index"], refreshed["prediction"])
        }
        assert predictions == {1: "ok", 2: "retried-2"}


if __name__ == "__main__":
    test_final_result_file_is_skipped_by_default()
    test_ignore_failed_retries_failed_rows_from_final_result_file()
    print("Inference image retry-failed tests passed.")
