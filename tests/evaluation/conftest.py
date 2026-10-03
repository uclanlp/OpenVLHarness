"""Scoped deterministic scheduling for retained evaluator regression tests."""
import pytest


@pytest.fixture
def synchronous_evaluation(monkeypatch):
    from mini_vlm_toolkit import inference_image

    def run(func, tasks, save=None, keys=None, **kwargs):
        results = [func(**task) for task in tasks]
        if save is not None:
            inference_image.dump(dict(zip(keys, results)), save)
        return results

    monkeypatch.setattr(inference_image, "track_progress_rich", run)
