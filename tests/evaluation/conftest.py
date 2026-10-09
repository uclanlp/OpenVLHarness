"""Scoped deterministic scheduling for retained evaluator regression tests."""
import pytest


@pytest.fixture
def synchronous_evaluation(monkeypatch):
    from evaluation_kit import runner

    def run(func, tasks, save=None, keys=None, **kwargs):
        results = [func(**task) for task in tasks]
        if save is not None:
            runner.dump(dict(zip(keys, results)), save)
        return results

    monkeypatch.setattr(runner, "track_progress_rich", run)
