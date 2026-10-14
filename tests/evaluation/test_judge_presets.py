"""configs/judge.yaml takes the judge from the environment and reaches the client unchanged."""
from pathlib import Path

import pytest
import yaml

from evaluation_kit.dataset.utils import judge_util
from evaluation_kit.utils.judge_utils import load_judge_config

JUDGE_YAML = Path(__file__).resolve().parents[2] / "configs" / "judge.yaml"
JUDGE_ENV = {"JUDGE_MODEL": "my-judge", "JUDGE_API_BASE": "http://judge.invalid/v1/chat/completions",
             "JUDGE_API_KEY": "k"}


def test_judge_configuration_takes_the_judge_from_the_environment(monkeypatch):
    for name, value in JUDGE_ENV.items():
        monkeypatch.setenv(name, value)
    assert yaml.safe_load(JUDGE_YAML.read_text())["judge_config_version"]
    kwargs = load_judge_config(str(JUDGE_YAML))
    assert kwargs["model"] == "my-judge"
    assert kwargs["api_base"] == JUDGE_ENV["JUDGE_API_BASE"]
    assert kwargs["key"] == "k"
    assert "judge_prompt" not in kwargs


def test_unset_judge_variables_are_named(monkeypatch):
    for name in JUDGE_ENV:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="judge configuration .*judge.yaml: JUDGE_MODEL \\(model_name\\)"):
        load_judge_config(str(JUDGE_YAML))


def test_build_judge_passes_the_configured_model_through(monkeypatch):
    for name, value in JUDGE_ENV.items():
        monkeypatch.setenv(name, value)
    captured = {}

    class Client:
        def __init__(self, model, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs

    monkeypatch.setattr("evaluation_kit.dataset.api.OpenAIWrapper", Client)
    judge_util.build_judge(**load_judge_config(str(JUDGE_YAML)))
    assert captured["model"] == "my-judge"
    assert captured["kwargs"] == {"api_base": JUDGE_ENV["JUDGE_API_BASE"], "key": "k", "retry": 3,
                                  "max_tokens": judge_util.JUDGE_MAX_TOKENS_DEFAULT}
