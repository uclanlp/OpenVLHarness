import importlib.util
import pytest

if importlib.util.find_spec("mini_vlm_toolkit") is None:
    pytest.skip("Retained evaluator is available only from the research checkout", allow_module_level=True)

for dependency in ("pandas", "torch", "validators", "openpyxl", "xlsxwriter"):
    pytest.importorskip(dependency, reason="Retained evaluator dependencies are optional")

import json
import os
import tempfile

from mini_vlm_toolkit.utils.reporting_utils import (
    build_complete_messages_from_trajectory,
    write_full_trajectory_jsons,
)


def test_build_complete_messages_appends_missing_final_assistant():
    trajectory = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
        ],
        "chat_history": [
            {
                "turn": 1,
                "assistant": {"content": "final answer text", "tool_calls": []},
                "tool_responses": [],
                "final_answer": "final answer text",
            }
        ],
    }

    complete = build_complete_messages_from_trajectory(trajectory)
    assert isinstance(complete, list)
    assert len(complete) == 3
    assert complete[-1]["role"] == "assistant"
    assert complete[-1]["content"] == "final answer text"


def test_build_complete_messages_does_not_duplicate_existing_final_assistant():
    trajectory = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "already there"},
        ],
        "chat_history": [
            {
                "turn": 1,
                "assistant": {"content": "already there", "tool_calls": []},
                "tool_responses": [],
                "final_answer": "already there",
            }
        ],
    }

    complete = build_complete_messages_from_trajectory(trajectory)
    assert isinstance(complete, list)
    assert len(complete) == 2
    assert complete[-1]["content"] == "already there"


def test_write_full_trajectory_jsons_matches_html_selection_limit():
    records = [
        {
            "task_id": "short",
            "dataset": "D",
            "model_name": "M",
            "agent_class": "A",
            "prediction": "p1",
            "trajectory": {
                "messages": [{"role": "user", "content": "q1"}],
                "chat_history": [{"turn": 1, "assistant": {"content": "a1"}, "final_answer": "a1"}],
                "final_answer": "a1",
            },
        },
        {
            "task_id": "long",
            "dataset": "D",
            "model_name": "M",
            "agent_class": "A",
            "prediction": "p2",
            "trajectory": {
                "messages": [{"role": "user", "content": "q2"}],
                "chat_history": [
                    {"turn": 1, "assistant": {"content": "step1"}},
                    {"turn": 2, "assistant": {"content": "step2"}},
                ],
                "final_answer": "step2",
            },
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        write_full_trajectory_jsons(records, tmpdir, limit=1)
        files = sorted(os.listdir(tmpdir))
        assert files == ["long.json"]
        with open(os.path.join(tmpdir, "long.json"), "r", encoding="utf8") as f:
            payload = json.load(f)
        assert payload["task_id"] == "long"
        assert "trajectory" not in payload
        assert isinstance(payload.get("messages"), list)
        assert payload["messages"][-1]["role"] == "assistant"
        assert payload["messages"][-1]["content"] == "step2"
