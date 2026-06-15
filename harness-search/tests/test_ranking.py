#!/usr/bin/env python3
"""Fixture test for harness-search state ranking."""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_STORE = REPO_ROOT / "harness-search" / "scripts" / "state_store.py"


def load_module():
    spec = importlib.util.spec_from_file_location("harness_search_state_store", STATE_STORE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_module()
    states = [
        {"state_id": "s0003", "status": "evaluated", "score": 0.5, "creation_index": 3},
        {"state_id": "s0001", "status": "evaluated", "score": 0.6, "creation_index": 1},
        {"state_id": "s0002", "status": "evaluated", "score": 0.6, "creation_index": 2},
        {"state_id": "s0004", "status": "failed", "score": 1.0, "creation_index": 4},
    ]
    ranked = module.rank_states(states, 3)
    assert [state["state_id"] for state in ranked] == ["s0001", "s0002", "s0003"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
