#!/usr/bin/env python3
"""Small fixture test for full-file overlay states."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_STORE = REPO_ROOT / "harness-search" / "scripts" / "state_store.py"


def load_module():
    spec = importlib.util.spec_from_file_location("harness_search_state_store", STATE_STORE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> int:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = root / "base"
        states = root / "states"
        allowed = "mini_vlm_toolkit/agent/constant.py"
        write(base / allowed, "BASE = 'old'\n")

        parent = module.create_initial_state(states)
        proposal = root / "proposal" / "p001"
        write(proposal / "files" / allowed, "BASE = 'new'\n")
        write_json(proposal / "metadata.json", {
            "proposal_id": "p001",
            "summary": "change prompt text",
        })

        child = module.create_child_state_from_proposal(
            base_repo=base,
            states_root=states,
            parent_state_dir=parent,
            proposal_dir=proposal,
            state_id="s0001",
            created_round=1,
            creation_index=1,
        )
        scratch = root / "scratch"
        module.materialize_state(base, child, scratch)
        assert (scratch / allowed).read_text(encoding="utf-8") == "BASE = 'new'\n"
        diff = (child / "patch.diff").read_text(encoding="utf-8")
        assert "BASE = 'old'" in diff
        assert "BASE = 'new'" in diff

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = root / "base"
        states = root / "states"
        allowed = "mini_vlm_toolkit/agent/constant.py"
        write(base / allowed, "A\nB\nC\nD\n")

        parent = module.create_initial_state(states)
        p001 = root / "proposal" / "p001"
        p002 = root / "proposal" / "p002"
        p003 = root / "proposal" / "p003"
        write(p001 / "files" / allowed, "A\nB1\nC\nD\n")
        write(p002 / "files" / allowed, "A\nB\nC2\nD\n")
        write(p003 / "files" / allowed, "A\nB3\nC\nD\n")
        write_json(p001 / "metadata.json", {"proposal_id": "p001", "summary": "change B"})
        write_json(p002 / "metadata.json", {"proposal_id": "p002", "summary": "change C"})
        write_json(p003 / "metadata.json", {"proposal_id": "p003", "summary": "also change B"})

        child = module.create_child_state_from_proposals(
            base_repo=base,
            states_root=states,
            parent_state_dir=parent,
            proposal_dirs=[p001, p002],
            state_id="s0001",
            created_round=1,
            creation_index=1,
        )
        scratch = root / "scratch"
        module.materialize_state(base, child, scratch)
        assert (scratch / allowed).read_text(encoding="utf-8") == "A\nB1\nC2\nD\n"
        state = json.loads((child / "state.json").read_text(encoding="utf-8"))
        assert state["proposal_id"] == "p001+p002"
        assert state["source_proposal_ids"] == ["p001", "p002"]
        assert state["combo_size"] == 2
        assert state["is_crossover"] is True

        try:
            module.create_child_state_from_proposals(
                base_repo=base,
                states_root=states,
                parent_state_dir=parent,
                proposal_dirs=[p001, p003],
                state_id="s0002",
                created_round=1,
                creation_index=2,
            )
        except module.ProposalConflict as exc:
            assert "p001 and p003" in str(exc)
        else:
            raise AssertionError("expected ProposalConflict for overlapping edits")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
