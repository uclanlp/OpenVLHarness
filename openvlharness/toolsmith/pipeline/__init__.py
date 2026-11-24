"""Grow a ToolSmith catalog from a baseline agent's failed rollouts.

Phase 1 mines capability gaps (mining), Phase 2 skips gaps the catalog already
covers (reuse), Phase 3 writes candidate tools (generation) and Phase 4 admits
those that pass both gates and are not near-duplicates (verification).
"""
import hashlib
import json
import logging
import os
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...bootstrap import build_capabilities, load_agent_config
from ...services.models import AuxiliaryModel
from ..catalog import load_catalog
from .generation import Generator, describe_tools
from .mining import Gap, judged_predictions, mine
from .reuse import closest
from .verification import review, run_once, scan

logger = logging.getLogger(__name__)

SETTINGS = {"encoder": "BAAI/bge-m3", "clusters": 500, "reuse_threshold": 0.40, "retries": 2,
            "timeout_seconds": 10, "critic_mean": 3.0, "critic_min": 1, "dedup_threshold": 0.55, "workers": 8}
# Model entry fields that only say how to reach the model, not which model answers or how.
CONNECTION = {"base", "key", "timeout", "retry"}


class Pipeline:
    """Generate, verify and admit tools for mined gaps into the agent configuration's catalog."""

    def __init__(self, config, work_dir):
        """Load the base tools, the starting catalog and the three pipeline models.

        Args:
            config: Pipeline settings with agent (agent-system YAML path), models
                (generator, critic and analyst model entries) and the paper
                hyperparameters of SETTINGS.
            work_dir: Directory for gaps.jsonl, mining.json and decisions.jsonl.

        Raises:
            ValueError: Settings are unknown or incomplete, or the agent configuration
                names no toolsmith.catalog to grow.
        """
        unknown = set(config) - set(SETTINGS) - {"pipeline_config_version", "agent", "models"}
        if unknown:
            raise ValueError(f"Unknown pipeline settings: {sorted(unknown)}")
        if set(config.get("models") or {}) != {"generator", "critic", "analyst"}:
            raise ValueError("models must define generator, critic and analyst")
        self.settings = {**SETTINGS, **{key: value for key, value in config.items() if key in SETTINGS}}
        agent = load_agent_config(config["agent"])
        if not (agent.get("toolsmith") or {}).get("catalog"):
            raise ValueError("Set toolsmith.catalog in the agent configuration to the catalog directory to grow")
        self.catalog = Path(agent["toolsmith"]["catalog"])
        base = {capability.name: capability.public_schema()["function"] for capability in build_capabilities(agent).values()}
        tools = load_catalog(self.catalog)
        self.descriptions = {**{name: schema["description"] for name, schema in base.items()},
                             **{name: tool.tool_description for name, tool in tools.items()}}
        self.models = {role: AuxiliaryModel(config["models"][role]) for role in ("generator", "critic", "analyst")}
        self.analyst = {key: value for key, value in config["models"]["analyst"].items() if key not in CONNECTION}
        self.generator = Generator(self.models["generator"], self.models["analyst"], describe_tools(base, tools))
        self.work_dir = Path(work_dir)
        self.lock = threading.Lock()

    def mining_inputs(self, rollouts):
        """Describe what mining reads: each rollout's judged predictions and the mining settings."""
        entries = []
        for directory in rollouts:
            predictions = judged_predictions(directory)
            entries.append({"directory": str(Path(directory).resolve()), "predictions": predictions.name,
                            "sha256": hashlib.sha256(predictions.read_bytes()).hexdigest()})
        return json.loads(json.dumps({"rollouts": sorted(entries, key=lambda entry: entry["directory"]),
                                      "encoder": self.settings["encoder"], "clusters": self.settings["clusters"],
                                      "analyst": self.analyst}))

    def gaps(self, rollouts, *, rebuild=False):
        """Return the cluster representatives, mining them once and reusing gaps.jsonl afterwards.

        mining.json records the rollouts and mining settings gaps.jsonl came from, and
        gaps.jsonl is reused only for the same ones; rebuild mines the rollouts again.

        Raises:
            ValueError: A rollout has no judged predictions, or gaps.jsonl exists for
                other rollouts or settings, or without a record of them, and rebuild is unset.
        """
        path, record = self.work_dir / "gaps.jsonl", self.work_dir / "mining.json"
        inputs = self.mining_inputs(rollouts)
        if path.exists() and not rebuild:
            recorded = json.loads(record.read_text()) if record.exists() else None
            if recorded != inputs:
                raise ValueError(f"{path} was mined {_difference(recorded, inputs)}. Pass --rebuild-gaps to "
                                 "mine these rollouts again, or use another --work-dir.")
            return [Gap(**json.loads(line)) for line in path.read_text().splitlines()]
        gaps = mine(rollouts, self.models["analyst"], encoder=self.settings["encoder"],
                    clusters=self.settings["clusters"], workers=self.settings["workers"])
        self.work_dir.mkdir(parents=True, exist_ok=True)
        # The record goes last, so an interrupted write never leaves gaps.jsonl with a matching record.
        record.unlink(missing_ok=True)
        path.write_text("".join(gap.to_json() + "\n" for gap in gaps))
        record.write_text(json.dumps(inputs, indent=2, ensure_ascii=False) + "\n")
        return gaps

    def run(self, rollouts, *, rebuild_gaps=False):
        """Process every gap not yet decided and return the count of each decision.

        A gap whose model or network calls fail stays undecided and counts as
        failed; rerunning the pipeline retries it. rebuild_gaps mines the rollouts
        again instead of reusing gaps.jsonl.

        Raises:
            ValueError: The gaps cannot be mined or reused (see gaps).
        """
        record = self.work_dir / "decisions.jsonl"
        decided = {json.loads(line)["source"] for line in record.read_text().splitlines()} if record.exists() else set()
        pending = [gap for gap in self.gaps(rollouts, rebuild=rebuild_gaps) if gap.source not in decided]
        with ThreadPoolExecutor(self.settings["workers"]) as pool:
            results = list(pool.map(self.attempt, pending))
        return Counter(result["decision"] for result in results)

    def attempt(self, gap):
        """Process one gap, leaving it undecided when a model or network call fails."""
        try:
            return self.process(gap)
        except Exception as exc:
            logger.warning("Gap %s stays undecided: %s: %s", gap.source, type(exc).__name__, exc)
            return {"decision": "failed"}

    def process(self, gap):
        """Reuse, or generate and verify with feedback-driven retries; record the decision."""
        with self.lock:
            match, similarity = closest(f"{gap.question} {gap.reason}", self.descriptions)
        if similarity > self.settings["reuse_threshold"]:
            return self.decide(gap, {"decision": "reused", "tool": match, "similarity": similarity})
        source, feedback = self.generator.write(gap), []
        for attempt in range(self.settings["retries"] + 1):
            outcome = self.verify(gap, source)
            if "decision" in outcome:
                return self.decide(gap, {**outcome, "attempts": feedback})
            feedback.append(outcome["feedback"])
            if attempt < self.settings["retries"]:
                source = self.generator.revise(gap, source or "", outcome["feedback"])
        return self.decide(gap, {"decision": "rejected", "attempts": feedback})

    def verify(self, gap, source):
        """Run both gates and the catalog update; return a decision or gate feedback for a retry."""
        if source is None:
            return {"feedback": "the reply held no Python source in the requested JSON"}
        issues = scan(source)
        metadata = None
        if not issues:
            metadata, issue = run_once(source, self.settings["timeout_seconds"])
            issues = [issue] if issue else []
        if metadata and not re.fullmatch(r"[a-z][a-z0-9_]*", metadata["tool_name"]):
            issues = [f"tool_name {metadata['tool_name']!r} must be snake_case"]
        elif metadata and (metadata["tool_name"] in self.descriptions
                           or os.path.lexists(self.catalog / f"{metadata['tool_name']}.py")):
            # A module file of that name may register a different public name; it is a collision too.
            issues = [f"a tool named {metadata['tool_name']} already exists; choose another name"]
        if issues:
            return {"feedback": "Gate 1: " + "; ".join(issues)}
        passed, critique, scores = review(self.models["critic"], gap.question, metadata, source,
                                          mean=self.settings["critic_mean"], minimum=self.settings["critic_min"])
        if not passed:
            return {"feedback": "Gate 2: " + critique}
        return self.admit(metadata, source, scores)

    def admit(self, metadata, source, scores):
        """Admit a verified candidate unless its description nearly duplicates an existing tool.

        The module file is created exclusively: an existing file of that name is a
        collision even when the module in it registers a different tool name.
        """
        name = metadata["tool_name"]
        path = self.catalog / f"{name}.py"
        with self.lock:
            match, similarity = closest(metadata["tool_description"], self.descriptions)
            if similarity >= self.settings["dedup_threshold"] or name in self.descriptions:
                return {"decision": "duplicate", "tool": name, "match": match, "similarity": similarity}
            try:
                stream = open(path, "x", encoding="utf-8")
            except FileExistsError:
                return {"decision": "duplicate", "tool": name, "file": path.name}
            try:
                with stream:
                    stream.write(source)
            except BaseException:
                path.unlink(missing_ok=True)
                raise
            self.descriptions[name] = metadata["tool_description"]
        return {"decision": "admitted", "tool": name, "scores": scores}

    def decide(self, gap, result):
        """Append one decision to decisions.jsonl and return it."""
        line = json.dumps({"source": gap.source, "question": gap.question, **result}, ensure_ascii=False)
        with self.lock, open(self.work_dir / "decisions.jsonl", "a") as stream:
            stream.write(line + "\n")
        return result


def _difference(recorded, current):
    """Say how the recorded mining inputs differ from the current ones, for the cache error."""
    if recorded is None:
        return "without a record of its rollouts and settings (no mining.json)"
    changes = []
    before = {entry["directory"]: entry for entry in recorded.get("rollouts", [])}
    after = {entry["directory"]: entry for entry in current["rollouts"]}
    changes += [f"rollout {name} is new" for name in sorted(set(after) - set(before))]
    changes += [f"rollout {name} is no longer given" for name in sorted(set(before) - set(after))]
    changes += [f"the judged predictions of {name} changed" for name in sorted(set(before) & set(after))
                if before[name] != after[name]]
    changes += [f"{key} was {recorded.get(key)!r}, now {current[key]!r}" for key in ("encoder", "clusters", "analyst")
                if recorded.get(key) != current[key]]
    return "from other inputs: " + "; ".join(changes or ["the record differs"])
