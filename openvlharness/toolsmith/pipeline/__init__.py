"""Grow a ToolSmith catalog from a baseline agent's failed rollouts.

Phase 1 mines capability gaps (mining), Phase 2 skips gaps the catalog already
covers (reuse), Phase 3 writes candidate tools (generation) and Phase 4 admits
those that pass both gates and are not near-duplicates (verification).
"""
import hashlib
import json
import logging
import math
import os
import re
import sys
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...bootstrap import build_capabilities, load_agent_config
from ...services.models import AuxiliaryModel
from ...services.tools.base import RUNTIME_FIELDS
from ..worker import catalog_tools, memory_limit_enforced
from .generation import Generator, describe_tools
from .mining import Gap, evidence_digest, judged_predictions, mine
from .reuse import closest
from .verification import declared_name, review, run_once, scan

logger = logging.getLogger(__name__)

SETTINGS = {"encoder": "BAAI/bge-m3", "clusters": 500, "reuse_threshold": 0.40, "retries": 2,
            "timeout_seconds": 10, "critic_mean": 3.0, "critic_min": 1, "dedup_threshold": 0.55, "workers": 8,
            "isolation": True}
# Model entry fields that only say how to reach the model, not which model answers or how.
CONNECTION = {"base", "key", "timeout", "retry"}
DECISION_SETTINGS = ("reuse_threshold", "retries", "timeout_seconds", "critic_mean", "critic_min", "dedup_threshold",
                     "isolation")


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
            ValueError: Settings are unknown, incomplete or of the wrong type, the agent
                configuration names no toolsmith.catalog to grow, or isolation is on and
                this host cannot confine Gate 1 runs.
        """
        unknown = set(config) - set(SETTINGS) - {"pipeline_config_version", "agent", "models"}
        if unknown:
            raise ValueError(f"Unknown pipeline settings: {sorted(unknown)}")
        if set(config.get("models") or {}) != {"generator", "critic", "analyst"}:
            raise ValueError("models must define generator, critic and analyst")
        self.settings = {**SETTINGS, **{key: value for key, value in config.items() if key in SETTINGS}}
        _check_settings(self.settings)
        if self.settings["isolation"]:
            from ..isolation import available

            confinable, reason = available()
            if not confinable:
                raise ValueError(f"Gate 1 runs candidates confined by Linux user namespaces, which this host does not "
                                 f"allow ({reason}). Set isolation: false to run them without it.")
        else:
            logger.warning("isolation is off: Gate 1 runs candidates with the pipeline's file and network access.")
        if not memory_limit_enforced():
            logger.warning("%s does not enforce address-space limits: Gate 1 runs candidates without its 2 GiB cap.",
                           sys.platform)
        agent = load_agent_config(config["agent"])
        if not (agent.get("toolsmith") or {}).get("catalog"):
            raise ValueError("Set toolsmith.catalog in the agent configuration to the catalog directory to grow")
        self.catalog = Path(agent["toolsmith"]["catalog"])
        base = {capability.name: capability.public_schema()["function"] for capability in build_capabilities(agent).values()}
        # Read in a worker, so catalog code does not run in the pipeline's own process.
        tools = catalog_tools(self.catalog, settings={**(agent["toolsmith"].get("worker") or {}),
                                                      "isolation": self.settings["isolation"]})
        self.descriptions = {**{name: schema["description"] for name, schema in base.items()},
                             **{name: tool.tool_description for name, tool in tools.items()}}
        self.models = {role: AuxiliaryModel(config["models"][role]) for role in ("generator", "critic", "analyst")}
        self.analyst = {key: value for key, value in config["models"]["analyst"].items() if key not in CONNECTION}
        # What a decision depends on besides its gap: how the three models answer, and the gates' settings.
        self.decision_settings = hashlib.sha256(json.dumps({
            "models": {role: {key: value for key, value in config["models"][role].items() if key not in CONNECTION}
                       for role in ("generator", "critic", "analyst")},
            **{key: self.settings[key] for key in DECISION_SETTINGS}}, sort_keys=True).encode()).hexdigest()
        self.generator = Generator(self.models["generator"], self.models["analyst"], describe_tools(base, tools))
        self.work_dir = Path(work_dir)
        self.lock = threading.Lock()

    def mining_inputs(self, rollouts):
        """Describe what mining reads: each rollout's judged predictions and failed trajectories, and the settings."""
        entries = []
        for directory in rollouts:
            predictions = judged_predictions(directory)
            entries.append({"directory": str(Path(directory).resolve()), "predictions": predictions.name,
                            "sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                            "trajectories_sha256": evidence_digest(directory)})
        return json.loads(json.dumps({"rollouts": sorted(entries, key=lambda entry: entry["directory"]),
                                      "encoder": self.settings["encoder"], "clusters": self.settings["clusters"],
                                      "analyst": self.analyst}))

    def gaps(self, rollouts, *, rebuild=False):
        """Return the cluster representatives, mining them once and reusing gaps.jsonl afterwards.

        mining.json records the rollouts and mining settings gaps.jsonl came from, and
        gaps.jsonl is reused only for the same ones; rebuild mines the rollouts again.
        Self-critiques are kept in critiques.json: when some failed, mining.json lists
        their queries, and the next run mines again, asking the analyst only about those.

        Raises:
            ValueError: A rollout has no judged predictions, or gaps.jsonl exists for
                other rollouts or settings, or without a record of them, and rebuild is unset.
        """
        path, record, cache = (self.work_dir / name for name in ("gaps.jsonl", "mining.json", "critiques.json"))
        inputs = self.mining_inputs(rollouts)
        known = {}
        if path.exists() and not rebuild:
            recorded = json.loads(record.read_text()) if record.exists() else None
            retry = (recorded or {}).pop("failed_critiques", [])
            if recorded != inputs:
                raise ValueError(f"{path} was mined {_difference(recorded, inputs)}. Pass --rebuild-gaps to "
                                 "mine these rollouts again, or use another --work-dir.")
            if not retry:
                return [Gap(**json.loads(line)) for line in path.read_text().splitlines()]
            stored = json.loads(cache.read_text()) if cache.exists() else {}
            known = stored.get("critiques", {}) if stored.get("analyst") == self.analyst else {}
            logger.info("Retrying the %d self-critiques that failed when %s was mined", len(retry), path)
        gaps, critiques, failed = mine(rollouts, self.models["analyst"], encoder=self.settings["encoder"],
                                       clusters=self.settings["clusters"], workers=self.settings["workers"],
                                       known=known)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"analyst": self.analyst, "critiques": critiques}, ensure_ascii=False) + "\n")
        # The record goes last, so an interrupted write never leaves gaps.jsonl with a matching record.
        record.unlink(missing_ok=True)
        path.write_text("".join(gap.to_json() + "\n" for gap in gaps))
        record.write_text(json.dumps({**inputs, "failed_critiques": sorted(failed)}, indent=2, ensure_ascii=False)
                          + "\n")
        if failed:
            logger.warning("%d self-critiques failed and were left out (%s); run again to retry them", len(failed),
                           ", ".join(sorted(failed)[:10]) + (", ..." if len(failed) > 10 else ""))
        return gaps

    def run(self, rollouts, *, rebuild_gaps=False):
        """Process every gap not yet decided and return the count of each decision.

        A gap whose model or network calls fail stays undecided and counts as
        failed; rerunning the pipeline retries it. rebuild_gaps mines the rollouts
        again instead of reusing gaps.jsonl. A decision is reused only for a gap
        with the same inputs and the same models and gate settings (the digests
        decisions record), and only while the tool it admitted, reused or matched
        is still in the catalog; any other gap is decided again.

        Raises:
            ValueError: The gaps cannot be mined or reused (see gaps).
        """
        record = self.work_dir / "decisions.jsonl"
        decided = set()
        if record.exists():
            entries = map(json.loads, record.read_text().splitlines())
            decided = {(entry["source"], entry.get("gap_sha256")) for entry in entries
                       if entry.get("settings_sha256") == self.decision_settings and self.holds(entry)}
        pending = [gap for gap in self.gaps(rollouts, rebuild=rebuild_gaps)
                   if (gap.source, _gap_digest(gap)) not in decided]
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
            _, issue = run_once(source, self.settings["timeout_seconds"], isolation=self.settings["isolation"])
            issues = [issue] if issue else []
        if not issues:
            # What Gate 1's run prints, the candidate can print too: take the metadata from a load instead.
            metadata, issue = self.loaded(source)
            issues = [issue] if issue else []
        reserved = sorted(RUNTIME_FIELDS & set((metadata or {}).get("input_schema", {}).get("properties", {})))
        if reserved:
            # The router refuses these as task arguments, so such a tool could never be called.
            issues = [f"input_schema declares {', '.join(reserved)}, which the router reserves; use other names"]
        elif metadata and declared_name(source) != metadata["tool_name"]:
            issues = [f"the tool registers {metadata['tool_name']!r}; pass its tool_name as one string literal"]
        elif metadata and not re.fullmatch(r"[a-z][a-z0-9_]*", metadata["tool_name"]):
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

    def loaded(self, source):
        """Load a candidate as the router will, in a worker (confined unless isolation is off); return its metadata.

        Returns:
            (metadata, None), or (None, issue) when the candidate does not load as one tool.
        """
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "candidate.py").write_text(source)
            try:
                tools = catalog_tools(directory, legacy=False, settings={
                    "isolation": self.settings["isolation"], "network": False, "write_files": False, "memory_gib": 2})
            except (ValueError, RuntimeError) as exc:
                return None, f"loading the tool failed: {exc}"
        if len(tools) != 1:
            return None, "the module registers no tool the agent can offer"
        tool = next(iter(tools.values()))
        return {"tool_name": tool.tool_name, "tool_description": tool.tool_description,
                "input_schema": tool.input_schema}, None

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

    def holds(self, entry):
        """Whether a recorded decision still describes the catalog: the tool it admitted, reused or matched exists."""
        if entry.get("decision") in {"admitted", "reused"}:
            return entry.get("tool") in self.descriptions
        if entry.get("decision") == "duplicate":
            if "file" in entry:
                return (self.catalog / entry["file"]).exists()
            return entry.get("match") in self.descriptions
        return True

    def decide(self, gap, result):
        """Append one decision, with the digests of its gap and settings, to decisions.jsonl and return it."""
        line = json.dumps({"source": gap.source, "question": gap.question, "gap_sha256": _gap_digest(gap),
                           "settings_sha256": self.decision_settings, **result}, ensure_ascii=False)
        with self.lock, open(self.work_dir / "decisions.jsonl", "a") as stream:
            stream.write(line + "\n")
        return result


def _check_settings(settings):
    """Raise ValueError unless each setting has its type and range (a null timeout would never expire)."""
    if not (isinstance(settings["encoder"], str) and settings["encoder"].strip()):
        raise ValueError("encoder must name a sentence-transformers model")
    for key, least in (("clusters", 1), ("workers", 1), ("retries", 0)):
        if type(settings[key]) is not int or settings[key] < least:
            raise ValueError(f"{key} must be an integer of at least {least}")
    for key in ("timeout_seconds", "reuse_threshold", "dedup_threshold", "critic_mean", "critic_min"):
        if type(settings[key]) not in (int, float) or not math.isfinite(settings[key]):
            raise ValueError(f"{key} must be a number")
    if settings["timeout_seconds"] <= 0:
        raise ValueError("timeout_seconds must be a positive number")
    if type(settings["isolation"]) is not bool:
        raise ValueError("isolation must be true or false")


def _gap_digest(gap):
    """SHA-256 of everything a decision depends on: the gap's query, answers, tool activity and critique."""
    return hashlib.sha256(gap.to_json().encode()).hexdigest()


def _difference(recorded, current):
    """Say how the recorded mining inputs differ from the current ones, for the cache error."""
    if recorded is None:
        return "without a record of its rollouts and settings (no mining.json)"
    recorded = {key: value for key, value in recorded.items() if key != "failed_critiques"}
    changes = []
    before = {entry["directory"]: entry for entry in recorded.get("rollouts", [])}
    after = {entry["directory"]: entry for entry in current["rollouts"]}
    changes += [f"rollout {name} is new" for name in sorted(set(after) - set(before))]
    changes += [f"rollout {name} is no longer given" for name in sorted(set(before) - set(after))]
    for name in sorted(set(before) & set(after)):
        if (before[name].get("predictions"), before[name].get("sha256")) != (after[name]["predictions"],
                                                                             after[name]["sha256"]):
            changes.append(f"the judged predictions of {name} changed")
        elif before[name] != after[name]:
            changes.append(f"the trajectories of failed queries in {name} changed")
    changes += [f"{key} was {recorded.get(key)!r}, now {current[key]!r}" for key in ("encoder", "clusters", "analyst")
                if recorded.get(key) != current[key]]
    return "from other inputs: " + "; ".join(changes or ["the record differs"])
