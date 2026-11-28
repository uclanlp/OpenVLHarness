"""Grow a ToolSmith catalog from a baseline agent's failed rollouts.

Phase 1 mines capability gaps (mining), Phase 2 skips gaps the catalog already
covers (reuse), Phase 3 writes candidate tools (generation) and Phase 4 admits
those that pass both gates and are not near-duplicates (verification).
"""
import errno
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
            "isolation": True, "max_tokens": 12800}
# Model entry fields that only say how to reach the model, not which model answers or how.
CONNECTION = {"base", "key", "timeout", "retry"}
DECISION_SETTINGS = ("reuse_threshold", "retries", "timeout_seconds", "critic_mean", "critic_min", "dedup_threshold",
                     "isolation")
# A gap or a self-critique that failed in this many runs with the same settings is no longer retried
# (--retry-failed retries it, and so does a change to any setting its failures depend on).
FAILED_RUNS = 3
# What mining.json records about a mining besides its inputs.
MINING_RESULTS = ("failed_critiques", "abandoned_critiques", "labels", "critique_failure_settings_sha256")


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
        agent = load_agent_config(config["agent"], stand_in=_unused)
        if not (agent.get("toolsmith") or {}).get("catalog"):
            raise ValueError("Set toolsmith.catalog in the agent configuration to the catalog directory to grow")
        self.catalog = Path(agent["toolsmith"]["catalog"])
        base = {capability.name: capability.public_schema()["function"]
                for capability in build_capabilities(agent).values()}
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
        # What a failure depends on besides its gap: every setting of the models that may fail, how to
        # reach them included (another endpoint or timeout can end the failures), and max_tokens. Credentials
        # stay out, the key and any in the endpoint URL: a digest over them would let a work directory
        # confirm a guessed one offline, so a new credential alone retries nothing; --retry-failed does.
        models = {role: _without_credentials(config["models"][role]) for role in ("generator", "critic", "analyst")}
        self.failure_settings = _digest({"models": models, "decision": self.decision_settings,
                                         "max_tokens": self.settings["max_tokens"]})
        self.critique_failure_settings = _digest({"analyst": models["analyst"],
                                                  "max_tokens": self.settings["max_tokens"]})
        self.generator = Generator(self.models["generator"], self.models["analyst"], describe_tools(base, tools),
                                   max_tokens=self.settings["max_tokens"])
        self.work_dir = Path(work_dir)
        self.lock = threading.Lock()
        self.critique_failures = 0  # self-critiques that failed in this run's mining
        self.abandoned = []  # gaps that failed in FAILED_RUNS runs and that this run left alone

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

    def gaps(self, rollouts, *, rebuild=False, retry_failed=False):
        """Return the cluster representatives, mining them once and reusing gaps.jsonl afterwards.

        mining.json records the rollouts and mining settings gaps.jsonl came from, and
        gaps.jsonl is reused only for the same ones; rebuild mines the rollouts again.
        Each self-critique is appended to critiques.jsonl as it completes, and every
        mining, a rebuild too, reuses the saved critiques of the same analyst: a mining
        that stopped part way, or failed afterwards while clustering, is not paid for
        twice. When some critiques failed, mining.json lists their queries, and the next
        run mines again, so the analyst is asked only about those; mining again clusters
        every gap again, so a few representatives can change. A query whose critique
        failed in FAILED_RUNS minings with the same analyst settings (its endpoint and
        timeout included, not its key) and max_tokens is left out and no longer retried,
        unless retry_failed is set.

        Raises:
            ValueError: A rollout has no judged predictions, or gaps.jsonl exists for
                other rollouts or settings, or without a record of them, and rebuild is unset.
        """
        path, record, log = (self.work_dir / name for name in ("gaps.jsonl", "mining.json", "critiques.jsonl"))
        inputs = self.mining_inputs(rollouts)
        if path.exists() and not rebuild:
            recorded = json.loads(record.read_text(encoding="utf-8")) if record.exists() else None
            results = {key: (recorded or {}).pop(key, None) for key in MINING_RESULTS}
            if recorded != inputs:
                raise ValueError(f"{path} was mined {_difference(recorded, inputs)}. Pass --rebuild-gaps to "
                                 "mine these rollouts again, or use another --work-dir.")
            # Critiques set aside are asked again with --retry-failed, or once the settings they failed with change.
            again = retry_failed or results["critique_failure_settings_sha256"] != self.critique_failure_settings
            retry = (results["failed_critiques"] or []) + ((results["abandoned_critiques"] or []) if again else [])
            if not retry:
                return [Gap(**json.loads(line)) for line in json_lines(path)]
            logger.info("Retrying the %d self-critiques that failed when %s was mined", len(retry), path)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        saved = log.read_bytes() if log.exists() else b""
        known, failed_before = _saved_critiques(saved, inputs["analyst"], self.critique_failure_settings)
        if known:
            logger.info("Reusing %d self-critiques saved in %s", len(known), log)
        lock = threading.Lock()
        with log.open("ab") as stream:
            if saved and not saved.endswith(b"\n"):
                stream.write(b"\n")  # end a line cut short when a run stopped, so the next line parses

            def save(key, label, reason):
                entry = {"analyst": inputs["analyst"], "key": key,
                         **({"label": label, "reason": reason} if label is not None else
                            {"failed": reason, "failure_settings_sha256": self.critique_failure_settings})}
                line = json.dumps(entry, ensure_ascii=False) + "\n"
                with lock:
                    stream.write(line.encode())
                    stream.flush()
            mining = mine(rollouts, self.models["analyst"], encoder=self.settings["encoder"],
                          clusters=self.settings["clusters"], workers=self.settings["workers"], known=known,
                          save=save, max_tokens=self.settings["max_tokens"], failed_before=failed_before,
                          give_up_after=None if retry_failed else FAILED_RUNS)
        self.critique_failures = mining.failures
        # The record goes last, so an interrupted write never leaves gaps.jsonl with a matching record.
        record.unlink(missing_ok=True)
        path.write_text("".join(gap.to_json() + "\n" for gap in mining.representatives), encoding="utf-8")
        record.write_text(json.dumps({**inputs, "failed_critiques": sorted(mining.failed),
                                      "abandoned_critiques": sorted(mining.abandoned), "labels": mining.labels,
                                      "critique_failure_settings_sha256": self.critique_failure_settings},
                                     indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if mining.failed:
            logger.warning("%d self-critiques failed and were left out (%s); run again to retry them",
                           len(mining.failed), ", ".join(sorted(mining.failed)[:10])
                           + (", ..." if len(mining.failed) > 10 else ""))
        if mining.abandoned:
            logger.warning("%d self-critiques failed in %d minings with these settings and are set aside; pass "
                           "--retry-failed to ask again, also after a new API key (a change to the analyst's other "
                           "settings or max_tokens retries them too)",
                           len(mining.abandoned), FAILED_RUNS)
        return mining.representatives

    def run(self, rollouts, *, rebuild_gaps=False, retry_failed=False):
        """Process every gap not yet decided and return the count of each decision.

        A gap whose model or network calls fail stays undecided, counts as failed, and is
        recorded with its error; rerunning the pipeline retries it, until it has failed
        in FAILED_RUNS runs with the same settings (failure_settings: the models' settings,
        how to reach them included but not the key, the gates' and max_tokens). After that it is left
        alone (see abandoned) unless retry_failed is set or one of those settings
        changes; a decision clears the failures before it. rebuild_gaps mines the rollouts again
        instead of reusing gaps.jsonl. A decision is reused only for a gap with the same
        inputs and the same models and gate settings (the digests decisions record),
        and only while the catalog state it depended on lasts (see holds); any other gap
        is decided again. A gap's latest decision with these settings is the one that counts.

        Raises:
            ValueError: The gaps cannot be mined or reused (see gaps).
        """
        record = self.work_dir / "decisions.jsonl"
        latest, failures = {}, Counter()
        for line in json_lines(record) if record.exists() else []:
            try:
                entry = json.loads(line)
                key = (entry["source"], entry.get("gap_sha256"))
            except (ValueError, KeyError, TypeError):
                continue  # a line cut short when a run stopped
            if entry.get("decision") == "failed":
                failures[key] += entry.get("failure_settings_sha256") == self.failure_settings
            elif entry.get("settings_sha256") == self.decision_settings:
                latest[key] = entry
                failures.pop(key, None)  # a decision ends the failures before it
        decided = {key for key, entry in latest.items() if self.holds(entry)}
        given_up = set() if retry_failed else {key for key, count in failures.items() if count >= FAILED_RUNS}
        # A gap twice in gaps.jsonl is one task, processed and charged once.
        gaps = list({(gap.source, _gap_digest(gap)): gap for gap in self.gaps(
            rollouts, rebuild=rebuild_gaps, retry_failed=retry_failed)}.values())
        self.abandoned = sorted(gap.source for gap in gaps
                                if (gap.source, _gap_digest(gap)) in given_up - decided)
        pending = [gap for gap in gaps if (gap.source, _gap_digest(gap)) not in decided | given_up]
        with ThreadPoolExecutor(self.settings["workers"]) as pool:
            results = list(pool.map(self.attempt, pending))
        return Counter(result["decision"] for result in results)

    def attempt(self, gap):
        """Process one gap; when a model or network call fails, record the failure and leave it undecided."""
        try:
            return self.process(gap)
        except Exception as exc:
            logger.warning("Gap %s stays undecided: %s: %s", gap.source, type(exc).__name__, exc)
            self.decide(gap, {"decision": "failed", "error": f"{type(exc).__name__}: {exc}"[:500],
                              "failure_settings_sha256": self.failure_settings})
            return {"decision": "failed"}

    def process(self, gap):
        """Reuse, or generate and verify with feedback-driven retries; record the decision."""
        with self.lock:
            match, similarity = closest(f"{gap.question} {gap.reason}", self.descriptions)
        if similarity > self.settings["reuse_threshold"]:
            return self.decide(gap, {"decision": "reused", "tool": match, "similarity": similarity})
        source, feedback, collisions = self.generator.write(gap), [], set()

        def record(outcome):
            # The names the attempts collided with: the outcome holds only while they are all taken.
            taken = {"collisions": sorted(collisions)} if collisions else {}
            return self.decide(gap, {**outcome, "attempts": feedback, **taken})
        for attempt in range(self.settings["retries"] + 1):
            outcome = self.verify(gap, source)
            if "decision" in outcome:
                return record(outcome)
            feedback.append(outcome["feedback"])
            if "collision" in outcome:
                collisions.add(outcome["collision"])
            if attempt < self.settings["retries"]:
                source = self.generator.revise(gap, source or "", outcome["feedback"])
        return record({"decision": "rejected"})

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
        elif metadata and self.taken(metadata["tool_name"]):
            return {"feedback": f"Gate 1: a tool named {metadata['tool_name']} already exists; choose another name",
                    "collision": metadata["tool_name"]}
        if issues:
            return {"feedback": "Gate 1: " + "; ".join(issues)}
        passed, critique, scores = review(self.models["critic"], gap.question, metadata, source,
                                          mean=self.settings["critic_mean"], minimum=self.settings["critic_min"],
                                          max_tokens=self.settings["max_tokens"])
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
                tools = catalog_tools(directory, settings={
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

        The module is published whole: written and synced to a hidden temporary file in
        the catalog, given the mode a newly created file gets (0666 under the umask, not
        mkstemp's 0600), then hard-linked to its name, which fails if the name exists. A
        reader (a router starting, say) never sees a partial module, and a run stopped
        midway leaves none. Where the file system has no hard links, the name is checked
        and the file renamed to it, which leaves a small race (logged once). An existing
        file of that name is a collision even when the module in it registers a
        different tool name.
        """
        name = metadata["tool_name"]
        path = self.catalog / f"{name}.py"
        with self.lock:
            match, similarity = closest(metadata["tool_description"], self.descriptions)
            if similarity >= self.settings["dedup_threshold"]:
                return {"decision": "duplicate", "tool": name, "match": match, "similarity": similarity}
            if name in self.descriptions:  # another gap of this run admitted it since the collision check
                return {"decision": "duplicate", "tool": name, "match": name}
            # A leading dot and no .py suffix: catalog_modules never lists the temporary file.
            descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=self.catalog)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write(source)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, 0o666 & ~_umask())
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    return {"decision": "duplicate", "tool": name, "file": path.name}
                except OSError as exc:
                    if exc.errno not in NO_HARD_LINKS:
                        raise
                    _warn_no_hard_links(self.catalog, exc)
                    if os.path.lexists(path):
                        return {"decision": "duplicate", "tool": name, "file": path.name}
                    os.rename(temporary, path)
            finally:
                if os.path.lexists(temporary):
                    os.unlink(temporary)
            # The tool is in the catalog from here on, whatever follows.
            self.descriptions[name] = metadata["tool_description"]
            _sync_directory(self.catalog)  # the new name survives a crash too
        return {"decision": "admitted", "tool": name, "scores": scores}

    def taken(self, name):
        """Whether a tool, or a module file (whatever name it registers), already has this name."""
        return name in self.descriptions or os.path.lexists(self.catalog / f"{name}.py")

    def holds(self, entry):
        """Whether the catalog still bears out a recorded decision.

        The tool it admitted, reused or matched must exist; a rejection or a duplicate also
        needs every name its attempts collided with to be taken still. A rejection without
        collisions does not depend on the catalog.
        """
        if entry.get("decision") in {"admitted", "reused"}:
            return entry.get("tool") in self.descriptions
        if not all(self.taken(name) for name in entry.get("collisions") or []):
            return False
        if entry.get("decision") == "duplicate":
            if "file" in entry:
                return os.path.lexists(self.catalog / entry["file"])
            return entry.get("match") in self.descriptions
        return True

    def decide(self, gap, result):
        """Append one decision, with the digests of its gap and settings, to decisions.jsonl and return it.

        A line a stopped run left cut short is ended first, so this one stays a line of its own.
        """
        line = json.dumps({"source": gap.source, "question": gap.question, "gap_sha256": _gap_digest(gap),
                           "settings_sha256": self.decision_settings, **result}, ensure_ascii=False)
        with self.lock, open(self.work_dir / "decisions.jsonl", "a+b") as stream:
            if stream.seek(0, os.SEEK_END):
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b"\n":
                    stream.write(b"\n")  # in append mode every write goes to the end
            stream.write(line.encode("utf-8") + b"\n")
        return result


def _check_settings(settings):
    """Raise ValueError unless each setting has its type and range (a null timeout would never expire)."""
    if not (isinstance(settings["encoder"], str) and settings["encoder"].strip()):
        raise ValueError("encoder must name a sentence-transformers model")
    for key, least in (("clusters", 1), ("workers", 1), ("retries", 0), ("max_tokens", 1)):
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


# The preset fields that only say where the agent's services are and how to sign in to them:
# the pipeline contacts none of them, so their variables need not be set for it.
UNUSED_FIELDS = re.compile(r"services\.(?:client\.base_url|tools\.[^.]+\.server_url|models\.[^.]+\.(base|key))")


def _unused(path):
    """Stand in for an unset variable in a preset field the pipeline never contacts: a service endpoint or key.

    Any other unset variable, in the toolsmith section, a model alias or anywhere else,
    still stops the pipeline as it stops the router.
    """
    found = UNUSED_FIELDS.fullmatch(path)
    if not found:
        return None
    return "unset" if found.group(1) == "key" else "http://unset.invalid"


def json_lines(path):
    """The lines of a JSON-lines file, split at line feeds only: the JSON text may hold U+2028 and the like.

    Each line is decoded on its own, so a last record a crash cut inside a multibyte
    character spoils only that line (it no longer parses as JSON), not the file.
    """
    return [line.decode("utf-8", "replace") for line in path.read_bytes().split(b"\n") if line.strip()]


# What os.link raises on a file system without hard links (or across devices): publish by rename instead.
NO_HARD_LINKS = {errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV, errno.EMLINK}
_warned_no_hard_links = []


def _warn_no_hard_links(directory, exc):
    """Log once that modules are published by check-then-rename, which another writer could race."""
    if not _warned_no_hard_links:
        _warned_no_hard_links.append(True)
        logger.warning("The catalog %s has no hard links (%s); new modules are published by checking the name and "
                       "renaming, so a file created there at the same moment by another process could be replaced",
                       directory, os.strerror(exc.errno))


_warned_no_directory_sync = []


def _sync_directory(directory):
    """Sync a directory's entries; some network and FUSE file systems cannot, which is logged once."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        if not _warned_no_directory_sync:
            _warned_no_directory_sync.append(True)
            logger.warning("The catalog directory %s cannot be synced (%s); a module just published could be lost in "
                           "a system crash", directory, exc)


def _umask():
    """The process umask, read without changing it where Linux reports it."""
    try:
        with open("/proc/self/status", encoding="ascii") as stream:
            return int(next(line for line in stream if line.startswith("Umask:")).split()[1], 8)
    except (OSError, StopIteration, ValueError):
        mask = os.umask(0o022)
        os.umask(mask)
        return mask


# Query parameters that carry a credential in an endpoint URL.
CREDENTIAL_PARAMETERS = re.compile(r"key|api[-_]?key|apikey|token|access[-_]?token|secret|password|passwd|"
                                   r"signature|sig|auth|authorization|credential|code", re.IGNORECASE)


def _without_credentials(model):
    """A model's settings without its key, and its base URL without userinfo or credential query parameters."""
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    settings = {key: value for key, value in model.items() if key != "key"}
    if isinstance(settings.get("base"), str):
        url = urlsplit(settings["base"])
        host = url.netloc.rsplit("@", 1)[-1]  # user:password@ goes
        query = [(name, value) for name, value in parse_qsl(url.query, keep_blank_values=True)
                 if not CREDENTIAL_PARAMETERS.fullmatch(name)]
        settings["base"] = urlunsplit((url.scheme, host, url.path, urlencode(query), url.fragment))
    return settings


def _digest(value):
    """SHA-256 of a JSON value's canonical text."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _saved_critiques(saved, analyst, failure_settings):
    """Return the critiques of analyst in the bytes of a critiques.jsonl and the failures among them.

    Returns:
        (critiques, failures): critique_key to [label, reason], and critique_key to the
        number of critiques that failed with failure_settings (the analyst's settings,
        how to reach it but not its key, and max_tokens; see Pipeline). A later line for the
        same key wins, and a line that does not parse (cut short when a run stopped) is
        skipped.
    """
    critiques, failures = {}, Counter()
    for line in saved.decode("utf-8", "replace").split("\n"):
        try:
            entry = json.loads(line)
            if entry["analyst"] != analyst:
                continue
            if "failed" in entry:
                failures[entry["key"]] += entry.get("failure_settings_sha256") == failure_settings
            else:
                critiques[entry["key"]] = [entry["label"], entry["reason"]]
        except (ValueError, KeyError, TypeError):
            continue
    return critiques, failures


def _difference(recorded, current):
    """Say how the recorded mining inputs differ from the current ones, for the cache error."""
    if recorded is None:
        return "without a record of its rollouts and settings (no mining.json)"
    recorded = {key: value for key, value in recorded.items() if key not in MINING_RESULTS}
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
