"""Phase 1: mine capability gaps from failed rollouts and keep one representative per cluster."""
import hashlib
import json
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .prompts import prompt

logger = logging.getLogger(__name__)

LABELS = ("NEW_TOOL_NEEDED", "ARGUMENT_MISTAKE", "SEARCH_NOISE", "AMBIGUOUS")


@dataclass
class Gap:
    """A failed query with the tools the agent tried and the self-critique's verdict."""

    source: str
    question: str
    prediction: str
    gold: str
    image: bool
    tried_tools: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    label: str = ""
    reason: str = ""

    def to_json(self):
        """Return the gap as one JSON line."""
        return json.dumps(asdict(self), ensure_ascii=False)


# The evaluator writes judged rows as <model>_judge[_judge_<prompt>] in the run's PRED_FORMAT;
# <model>_acc... files beside them hold scores.
JUDGED = re.compile(r"(?!.*_acc[_.]).+_judge[^.]*\.(?:xlsx|tsv|json)")


def judged_predictions(directory):
    """Return the judged predictions file that failures reads from a dataset directory.

    Raises:
        ValueError: The directory holds no judged predictions, or several.
    """
    judged = sorted(path for path in Path(directory).iterdir() if path.is_file() and JUDGED.fullmatch(path.name))
    if not judged:
        datasets = sorted(str(child) for child in Path(directory).iterdir() if child.is_dir() and any(
            path.is_file() and JUDGED.fullmatch(path.name) for path in child.iterdir()))
        if datasets or Path(directory, "run_manifest.json").is_file():
            raise ValueError(f"{directory} is a run directory; pass its dataset directories to --rollouts instead"
                             + (f": {' '.join(datasets)}" if datasets else ""))
        raise ValueError(f"{directory} has no judged predictions (<model>_judge*.xlsx, .tsv or .json); "
                         "evaluate the run first")
    if len(judged) > 1:
        raise ValueError(f"{directory} holds several judged predictions ({', '.join(path.name for path in judged)}); "
                         "mine a directory judged once")
    return judged[0]


def judged_rows(path):
    """Read judged predictions with the evaluator's format-aware loader and return them as mappings."""
    import pandas as pd
    from evaluation_kit.smp.file import load

    data = load(str(path))
    return (data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)).to_dict("records")


def failed_rows(directory):
    """Yield each judged row marked incorrect with the path of its trajectory, which may not exist.

    Raises:
        ValueError: See judged_predictions; or the judged rows have no hit column, like
            the score summary a dataset judged other than per query writes.
    """
    directory = Path(directory)
    path = judged_predictions(directory)
    rows = judged_rows(path)
    if rows and not any("hit" in row for row in rows):
        raise ValueError(f"{path} has no hit column: mining reads a verdict per query, as the judged "
                         "predictions of the ToolSmith training pools hold")
    for row in rows:
        if row.get("hit") == 0:
            yield row, directory / "full_trajectories" / f"{row['index']}.json"


def evidence_digest(directory):
    """SHA-256 over the trajectories that failures reads, a missing trajectory included."""
    digest = hashlib.sha256()
    for row, trajectory in failed_rows(directory):
        content = hashlib.sha256(trajectory.read_bytes()).hexdigest() if trajectory.exists() else "missing"
        digest.update(f"{row['index']}\0{content}\n".encode())
    return digest.hexdigest()


def failures(directory):
    """Layer A: yield the queries the LLM judge marked incorrect in one dataset directory of a run.

    Args:
        directory: A dataset directory written by openvlharness-eval, holding the
            judged predictions (<model>_judge*.xlsx, .tsv or .json with a hit column)
            and full_trajectories/.

    Raises:
        ValueError: The directory holds no judged predictions.
    """
    directory = Path(directory)
    failed = missing = 0
    for row, trajectory in failed_rows(directory):
        failed += 1
        missing += not trajectory.exists()
        messages = json.loads(trajectory.read_text()).get("messages", []) if trajectory.exists() else []
        tools, errors = _tool_activity(messages)
        image = row.get("image_path")
        yield Gap(source=f"{directory.name}:{row['index']}", question=str(row.get("question", "")),
                  prediction=str(row.get("prediction", "")), gold=str(row.get("answer", "")),
                  image=isinstance(image, str) and bool(image.strip(" []'\"")), tried_tools=tools, errors=errors)
    if missing:
        logger.warning("%d of %d failed queries in %s have no trajectory (FULL_TRAJECTORIES_LIMIT may have capped "
                       "them); they are mined without the tools they tried", missing, failed, directory)


# A failed call reaches the model as "<error code>: <message>" (models.failure_text).
FAILURE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+: \S.*", re.DOTALL)


def _tool_activity(messages):
    """Return the called tool names and the last three tool error messages of a trajectory.

    Chat Completions trajectories record calls in assistant tool_calls and each
    result in a tool message ("ok", or the failure text); Responses trajectories
    use function_call and function_call_output items. A JSON execute envelope with
    "ok": false in any message also counts.
    """
    tools, errors = [], []
    for message in messages:
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or []:
            tools.append((call.get("function") or {}).get("name") or call.get("name"))
        if message.get("type") == "function_call":
            tools.append(message.get("name"))
        result = (message.get("content") if message.get("role") == "tool" else
                  message.get("output") if message.get("type") == "function_call_output" else None)
        if isinstance(result, str) and FAILURE.fullmatch(result.strip()):
            errors.append(result.strip()[:200])
            continue
        content = message.get("content", message.get("output"))
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        if '"ok": false' in text:
            errors.extend(re.findall(r'"message": "([^"]{1,200})', text))
    return [tool for tool in tools if tool], errors[-3:]


class UnparsedCritique(ValueError):
    """A complete self-critique reply that names no label: a failed critique, not an AMBIGUOUS one."""


# A "LABEL:" field starts its line, in any case, perhaps after a bullet, a number, what Markdown
# emphasis leaves or a short answer prefix ("Final answer:", "Answer:", "Reply:", "Output:");
# "the label: ..." or "label-reading" inside a sentence is prose. Its value begins with one label,
# perhaps bracketed or quoted: as one token, underscores and all, in any case, followed by the end
# of the field, a separator, or a reason word ("NEW_TOOL_NEEDED because ..."), and the reason after
# it may name other labels; never by prose that goes on about the label ("NEW_TOOL_NEEDED is not
# it", "... does not apply"); or with spaces or hyphens for the underscores, followed by the end or
# a separator.
# Without the field a line counts that is one label in capitals as the prompt lists them, perhaps a
# bullet; so does the prompt's own bullet form, "NEW_TOOL_NEEDED: <reason>", as the reply's first
# line (or the second, after one preamble line that names no label) with no other label line after
# it. Prose that names a label ("not NEW_TOOL_NEEDED",
# "Ambiguous? No.", "Search noise is unlikely") never counts.
_SEPARATOR = r"(?:\s*[:=：]|\s+[-–—]|[–—])\s*"  # a dash only after a space: "label-reading" is a word
_PREFIX = r"(?:(?:final\s+answer|answer|reply|output)\s*[:：][^\w\n]*)?"
_LABEL_FIELD = re.compile(r"^[^\w\n]*(?:\d+[.)][^\w\n]*)?" + _PREFIX + "label" + _SEPARATOR,
                          re.IGNORECASE | re.MULTILINE)
_WHY_FIELD = re.compile(r"\bwhy" + _SEPARATOR, re.IGNORECASE)
_WHY_HEADING = re.compile(r"^[^\w\n]*why[^\w\n]*\n\s*", re.IGNORECASE | re.MULTILINE)  # "### Why" on its own line
_NAMED = re.compile(r"\b(?:" + "|".join(LABELS) + r")\b")
_CLOSE, _END = r"(?![A-Za-z0-9_])[\])\"'>]*", r"\s*(?:\Z|[-–—.,;:!|(\[]\s*)"
_REASON_WORD = r"\s+(?=(?:because|since|as|due\s+to)\b)"  # not "is", "does", "not", "seems", ...
_LABEL_VALUE = re.compile(r"[\[(\"'<\s]*(?:(" + "|".join(LABELS) + ")" + _CLOSE + "(?:" + _REASON_WORD + "|" + _END
                          + ")|("
                          + "|".join(label.replace("_", "[ -]") for label in LABELS) + ")" + _CLOSE + _END + ")",
                          re.IGNORECASE)
# A label line: one label in capitals, perhaps as a bullet, alone (a period may end it) or followed
# at once by a colon and a reason (group 2).
_LABEL_LINE = re.compile(r"(?:[-+•]|\d+[.)])?\s*(" + "|".join(LABELS) + r")(?::(.*)|\.?)")


def parse_critique(reply):
    """Return the label and reason of a self-critique reply, read strictly.

    A reply that is one JSON object with a "label" key (in any case) gives its value, which
    must be one label, and the reason in its "why" or "reason" key. With "LABEL:" fields
    (in any case; Markdown emphasis aside) that start their lines, perhaps after "Final
    answer:", "Answer:", "Reply:" or "Output:", their values give the label: each begins
    with one label (in any case), which as one token may be followed by free text, and they
    agree. Without one, a line holding just one label in capitals, perhaps as a bullet,
    gives it; so does the reply's first line when it begins with one label in capitals and
    a colon, as the prompt lists them ("- NEW_TOOL_NEEDED: capability X is missing"), if no
    other label line follows. The reason follows "WHY:", or else the label.

    Raises:
        UnparsedCritique: No label read that way: prose that names one, a negation, an
            unknown LABEL value, or several labels.
    """
    found = _json_critique(reply)
    if found:
        return found
    text = re.sub(r"[*`#]|_{2,}", "", reply or "")  # Markdown emphasis, code and headings
    why = min((found for found in (_WHY_FIELD.search(text), _WHY_HEADING.search(text)) if found),
              key=lambda found: found.start(), default=None)
    marks = list(_LABEL_FIELD.finditer(text))
    if marks:
        labels, rest = set(), None
        for mark in marks:
            end = text.find("\n", mark.end())
            end = len(text) if end < 0 else end
            later = _WHY_FIELD.search(text, mark.end(), end)
            value = text[mark.end():later.start() if later else end]
            found = _LABEL_VALUE.match(value)
            if not found:
                raise UnparsedCritique(f"the self-critique's LABEL is not one of {', '.join(LABELS)}: {value[:200]!r}")
            labels.add(re.sub(r"[ -]", "_", found.group(1) or found.group(2)).upper())
            rest = value[found.end():] if rest is None else rest
    else:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        found = [(index, match) for index, line in enumerate(lines) if (match := _LABEL_LINE.fullmatch(line))]
        # "NEW_TOOL_NEEDED: <reason>" counts only as the first line, or the second after one preamble
        # line that names no label ("Here is my verdict:"), and as the only label line, not as one
        # bullet of a list of labels.
        preamble = bool(lines) and not (found and found[0][0] == 0) and not _NAMED.search(lines[0])
        first = 1 if preamble else 0
        if any(match.group(2) is not None for _, match in found) and [index for index, _ in found] != [first]:
            problem = "several labels" if len(found) > 1 else "its label after other text"
            raise UnparsedCritique(f"the self-critique gives {problem} of {', '.join(LABELS)}: {text[:200]!r}")
        labels = {match.group(1) for _, match in found}
        rest = (found[0][1].group(2) or "") if found else ""
    if len(labels) != 1:
        raise UnparsedCritique(f"the self-critique gives {'several labels' if labels else 'no label'} of "
                               f"{', '.join(LABELS)}: {text[:200]!r}")
    reason = text[why.end():].split("\n", 1)[0] if why else rest
    return labels.pop(), reason.strip(" \t:=-–—：()[]\"'").strip()


def _json_critique(reply):
    """Return the label and reason of a reply that is one JSON object with a "label" key, else None.

    The object may sit in a Markdown code fence. Its keys count in any case.

    Raises:
        UnparsedCritique: The object's label is not exactly one label.
    """
    body = re.sub(r"\A```[A-Za-z]*\s*|\s*```\Z", "", (reply or "").strip())
    try:
        data = json.loads(body) if body.startswith("{") else None
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    fields = {}
    for key, value in data.items():
        fields.setdefault(str(key).lower(), []).append(value)
    if "label" not in fields:
        return None
    labels = {re.sub(r"[ -]", "_", value.strip()).upper() if isinstance(value, str) else None
              for value in fields["label"]}
    if len(labels) != 1 or not labels <= set(LABELS):
        raise UnparsedCritique(f"the self-critique's label is not one of {', '.join(LABELS)}: "
                               f"{fields['label'][:2]!r:.200}")
    reason = next((value for key in ("why", "reason") for value in fields.get(key, []) if isinstance(value, str)), "")
    return labels.pop(), reason.strip()


def critique(gap, analyst, max_tokens):
    """Layer B: label why the run failed; only NEW_TOOL_NEEDED gaps go on to generation.

    Raises:
        IncompleteReply: The analyst's reply ran out of tokens or holds no text.
        UnparsedCritique: The reply names no label (see parse_critique).
        Either way the failure is not a label: the critique is retried by a later run.
    """
    reply = analyst.generate(prompt("self_critique.user").format(
        question=gap.question, prediction=gap.prediction, gold=gap.gold,
        tool_counts=json.dumps(dict(Counter(gap.tried_tools))),
        recent_errors=json.dumps(gap.errors, ensure_ascii=False)),
        system_prompt=prompt("self_critique.system"), max_tokens=max_tokens, require_complete=True)
    gap.label, gap.reason = parse_critique(reply)
    return gap


def representatives(gaps, encoder, clusters):
    """Cluster gaps by question, prediction and gold answer; return the member nearest each centroid.

    Gaps are embedded with the sentence encoder and grouped by agglomerative
    clustering (average linkage, cosine distance). With no more gaps than
    clusters, every gap is its own representative.
    """
    if len(gaps) <= clusters:
        return list(gaps)
    from sklearn.cluster import AgglomerativeClustering

    from ..retrieval import load_encoder

    texts = [f"Question: {g.question}\nPredicted: {g.prediction[:200]}\nGold: {g.gold}" for g in gaps]
    embeddings = load_encoder(encoder).encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    labels = AgglomerativeClustering(n_clusters=clusters, metric="cosine", linkage="average").fit_predict(embeddings)
    chosen = []
    for cluster in range(clusters):
        members = np.flatnonzero(labels == cluster)
        centroid = embeddings[members].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        chosen.append(gaps[members[int(np.argmax(embeddings[members] @ centroid))]])
    return chosen


def critique_key(gap):
    """SHA-256 of a failed query's evidence, under which its critique is kept between runs."""
    return hashlib.sha256(Gap(**{**asdict(gap), "label": "", "reason": ""}).to_json().encode()).hexdigest()


@dataclass
class Mining:
    """What one mining found: the representatives, and what became of each critique."""

    representatives: list
    failed: list  # sources whose critique failed in this mining and is retried by the next
    abandoned: list  # sources whose critique failed in give_up_after minings, not asked again
    labels: dict  # the number of critiqued queries with each label
    failures: int = 0  # critiques that failed in this mining, abandoned ones included


def mine(directories, analyst, *, encoder, clusters, workers=8, known=None, save=None, max_tokens=12800,
         failed_before=None, give_up_after=None):
    """Run Layer A and Layer B over every rollout directory, then pick cluster representatives.

    A critique in known (critique_key to [label, reason]) is reused. save, when given, is
    called with (critique_key, label, reason) as each new critique completes, so a mining
    that stops part way keeps the critiques already paid for, and with (critique_key,
    None, error) as one fails. A failed critique leaves its query out of this mining, so
    one refused or failed model call does not discard the others. failed_before counts,
    by critique_key, the minings whose critique of a query failed; with give_up_after, a
    query whose critique failed that many times is not asked about again.

    Returns:
        A Mining.
    """
    known, failed_before = dict(known or {}), dict(failed_before or {})

    def attempt(gap):
        key = critique_key(gap)
        if key in known:
            gap.label, gap.reason = known[key]
            return "critiqued", gap
        if give_up_after is not None and failed_before.get(key, 0) >= give_up_after:
            return "abandoned", gap
        try:
            critique(gap, analyst, max_tokens)
        except Exception as exc:
            logger.warning("The self-critique of %s failed: %s: %s", gap.source, type(exc).__name__, exc)
            if save:
                save(key, None, f"{type(exc).__name__}: {exc}"[:500])
            given_up = give_up_after is not None and failed_before.get(key, 0) + 1 >= give_up_after
            return ("abandoned" if given_up else "failed"), None
        if save:
            save(key, gap.label, gap.reason)
        return "critiqued", gap

    gaps = [gap for directory in directories for gap in failures(directory)]
    outcomes = {"critiqued": [], "failed": [], "abandoned": []}
    failed_now = 0
    with ThreadPoolExecutor(workers) as pool:
        for gap, (outcome, result) in zip(gaps, pool.map(attempt, gaps)):
            failed_now += result is None
            outcomes[outcome].append(result if outcome == "critiqued" else gap.source)
    critiqued = outcomes["critiqued"]
    chosen = representatives([gap for gap in critiqued if gap.label == "NEW_TOOL_NEEDED"], encoder, clusters)
    return Mining(chosen, outcomes["failed"], outcomes["abandoned"], dict(Counter(gap.label for gap in critiqued)),
                  failed_now)
