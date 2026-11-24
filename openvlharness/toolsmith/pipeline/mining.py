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


def critique(gap, analyst):
    """Layer B: label why the run failed; only NEW_TOOL_NEEDED gaps go on to generation."""
    reply = analyst.generate(prompt("self_critique.user").format(
        question=gap.question, prediction=gap.prediction, gold=gap.gold,
        tool_counts=json.dumps(dict(Counter(gap.tried_tools))), recent_errors=json.dumps(gap.errors, ensure_ascii=False)),
        system_prompt=prompt("self_critique.system"), max_tokens=1024)
    label = re.search(r"LABEL:\s*([A-Z_]+)", reply or "")
    reason = re.search(r"WHY:\s*(.+)", reply or "")
    gap.label = label.group(1) if label and label.group(1) in LABELS else "AMBIGUOUS"
    gap.reason = reason.group(1).strip() if reason else ""
    return gap


def representatives(gaps, encoder, clusters):
    """Cluster gaps by question, prediction and gold answer; return the member nearest each centroid.

    Gaps are embedded with the sentence encoder and grouped by agglomerative
    clustering (average linkage, cosine distance). With no more gaps than
    clusters, every gap is its own representative.
    """
    if len(gaps) <= clusters:
        return list(gaps)
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering

    texts = [f"Question: {g.question}\nPredicted: {g.prediction[:200]}\nGold: {g.gold}" for g in gaps]
    embeddings = SentenceTransformer(encoder).encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    labels = AgglomerativeClustering(n_clusters=clusters, metric="cosine", linkage="average").fit_predict(embeddings)
    chosen = []
    for cluster in range(clusters):
        members = np.flatnonzero(labels == cluster)
        centroid = embeddings[members].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        chosen.append(gaps[members[int(np.argmax(embeddings[members] @ centroid))]])
    return chosen


def mine(directories, analyst, *, encoder, clusters, workers=8):
    """Run Layer A and Layer B over every rollout directory, then pick cluster representatives."""
    failed = [gap for directory in directories for gap in failures(directory)]
    with ThreadPoolExecutor(workers) as pool:
        critiqued = list(pool.map(lambda gap: critique(gap, analyst), failed))
    return representatives([gap for gap in critiqued if gap.label == "NEW_TOOL_NEEDED"], encoder, clusters)
