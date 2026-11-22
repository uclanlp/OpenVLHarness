"""Phase 1: mine capability gaps from failed rollouts and keep one representative per cluster."""
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .prompts import prompt

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


def failures(directory):
    """Layer A: yield the queries the LLM judge marked incorrect in one dataset directory of a run.

    Args:
        directory: A dataset directory written by openvlharness-eval, holding the
            judged predictions (*_judge*.xlsx with a hit column) and full_trajectories/.

    Raises:
        ValueError: The directory holds no judged predictions.
    """
    import pandas as pd

    directory = Path(directory)
    judged = sorted(directory.glob("*_judge*.xlsx"))
    if not judged:
        raise ValueError(f"{directory} has no judged predictions (*_judge*.xlsx); evaluate the run first")
    for row in pd.read_excel(judged[0]).to_dict("records"):
        if row.get("hit") != 0:
            continue
        trajectory = directory / "full_trajectories" / f"{row['index']}.json"
        messages = json.loads(trajectory.read_text()).get("messages", []) if trajectory.exists() else []
        tools, errors = _tool_activity(messages)
        image = row.get("image_path")
        yield Gap(source=f"{directory.name}:{row['index']}", question=str(row.get("question", "")),
                  prediction=str(row.get("prediction", "")), gold=str(row.get("answer", "")),
                  image=isinstance(image, str) and bool(image.strip(" []'\"")), tried_tools=tools, errors=errors)


def _tool_activity(messages):
    """Return the called tool names and the last three tool error messages of a trajectory."""
    tools, errors = [], []
    for message in messages:
        for call in message.get("tool_calls") or []:
            tools.append((call.get("function") or {}).get("name") or call.get("name"))
        content = message.get("content")
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
