<h1 align="center">ToolSmith</h1>

<h3 align="center">An Automatic Tool-Generation Framework for Multimodal Search Agents</h3>

<p align="center">
  <a href="https://threesr.github.io/toolsmith/"><img src="https://img.shields.io/badge/Project-Homepage-2653c1.svg?style=for-the-badge" height="25" alt="Project homepage"></a>
  <a href="https://aclanthology.org/2026.findings-emnlp.1699/"><img src="https://img.shields.io/badge/Findings-EMNLP%202026-b31b1b.svg?style=for-the-badge" height="25" alt="Findings of EMNLP 2026"></a>
</p>

ToolSmith grows the toolbox of a multimodal search agent from the agent's own failures.
It mines capability gaps from failed trajectories, has an LLM write each new tool as
Python code, admits a tool only after two verification gates, and retrieves the
relevant generated tools for every query at inference time. Generated tools run in the
OpenVLHarness tool router next to the base tools and reach them through it.

## How it works

1. **Trajectory mining.** The baseline agent runs on a training pool. Layer A keeps
   the trajectories the LLM judge marks incorrect; Layer B, an LLM self-critique,
   labels each one `NEW_TOOL_NEEDED`, `ARGUMENT_MISTAKE`, `SEARCH_NOISE` or
   `AMBIGUOUS`. Only `NEW_TOOL_NEEDED` trajectories become capability gaps. Gaps are
   embedded with a sentence encoder, clustered (agglomerative, average linkage,
   cosine distance), and one representative per cluster moves on.
2. **Reuse or grow.** A gap whose TF-IDF similarity to an existing tool description
   exceeds a threshold is covered by that tool and skipped.
3. **Tool generation.** A generator LLM writes the tool, conditioned on the gap, the
   expected answer shape, the callable tools and six example tools.
4. **Verification and inference.** Gate 1 scans the source for dynamic execution and
   I/O, then runs the tool once on dummy input in a separate, resource-limited process
   under a ten-second budget, with every tool call answering with a failure. Gate 2 is
   an LLM critic that scores parametric usage,
   description abstractness, edge-case handling and task alignment. A rejected
   candidate is revised with the gate's feedback. A tool that passes both gates is
   admitted unless its description nearly duplicates an existing tool. At inference,
   the agent sees the base tools plus the top-K generated tools retrieved for the
   query.

| Setting | Value |
| --- | --- |
| Sentence encoder | `BAAI/bge-m3` |
| Clusters | 500 |
| Reuse threshold (TF-IDF) | 0.40 |
| Retries per candidate | 2 |
| Gate 1 budget | 10 seconds |
| Gate 2 thresholds | mean score at least 3.0, every axis at least 1 |
| Duplicate threshold (TF-IDF) | 0.55 |
| Generated tools per query | 10 |

## Use a catalog

```bash
pip install -e ".[server,qwen,eval,toolsmith]"
pip install open_clip_torch  # for image_similarity and species_id
```

[configs/agent/qwen3_8b_toolsmith.yaml](../configs/agent/qwen3_8b_toolsmith.yaml) runs
ToolSmith with Qwen3-VL-8B, and
[qwen3_8b_toolsmith_baseline.yaml](../configs/agent/qwen3_8b_toolsmith_baseline.yaml) is
the same agent without generated tools. Both use the ToolSmith system prompt, the base
tools except depth and camera pose, temperature 0.3, top-p 0.95, repetition penalty 1.1,
up to 2,048 output tokens and a 120,000-token context. Start a preset's tool router and
evaluate with it as in the main [quick start](../README.md#quick-start):

```bash
python -m openvlharness.services.server --config configs/agent/qwen3_8b_toolsmith.yaml
openvlharness-eval --config configs/agent/qwen3_8b_toolsmith.yaml --data MMSearch
```

```yaml
toolsmith:
  catalog: path/to/catalog  # optional; the packaged catalog by default
  top_k: 10
  encoder: BAAI/bge-m3
```

The router loads every tool in the catalog. The agent embeds the tool descriptions once
and, for each question, adds the `top_k` generated tools closest to the question text.
Results are recorded under the `toolsmithagent` run directory.

The [packaged catalog](../openvlharness/toolsmith/tools) holds example tools, among them
`wikidata_query`, `gazetteer_lookup`, `unit_compute`, `media_catalog` (TMDB lookups
read `TMDB_API_KEY`), `image_similarity` and `species_id`, which other tools build on.

## Grow a catalog

1. Evaluate the baseline preset with an LLM judge on a training pool of your own; the
   paper used FVQA train (3,312 questions), InfoSeek val (3,000) and EncyclopedicVQA val
   (2,862). Keep evaluation data out of the pool.
2. Copy the packaged catalog to a directory, copy the ToolSmith preset and set its
   `toolsmith.catalog` to that directory.
3. Point `agent` in [configs/toolsmith/pipeline.yaml](../configs/toolsmith/pipeline.yaml)
   at that preset and run the pipeline on the evaluated dataset directories:

```bash
export TOOLSMITH_MODEL_BASE_URL=https://api.openai.com/v1
export TOOLSMITH_MODEL_API_KEY=$OPENAI_API_KEY
python -m openvlharness.toolsmith.pipeline --config configs/toolsmith/pipeline.yaml \
  --rollouts outputs/Qwen3-VL-8B-Instruct/mainagent/<run>/<dataset> --work-dir toolsmith_run
```

Admitted tools are written to the catalog. `gaps.jsonl` keeps the mined
representatives and `decisions.jsonl` records each gap's outcome with the gate feedback;
a rerun resumes from both. Serve the grown catalog with the same preset.

## Write a tool

A tool is a module with one `GeneratedTool` subclass. `call_tool` reaches the base tools
and the other catalog tools through the tool router configured in
`services.client.base_url`, so tool code holds no addresses or ports. An `image_id`
parameter arrives in `execute` as `image_base64`.

```python
from openvlharness.toolsmith import GeneratedTool, ToolCallError


class EntityPropertyLookup(GeneratedTool):
    def __init__(self):
        super().__init__(
            tool_name="entity_property_lookup",
            tool_description=("Look up one property of a named entity in Wikidata. "
                              "DO NOT USE FOR: entities known only from an image."),
            input_schema={"type": "object", "properties": {
                "entity": {"type": "string"}, "property": {"type": "string"}},
                "required": ["entity", "property"]})

    def execute(self, entity="", property=""):
        try:
            record = self.call_tool("wikidata_query", entity=entity, property=property)
        except ToolCallError as exc:
            return {"error": str(exc)}
        matched = record.get("matched") or {}
        if matched.get("confidence", 0) < 1.0:
            return {"error": f"no confident '{property}' for {entity}"}
        return ", ".join(matched["values"][:3])
```

The [examples](../openvlharness/toolsmith/pipeline/examples) shown to the generator and
the [prompts](../openvlharness/toolsmith/pipeline/prompts) of every pipeline step are
part of the package.

## Citation

```bibtex
@inproceedings{sun-etal-2026-toolsmith,
    title = "{T}ool{S}mith: An Automatic Tool-Generation Framework for Multimodal Search Agents",
    author = "Sun, Rui  and
      Huang, Junhua  and
      Zancato, Luca  and
      Islam, Md Mofijul  and
      Chang, Kai-Wei",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2026",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-emnlp.1699/",
    doi = "10.18653/2026.findings-emnlp.1699",
    pages = "36305--36324"
}
```
