<h1 align="center">ToolSmith</h1>

<h3 align="center">An Automatic Tool-Generation Framework for Multimodal Search Agents</h3>

<p align="center">
  <a href="https://threesr.github.io/toolsmith/"><img src="https://img.shields.io/badge/Project-Homepage-2653c1.svg?style=for-the-badge" height="25" alt="Project homepage"></a>
  <a href="https://aclanthology.org/2026.findings-emnlp.1699/"><img src="https://img.shields.io/badge/Findings-EMNLP%202026-b31b1b.svg?style=for-the-badge" height="25" alt="Findings of EMNLP 2026"></a>
  <a href="https://huggingface.co/datasets/uclanlp/ToolSmith-Datasets"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Data-Training%20pools-ffc107.svg?style=for-the-badge" height="25" alt="ToolSmith training pools on Hugging Face"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-MIT-2e7d32.svg?style=for-the-badge" height="25" alt="MIT License"></a>
</p>

ToolSmith grows the toolbox of a multimodal search agent from the agent's own failures.
It mines capability gaps from failed trajectories, has an LLM write each new tool as
Python code, admits a tool only after two verification gates, and retrieves the
relevant generated tools for every query at inference time. Generated tools run in a
resource-limited worker process of the OpenVLHarness tool router and reach the base tools
through it.

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
   under a ten-second budget, with every tool call answering with a failure. The process
   runs in Linux namespaces with no network, and sees only the interpreter, the
   packages and the candidate, read-only. The scan only narrows what a candidate can
   express; the namespaces are what keep it from the host. The name, description and
   schema the gates and the catalog use come from loading the candidate in a confined
   worker, as the router will, and the name must be a string literal that no tool uses yet. Gate 2 is
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
  worker:                   # optional; the process that runs the generated tools
    environment: [TMDB_API_KEY]  # credential-like variables the tools may read
    timeout_seconds: 540    # per call; default 90% of services.client.timeout_seconds
    memory_gib: 128         # address-space limit (enforced on Linux)
    write_files: false      # true for image uploads, model downloads and PyTorch
    isolation: true         # confine the worker with Linux namespaces
    network: true           # false: no network inside the confined worker
```

The router loads every tool in the catalog. The agent embeds the tool descriptions once
and, for each question, adds the `top_k` generated tools closest to the question text.
Results are recorded under the `toolsmithagent` run directory. The run identity includes a
hash of the catalog's module files, so a run resumes only with the catalog it started with,
and inference starts only when the router reports the same hash for the catalog it loaded
at startup; restart the router after editing the catalog.

The router never imports the generated tools. When it starts, it copies the catalog and
runs the copy in a separate worker process. The worker starts in a private directory and
inherits the router's environment without credential-like variables (names containing
`KEY`, `TOKEN`, `SECRET`, `PASSWORD` or `CREDENTIAL`, or starting with `AWS_` or
`GOOGLE_APPLICATION`); `worker.environment` grants the ones its tools read. The filter goes
by name only, so other secrets, such as `DATABASE_URL`, a `*_PASS` variable, `GITHUB_PAT` or
a password in a proxy URL, reach the worker: keep them out of the router's environment.
On Linux each
of its processes has its address space capped at `worker.memory_gib`, but a tool that starts
child processes gets that much in each, and memory in its temporary directories is counted
separately; the worker asks Linux to kill it first when memory runs out, a request a tool
can take back. CPU time is not limited. For hard memory and CPU limits, run the router in
a cgroup or a container. The worker cannot write files unless `worker.write_files` is set;
local image uploads, model downloads and PyTorch need writes. A tool reaches the other catalog tools
and `Text_Search_Tool`, `Image_Search_Tool`, `Webpage_Visit_Tool`, `OCR_Tool` and
`Zoom_In_Tool` through the router, and calls to other router tools are refused. A call that
raises, crashes the worker or runs past `worker.timeout_seconds` returns an error to the
agent, and the router starts a new worker for later calls. The ToolSmith preset grants the
credentials that the packaged `visual_matches` (Serper Lens and the image upload) and
`media_catalog` (TMDB) read, and file writes.

With `worker.isolation` (the default) the worker runs in Linux user, mount, PID and IPC
namespaces, under the router's uid and without root. It sees the interpreter, the Python
packages, the catalog copy, the model caches and the resolver, certificate and time-zone
settings from `/etc` read-only, and has a private `/tmp`, home directory and `/dev/shm`,
each a memory-backed file system of at most 1 GiB; a file that a granted variable names,
such as a key file, is readable too. It cannot create nested user namespaces, which would
give it back the capabilities to mount others, and on Linux 5.14 and later it runs at
most 4,096 processes and threads. It cannot see the router's other files (the repository, other home directories,
credential files such as `~/.aws/credentials`) or any process outside its namespace, and
Hugging Face token files are hidden. The host's model caches are never writable: with
`write_files`, writes and downloads into them land in a private in-memory layer of at
most 1 GiB that is discarded when the worker stops (download models on the host
beforehand), and the local image-upload directory, which the image host serves, is
writable. `worker.network: false` removes network access as
well. With network access the worker shares the host's network, including services that
listen on loopback, among them the router itself: a tool can call any router tool over
HTTP, including those `call_tool` refuses, such as the local Python executor, which is
not a sandbox. It can also reach the host's abstract UNIX sockets, and `/proc/net` lists
the host's connections. Without network access it cannot resolve names either, as the
resolver's socket is not mounted. Most packaged and research-catalog tools query web services directly and
need network access; `call_tool` works without it. On a host without unprivileged user
namespaces, or one where the namespace cannot refuse nested ones, the router refuses to
start the worker unless `worker.isolation` is false.

The agent and the pipeline read the tools' names, descriptions and schemas from a
short-lived worker started the same way, so catalog code does not run in the evaluator or
on the mining host either. A `TOOLSMITH_LEGACY_*` environment variable, which research
shims that patch the agent when a catalog module imports them rely on, makes both import
the catalog in their own process instead, with that process's privileges; they log a
warning when they do. Wherever a catalog loads, a tool whose input schema is not a valid
JSON Schema, or refers to other documents (`$ref`, `$id`, `$schema`), is skipped with a
warning, and Gate 1 rejects such a candidate.

The [packaged catalog](../openvlharness/toolsmith/tools) holds example tools, among them
`wikidata_query`, `gazetteer_lookup`, `unit_compute`, `media_catalog` (TMDB lookups
read `TMDB_API_KEY`), `image_similarity` and `species_id`, which other tools build on.

## Grow a catalog

1. Evaluate the baseline preset with an LLM judge on a training pool. The paper's pool,
   FVQA train (3,312 questions), InfoSeek val (3,000) and EncyclopedicVQA val (2,862), is
   on Hugging Face as [uclanlp/ToolSmith-Datasets](https://huggingface.co/datasets/uclanlp/ToolSmith-Datasets)
   (splits `FVQA_train`, `Infoseek_GDM_val`, `EncyclopedicVQA_val`, fetched on first use
   like the evaluation splits). Evaluation uses the splits in
   [uclanlp/OpenVLHarness-Evaluation-Datasets](https://huggingface.co/datasets/uclanlp/OpenVLHarness-Evaluation-Datasets).
   Keep evaluation data out of the pool.
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

Gate 1 needs unprivileged Linux user namespaces as well. Without them the pipeline stops,
unless its configuration sets `isolation: false`.

Admitted tools are written to the catalog. `gaps.jsonl` keeps the mined
representatives, `mining.json` the rollouts (with a hash of each judged predictions file
and of the failed queries' trajectories) and the mining settings they came from,
`critiques.json` the self-critiques, and `decisions.jsonl` each gap's outcome with the
gate feedback. A rerun resumes from them; with other rollouts or mining settings it stops
with an error, unless `--rebuild-gaps` asks to mine the rollouts again. A self-critique
that fails leaves its query out, and the next run retries only those. A decision is
redone when the models or gate settings change, or when the tool it admitted, reused or
matched is no longer in the catalog. Serve the grown catalog with the same preset.

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

## License

ToolSmith is part of OpenVLHarness and is released under the same
[MIT License](../LICENSE), including the pipeline, the inference code and the packaged
tool catalog. The training pools on Hugging Face keep the licenses of their source
datasets; third-party models and tools are listed in
[THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md).

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
