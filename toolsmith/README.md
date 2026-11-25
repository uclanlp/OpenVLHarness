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

## Contents

- [How ToolSmith works](#how-toolsmith-works)
- [Installation](#installation)
- [Quick start: evaluate with generated tools](#quick-start-evaluate-with-generated-tools)
- [Grow your own catalog](#grow-your-own-catalog)
- [Write your own tool](#write-your-own-tool)
- [Configuration reference](#configuration-reference)
- [Running generated tools safely](#running-generated-tools-safely)
- [The packaged catalog](#the-packaged-catalog)
- [Troubleshooting](#troubleshooting)

## How ToolSmith works

ToolSmith builds a catalog of generated tools in four phases, then uses it at
inference time.

1. **Trajectory mining.** The baseline agent answers a training pool, and an LLM judge
   marks its answers. For each incorrect answer, an LLM self-critique reads the question,
   the answer and the tools the agent tried, and labels the failure `NEW_TOOL_NEEDED`,
   `ARGUMENT_MISTAKE`, `SEARCH_NOISE` or `AMBIGUOUS`. Only `NEW_TOOL_NEEDED` failures
   become capability gaps. The gaps are clustered by sentence embedding, and one
   representative per cluster moves on.
2. **Reuse or grow.** A gap that an existing tool already describes well (TF-IDF
   similarity above a threshold) is covered by that tool and skipped.
3. **Tool generation.** A generator LLM writes a new tool as a Python class, given the
   gap, the expected answer shape, the tools it may call and six example tools.
4. **Verification.** Gate 1 checks the code statically and runs it once on dummy input
   in a confined process. Gate 2 is an LLM critic that scores the tool on parametric
   usage, description abstractness, edge-case handling and task alignment. A rejected
   tool is revised with the gate's feedback. A tool that passes both gates joins the
   catalog unless it nearly duplicates an existing one.

**Per-question retrieval.** At inference, the agent embeds every tool description once.
For each question, it offers the base tools plus the `top_k` generated tools whose
descriptions are closest to the question text.

The paper's settings:

| Setting | Value |
| --- | --- |
| Sentence encoder | `BAAI/bge-m3` |
| Clusters | 500 |
| Reuse threshold (TF-IDF) | 0.40 |
| Retries per candidate | 2 |
| Gate 1 budget | 10 seconds |
| Gate 2 thresholds | mean score at least 3.0, every axis at least 1 |
| Duplicate threshold (TF-IDF) | 0.55 |
| Generated tools per question | 10 |

## Installation

Install OpenVLHarness as in the main [installation](../README.md#installation) section,
with the ToolSmith extra:

```bash
python -m pip install -e '.[server,qwen,eval,toolsmith]'
python -m pip install open_clip_torch  # for the image_similarity and species_id tools
```

`server` runs the tool router, `eval` runs `openvlharness-eval`, and `toolsmith` adds the
sentence encoder and scikit-learn for retrieval and the pipeline. The default encoder,
`BAAI/bge-m3`, ships `pytorch_model.bin` weights, which transformers loads only with
PyTorch 2.6 or newer; the extra asks for it. With an older PyTorch, set `toolsmith.encoder`
(and `encoder` in the pipeline configuration) to a model with safetensors weights. The
evaluator and the pipeline load the encoder on the first visible GPU, or on the CPU without
one; choose the GPU with `CUDA_VISIBLE_DEVICES`.

`open_clip_torch` is only needed by the two tools that score images with CLIP and BioCLIP.
Their models are `timm/vit_base_patch32_clip_224.openai` (CLIP ViT-B/32 with OpenAI's
weights) and `imageomics/bioclip` on Hugging Face. The confined worker cannot write to your
model caches, so download both once from your shell:

```bash
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai'); open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')"
```

Generated tools run in a worker process that the router confines with Linux user
namespaces. Check that your host allows them, as an unprivileged user:

```bash
python -c "from openvlharness.toolsmith.isolation import available; print(available())"
```

`(True, '')` means you are ready. Otherwise the second value says why. Without user
namespaces, the router and the pipeline refuse to start. You can either enable them (see
[Troubleshooting](#troubleshooting)), or set `toolsmith.worker.isolation: false` in the
agent preset and `isolation: false` in the pipeline configuration. Without isolation,
generated tools can read every file and reach every service that your user can, so do
that only inside a container or virtual machine.

## Quick start: evaluate with generated tools

ToolSmith adds two presets for Qwen3-VL-8B:

| Preset | Agent |
| --- | --- |
| [qwen3_8b_toolsmith.yaml](../configs/agent/qwen3_8b_toolsmith.yaml) | ToolSmith: the base tools plus generated tools |
| [qwen3_8b_toolsmith_baseline.yaml](../configs/agent/qwen3_8b_toolsmith_baseline.yaml) | The same agent without generated tools |

Both use the ToolSmith system prompt and the base tools except depth estimation and
camera pose. Without a `toolsmith.catalog` setting, the ToolSmith preset uses the
[packaged catalog](#the-packaged-catalog).

**1. Set the environment.** Prepare the models and tool services, and set the same
variables as in the main [quick start](../README.md#quick-start): `MODEL_BASE_URL`,
`MODEL_API_KEY`, `AUXILIARY_MODEL_BASE_URL`, `AUXILIARY_MODEL_API_KEY`,
`TOOL_SERVER_URL`, `SAM3_SERVER_URL`, `OCR_SERVER_URL`, the judge variables, and the
Serper key (`SERPER_API_KEY` or `SERPER_API_KEY_FILE`) with an image host for reverse
image search. The packaged `media_catalog` tool also reads `TMDB_API_KEY` for films, TV
series and people; without it, those lookups return an error.

**2. Start the router** with the ToolSmith preset:

```bash
python -m openvlharness.services.server --config configs/agent/qwen3_8b_toolsmith.yaml
```

At startup, the router copies the catalog and loads it in a separate worker process.
It logs which credential-like variables the worker does not inherit (see
[Configuration reference](#configuration-reference)).

**3. Evaluate** in another terminal, with `LMUData` set as in the main
[evaluation](../README.md#evaluation) section:

```bash
openvlharness-eval --config configs/agent/qwen3_8b_toolsmith.yaml \
  --data MMSearch FVQA_test_300 --work-dir ./outputs
```

Before it loads anything, the evaluator checks that the router serves the catalog the
configuration names: the digest the router's `/env` reports must match the catalog's
files (`toolsmith.catalog`, or the packaged catalog). When it cannot read those files
(not on its host, a mistyped or relative path, no permission), it stops and names the
path; set `toolsmith.trust_router_catalog: true` to evaluate on a host without the
catalog's files, and it runs on the router's catalog and logs that it does. The
evaluator then takes the generated tools' names, descriptions and input schemas from the
router (`GET /toolsmith/catalog`, which a ToolSmith router serves from its worker, with
the digest of that snapshot), validates them as the catalog loader does, and records that
digest in the run identity. So the evaluator needs no Linux user namespaces of its own,
and a tool admitted while it starts cannot make the run record another catalog than the
one it uses. Only when the router has no such list (it answers 404: a router from an
older release) does the evaluator read the catalog itself, in a local worker confined as
`toolsmith.worker` says, and log that it does.

**4. Find the outputs.** Results go to
`./outputs/Qwen3-VL-8B-Instruct/toolsmithagent/<run_id>/<dataset>/`, laid out as in
[Inspect and resume runs](../README.md#inspect-and-resume-runs). The run manifest records
a hash of the catalog's files and of the version of the rules the loader reads them by
(`toolsmith_catalog_sha256`), so `--resume` continues a run only with the catalog it
started with, loaded as it was then.

**5. See which generated tools a question used.** The tools a question called are in
its trajectory:

```bash
python - outputs/Qwen3-VL-8B-Instruct/toolsmithagent/<run_id>/MMSearch/full_trajectories/<sample_id>.json <<'EOF'
import json, sys
messages = json.load(open(sys.argv[1]))["messages"]
print([call["function"]["name"] for message in messages for call in message.get("tool_calls") or []])
EOF
```

Generated tools have `snake_case` names; base tools end in `_Tool`. (GPT presets use the
Responses API, where calls are `messages` items of type `function_call`.) The tools a
question was offered depend only on its text, so you can list them again:

```python
from openvlharness.bootstrap import build_agent

agent = build_agent("configs/agent/qwen3_8b_toolsmith.yaml")
print(agent.retriever.retrieve("Which architect designed the building in this photo?"))
```

`build_agent` reads the catalog in a short-lived worker and loads the sentence encoder;
it needs the same model variables as the evaluator.

**6. Compare with the baseline.** Restart the router with the baseline preset and
evaluate the same datasets:

```bash
python -m openvlharness.services.server --config configs/agent/qwen3_8b_toolsmith_baseline.yaml
openvlharness-eval --config configs/agent/qwen3_8b_toolsmith_baseline.yaml \
  --data MMSearch FVQA_test_300 --work-dir ./outputs
```

Its results go under `mainagent/` instead of `toolsmithagent/`, next to the ToolSmith run.

## Grow your own catalog

The pipeline reads the judged runs of the baseline agent and writes new tools into a
catalog directory. Run every command from the repository root, because relative paths
in the configurations are read from the current directory.

**1. Evaluate the baseline on a training pool.** The paper's pool is on Hugging Face as
[uclanlp/ToolSmith-Datasets](https://huggingface.co/datasets/uclanlp/ToolSmith-Datasets):
`FVQA_train` (3,312 questions), `Infoseek_GDM_val` (3,000) and `EncyclopedicVQA_val`
(2,862). These splits download into `LMUData` on first use, like the evaluation splits.
Keep evaluation data out of the pool. Start the baseline router as in the quick start,
then evaluate with an LLM judge ([configs/judge.yaml](../configs/judge.yaml), which reads
`JUDGE_MODEL`, `JUDGE_API_BASE` and `JUDGE_API_KEY`):

```bash
openvlharness-eval --config configs/agent/qwen3_8b_toolsmith_baseline.yaml \
  --judge-config configs/judge.yaml \
  --data FVQA_train Infoseek_GDM_val EncyclopedicVQA_val --work-dir ./outputs
```

Each dataset directory of this run, such as
`./outputs/Qwen3-VL-8B-Instruct/mainagent/agent_toolsmith_v1_judge_paper_v1_<time>_default/FVQA_train/`,
now holds judged predictions (`<model>_judge_judge_vdr.xlsx`) and `full_trajectories/`.
The pipeline reads both.

**2. Make a catalog to grow.** Copy the packaged catalog and a preset that points at it:

```bash
cp -r openvlharness/toolsmith/tools my_catalog
cp configs/agent/qwen3_8b_toolsmith.yaml configs/agent/my_toolsmith.yaml
```

In `configs/agent/my_toolsmith.yaml`, add the catalog to the `toolsmith` section the copy
already has, and keep the section's other keys, among them the `worker` grants the
packaged tools need:

```yaml
toolsmith:
  catalog: my_catalog   # the one line to add
  top_k: 10
  encoder: BAAI/bge-m3
  worker:
    environment: [SERPER_API_KEY, SERPER_API_KEY_FILE, TMDB_API_KEY, GOOGLE_APPLICATION_CREDENTIALS,
                  AWS_S3_BUCKET_NAME, AWS_S3_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]
    write_files: true
```

The pipeline requires this setting, so it never grows the packaged catalog by accident.
Start from a copy of the packaged catalog: the generator's examples call some of its tools.

**3. Configure the pipeline.** Copy the paper's pipeline configuration and set `agent` to
your preset:

```bash
cp configs/toolsmith/pipeline.yaml configs/toolsmith/my_pipeline.yaml
```

```yaml
pipeline_config_version: paper_v1     # a label for your records
agent: configs/agent/my_toolsmith.yaml
models:
  generator: &gpt5
    alias: gpt-5-2025-08-07
    base: ${TOOLSMITH_MODEL_BASE_URL}
    key: ${TOOLSMITH_MODEL_API_KEY}
    timeout: 600
  critic: *gpt5
  analyst: *gpt5
encoder: BAAI/bge-m3
clusters: 500
reuse_threshold: 0.40
retries: 2
timeout_seconds: 10
critic_mean: 3.0
critic_min: 1
dedup_threshold: 0.55
workers: 8
max_tokens: 12800
```

The three model roles are:

- `generator` writes and revises the tools;
- `critic` is Gate 2, the LLM code review;
- `analyst` writes the self-critique that labels each failure, and classifies the
  expected answer shape for the generator.

Each role is a model entry like those under `services.models` in the agent presets: an
`alias` from the [model registry](../docs/models.md), the endpoint `base`, the API `key`,
and optional settings such as `timeout` and `retry`. The paper used GPT-5 for all three.

OpenAI shuts down `gpt-5` and `gpt-5-2025-08-07` on 2026-12-11. After that date, set the
alias of the three roles to `gpt-6.1-sol`, the replacement OpenAI recommends, or another
registered model. In the configuration above, the critic and the analyst reuse the
generator's entry, so one line changes all three. Results may then differ from the paper's.

`${...}` values are read from the environment, so set the variables before running:

```bash
export TOOLSMITH_MODEL_BASE_URL=https://api.openai.com/v1
export TOOLSMITH_MODEL_API_KEY=$OPENAI_API_KEY
```

The pipeline calls none of the agent preset's models or tool services, so the variables of
their endpoints and keys (`MODEL_BASE_URL`, `MODEL_API_KEY`, `TOOL_SERVER_URL`,
`SAM3_SERVER_URL` and the like) need not be set for it; any other variable the preset uses
must be, as for the router. The other keys are described in the
[configuration reference](#pipeline-configuration). `max_tokens` (12800, the paper's
budget for every call) can be raised if replies run out of it.

**4. Run the pipeline** on the dataset directories of step 1:

```bash
RUN=./outputs/Qwen3-VL-8B-Instruct/mainagent/<run_id>
python -m openvlharness.toolsmith.pipeline --config configs/toolsmith/my_pipeline.yaml \
  --rollouts $RUN/FVQA_train $RUN/Infoseek_GDM_val $RUN/EncyclopedicVQA_val \
  --work-dir toolsmith_work
```

Pass the dataset directories, not the run directory that holds them. When it finishes,
it prints how many gaps this run processed, or that none was left to decide, and how many
ended in each decision: `admitted` (a new tool joined the catalog), `reused` (an existing tool covers the gap), `rejected` (no candidate
passed the gates), `duplicate` (the new tool nearly matched an existing one) and `failed`
(a model or network call failed; the next run retries the gap). A model reply that ran out
of `max_tokens`, was stopped by the provider's content filter or came back empty counts as
a failed call, never as an answer, and so does a self-critique that names no label. Each
admitted tool is written to the catalog as `<tool_name>.py`. The command exits with status
0 when it is done, status 3 when a self-critique or a gap failed in this run (running it
again retries them), and status 1 when it cannot run at all; see the exit statuses under
[Configuration reference](#configuration-reference).

**5. Read the work directory.** `--work-dir` keeps the pipeline's state:

| File | Contents |
| --- | --- |
| `gaps.jsonl` | The mined cluster representatives, one gap per line |
| `mining.json` | The rollouts (with hashes of their judged predictions and failed trajectories) and the mining settings the gaps came from, the self-critiques that failed (to retry) or failed in three runs (set aside), and how many critiqued queries got each label |
| `critiques.jsonl` | Each self-critique, saved as it completes and reused by every later mining with the same analyst, and each one that failed, with a digest of the settings it failed with (delete it to have the analyst asked again) |
| `decisions.jsonl` | One line per decided gap: the decision, the tool, and the gates' feedback on each attempt; and one per failed attempt at a gap, with its error and a digest of the settings it failed with |

Rerunning the same command resumes. It reuses `gaps.jsonl` and skips gaps that already
have a decision. A run that stopped while mining keeps the self-critiques it finished, so
the next run asks the analyst only about the rest. If some self-critiques failed, the
next run asks the analyst again for those questions only, then clusters all the gaps
again: with the retried questions among them, a few clusters can get another
representative, and that run decides those new gaps too. A gap or a self-critique that
failed in three runs with the same settings is set aside after that: pass `--retry-failed`
to try it again, or change a setting its failures may depend on (the models' settings, their
endpoints and timeouts included, the gate settings or `max_tokens`), which retries it as
well. The work files keep only a digest of those settings. Credentials are not in the
digest (the API key, and a user name, password or `key`/`token`-like query parameter in an
endpoint URL), so that a work directory cannot confirm a guessed one; after changing only a
credential, pass `--retry-failed`. A gap
is decided again when its models or gate settings change, or when the tool it admitted,
reused or matched is no longer in the catalog. If the rollouts
or the mining settings (`encoder`, `clusters`, the analyst) change, the pipeline stops
with an error that says what changed. Pass `--rebuild-gaps` to mine the rollouts again
(it still reuses the saved self-critiques of failures that did not change), or use a new
`--work-dir`.

**6. Evaluate with the grown catalog.** The router loads the catalog when it starts, so
restart it with your preset, then evaluate:

```bash
python -m openvlharness.services.server --config configs/agent/my_toolsmith.yaml
openvlharness-eval --config configs/agent/my_toolsmith.yaml --data MMSearch --work-dir ./outputs
```

A grown catalog has a new hash, so these results start a new run rather than resuming
one made with the old catalog. If the router still runs the old catalog, the evaluator
stops before it loads anything and asks you to restart the router.

## Write your own tool

A tool is a Python module in the catalog directory with exactly one subclass of
`GeneratedTool`. Files whose names start with `_` are skipped.

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

- `__init__` passes three keyword arguments. `tool_name` is the name the model sees; it
  must be unique in the catalog. `tool_description` tells the model, and the retriever,
  when to use the tool. `input_schema` is a JSON Schema for the arguments whose root is
  `"type": "object"`: the router passes the arguments to `execute` as named arguments, so
  the catalog loader and Gate 1 refuse any other root.
- `execute(**arguments)` returns a string, a mapping or a list. Return an `error` field
  rather than raising for expected failures; an exception becomes a tool error that the
  model sees.
- To take an image, declare an `image_id` string argument. `execute` receives the image as
  `image_base64`, a base64 data URL.
- `self.call_tool(name, **arguments)` calls another catalog tool, or one of
  `Text_Search_Tool`, `Image_Search_Tool`, `Webpage_Visit_Tool`, `OCR_Tool` and
  `Zoom_In_Tool`, through the router. It returns that tool's result and raises
  `ToolCallError` when the tool fails. The router refuses calls to its other tools.
- A tool reaches the web through the router's outbound proxy: use a client that reads
  `HTTP_PROXY` and `HTTPS_PROXY`, such as `requests`, `httpx` or `urllib.request` (aiohttp
  only with `trust_env=True`), on ports 80 and 443 by default (`worker.network_ports`). In
  a confined worker, raw sockets and names a tool resolves itself reach nothing, and the
  proxy refuses other ports and every service on the router's host or its local networks (see
  [Running generated tools safely](#running-generated-tools-safely)).
- The schema must be a valid Draft 7 JSON Schema without the keywords that can make
  argument validation fetch documents or take exponential time: references (`$ref`, `$id`,
  `$schema`, `$anchor`) and regular expressions (`pattern`, `patternProperties`; check
  formats in `execute` instead). Only keywords count: an argument may be named `pattern`.
  A tool with any other schema, or one the check itself fails on, is skipped with a
  warning.
- Do not name arguments after the router's runtime settings (`server_url`,
  `timeout_seconds`, `llm_model`, `llm_base_url`, `llm_api_key`, `box_threshold`,
  `text_threshold`, `num_results`, `gl`, `hl`, `image_path`, `image_paths`). The router
  rejects calls that pass them.

Check a catalog before serving it. `strict=True` raises on the first problem instead of
skipping the tool:

```bash
python -c "from openvlharness.toolsmith.catalog import load_catalog; print(sorted(load_catalog('my_catalog', strict=True)))"
```

This imports the catalog in your shell's process, so check only code you trust this way.
Then call the tool through a running router:

```bash
curl -s http://127.0.0.1:8101/execute -H 'Content-Type: application/json' \
  -d '{"tool_name": "entity_property_lookup", "tool_call_id": "check-1", "args": {"entity": "Eiffel Tower", "property": "architect"}}'
```

The pipeline gives its generator the six [examples](../openvlharness/toolsmith/pipeline/examples)
and the [prompts](../openvlharness/toolsmith/pipeline/prompts) of every step; they show
the style that Gate 1 and Gate 2 expect. Tools written by the pipeline may import only
`re`, `json` and `datetime`. Tools you write yourself, like the packaged ones, may import
any installed package.

## Configuration reference

### Agent preset

A `toolsmith` section in an agent preset makes the agent a ToolSmith agent and makes the
router serve the catalog. All keys are optional:

```yaml
toolsmith:
  catalog: my_catalog          # default: the packaged catalog
  top_k: 10                    # generated tools offered per question
  encoder: BAAI/bge-m3         # sentence-transformers model for retrieval
  trust_router_catalog: false  # true: an evaluator without the catalog's files runs on the router's
  worker:
    environment: [TMDB_API_KEY]  # credential-like variables the tools may read; default []
    timeout_seconds: 540       # per call; default 90% of services.client.timeout_seconds
    memory_gib: 128            # memory of all the worker's processes together, in GiB (Linux)
    write_files: false         # true lets tools write files
    cache_layer_gib: 1         # with write_files: room for writes into the model caches, in GiB
    isolation: true            # confine the worker with Linux namespaces
    network: true              # HTTP and HTTPS through the router's proxy; false: no network
    network_ports: [80, 443]   # destination ports the proxy serves
    deny_networks: []          # networks the proxy refuses too, such as 192.0.2.0/24
```

The worker inherits the router's environment except variables whose names contain `KEY`,
`TOKEN`, `SECRET`, `PASSWORD` or `CREDENTIAL`, or start with `AWS_` or
`GOOGLE_APPLICATION`. List the ones your tools need under `worker.environment`. The
ToolSmith preset grants the Serper, TMDB and image-upload credentials, and sets
`write_files: true` because the image upload, model downloads and PyTorch write files.
The rows for each key in the [runtime reference](../docs/runtime.md) give their types and
details.

<a id="pipeline-configuration"></a>
### Pipeline

| Key | Default | Meaning |
| --- | --- | --- |
| `agent` | required | Agent preset whose `toolsmith.catalog` the pipeline grows |
| `models` | required | `generator`, `critic` and `analyst` model entries |
| `pipeline_config_version` | none | A label for your records; the pipeline does not read it |
| `encoder` | `BAAI/bge-m3` | Sentence-transformers model that embeds gaps for clustering |
| `clusters` | `500` | Number of clusters; with fewer gaps, every gap is kept |
| `reuse_threshold` | `0.40` | TF-IDF similarity above which an existing tool covers a gap |
| `retries` | `2` | Revisions after a gate rejects a candidate |
| `timeout_seconds` | `10` | Gate 1's time budget for one run of a candidate |
| `critic_mean` | `3.0` | Gate 2's minimum mean score over the four axes (0 to 5) |
| `critic_min` | `1` | Gate 2's minimum score on every axis |
| `dedup_threshold` | `0.55` | TF-IDF similarity of descriptions at or above which a candidate is a duplicate |
| `workers` | `8` | Gaps processed at once |
| `max_tokens` | `12800` | Output budget of every model call, the paper's for each role; for reasoning models it includes the reasoning tokens. Raise it if replies run out |
| `isolation` | `true` | Run Gate 1 and the pipeline's catalog loads in Linux namespaces |

Command-line options of `python -m openvlharness.toolsmith.pipeline`:

| Option | Meaning |
| --- | --- |
| `--config` | The pipeline configuration |
| `--rollouts` | One or more dataset directories of a judged baseline run |
| `--work-dir` | Directory for the pipeline's state |
| `--rebuild-gaps` | Mine the rollouts again instead of reusing `gaps.jsonl` |
| `--retry-failed` | Try again the gaps and self-critiques set aside after failing in three runs with the same settings |

Exit statuses of the command:

| Status | Meaning |
| --- | --- |
| `0` | Done. The summary lists the gaps and self-critiques set aside after failing in three runs; `--retry-failed` retries them |
| `1` | The pipeline cannot run; the error says why |
| `2` | The command line is wrong |
| `3` | A self-critique or a gap failed in this run (a model or network call failed). The summary comes first. Running the same command again retries them, until each has failed in three runs |

## Running generated tools safely

Generated tools are code written by a language model. Treat a catalog you did not write
or review as untrusted.

**What the router's worker confines.** The router never imports catalog code. It runs a
copy of the catalog in a separate worker process. The pipeline reads tool descriptions
in a worker of its own, and so does the agent when its router lists no catalog (it
normally takes them from the router); confined, that worker has no network. The
pipeline's worker is confined by the pipeline's own `isolation` setting instead of
`worker.isolation`. The worker:

- inherits the router's environment without credential-like variables, unless
  `worker.environment` grants them;
- runs each call under a time limit. A call that raises returns a tool error and the
  worker carries on; only a call past the limit, a crash, a stop for using too much
  memory or CPU (see **Memory and CPU** below) or a message that breaks the worker
  protocol (one over the size limit, or one the router cannot read) makes the router start
  a new worker for later calls;
- reaches only the other catalog tools and five base tools through `call_tool`:
  `Text_Search_Tool`, `Image_Search_Tool`, `Webpage_Visit_Tool`, `OCR_Tool` and
  `Zoom_In_Tool`;
- cannot write files unless `write_files` is set, and then no file may grow past
  `worker.memory_gib`. With `isolation`, even then the host's model caches stay
  read-only: writes to them go to a private in-memory layer of `worker.cache_layer_gib`
  (1 GiB by default), discarded with the worker;
- prints to the router's stderr without control characters, up to 8 MiB from each worker
  and 64 MiB from all the workers of a router; the router's log says when it drops the
  rest;
- with `isolation` (the default), runs in Linux user, mount, PID, IPC and network
  namespaces under your user, without root. It sees, read-only, the interpreter, every
  directory on the router's `sys.path` and `PYTHONPATH` (the Python packages, and whatever
  else those directories hold), the catalog copy, the model caches, the files that granted
  variables name (such as a `*_FILE` key file or service-account credentials), certificate
  bundles, a few settings from `/etc`, and the host's `/sys`. With `write_files` and the
  local image host (`IMAGE_UPLOAD_PROVIDER=local`) it can also write the upload directory,
  which that host serves. It has private temporary directories of at most 1 GiB each, and
  sees no other host files and no other processes. It cannot create nested user
  namespaces; on Linux 5.14 and later it runs at most 4,096 processes and threads, and on
  Linux 5.19 and later it has no SysV shared memory, message queues or semaphores.

**Network.** A confined worker has a network namespace of its own with only loopback: no
interface, port or abstract UNIX socket of the host reaches it, and `/proc/net` lists only
its own connections (`/sys/class/net` still lists the host's interfaces, read-only). With
`worker.network: true` (the default), its HTTP and HTTPS clients go through an outbound
proxy in the router: `HTTP_PROXY` and `HTTPS_PROXY` point at a relay on the worker's
loopback, and `NO_PROXY` keeps `localhost`, `127.0.0.1` and `::1` inside the sandbox. The
proxy serves `http://` requests and HTTPS tunnels (CONNECT) to the ports in
`worker.network_ports`, 80 and 443 by default, one request per connection. It resolves
each name on the router's host and refuses, with the same answer for all, a name that does
not resolve and one that resolves to a loopback, private, link-local, shared, multicast or
reserved address (or an IPv6 address that embeds one), to an address of the host, or to
an address in a network the host reaches without a gateway (its LANs, even with public
addresses) or in `worker.deny_networks`. It then connects to an address it checked. This
keeps tools from the services on this host that the proxy can tell are its own, such as the
router with its local Python executor (which is not a sandbox), other routers or model
servers, and from those on the host's LANs; public services elsewhere stay reachable. The
proxy does not recognize a public address that leads back to the host or its network (1:1
NAT or hairpin NAT), networks that only a policy routing table reaches, a site's own NAT64
prefix, or the public addresses at which cloud platforms serve their machines, except
Azure's 168.63.129.16, which it refuses. List those in `worker.deny_networks`, and on a
campus or corporate network whose hosts have public addresses its networks, such as
`192.0.2.0/24` and `2001:db8::/32`. The routes of a VPN interface count as networks without
a gateway: with a split tunnel that sends public ranges through such a route, the proxy
refuses public hosts in them, unless the route covers more than a /8 (IPv4) or /16 (IPv6).

The proxy reads request heads strictly (CRLF line ends, no other control characters, no
folded lines, a body only with `Content-Length`) and gives each head 30 seconds and
64 KiB; a connection that moves nothing for two minutes is closed within four. All the
generated tools of a router share 256 open connections: past them, new ones get 503, so a
tool that holds many open keeps the others off the network meanwhile. A worker process
opens at most 4,096 files and sockets. If the router has its own `HTTP_PROXY` or
`HTTPS_PROXY` (or, without them, `ALL_PROXY`), the outbound proxy reaches each checked
address through a CONNECT tunnel from that proxy. It must be an `http://` proxy (not
`https://` or SOCKS) that allows CONNECT to the ports in `worker.network_ports` (port 80
too, for `http://` requests) and to IP addresses: it is asked for the checked address,
never for a name, so a proxy that allows destinations by name refuses the tunnels. Names
are still resolved on the router's host, so where only that proxy can resolve public
names, generated tools get no network. The worker gets none of the router's proxy
variables. Tools that open raw sockets, resolve names themselves or use clients that
ignore `HTTP_PROXY` and `HTTPS_PROXY` (aiohttp without `trust_env`, a bare urllib3
`PoolManager`) get no network; the packaged tools use `requests` and Hugging Face
libraries, which use the proxy.

`worker.network: false` leaves the worker no network at all; `call_tool` still works. A
confined worker without network also gets `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`, so models already in the cache load at once instead of after
the Hub requests time out. The same holds when the tools' metadata is read
(`catalog_snapshot`): with isolation the catalog loads with no network and Hugging Face
offline, so a module that downloads a model it does not find in the cache at import fails
there. Download such models on the host first.

**Memory and CPU.** On Linux the router measures each worker's processes every 0.5
seconds: their proportional set size (memory that processes share is split among them),
plus what the sandbox's own memory-backed file systems hold (its `/tmp`, home directory,
`/dev/shm` and the cache layer). When the total passes `worker.memory_gib`, the router
stops the worker: its calls in flight fail with a `MemoryLimitExceeded` error, and the
next call starts a new worker. A file in those file systems that a tool also maps counts
once, and so does the memory a child shares with its parent after `vfork` or
`posix_spawn`, until it execs. Where `smaps_rollup` is missing (Linux before 4.14), the
router sums resident set sizes instead, which count memory that processes share after
`fork` once in each of them. Each process's address space is capped at
`worker.memory_gib` as well. Between calls, a worker whose processes use more than 2
CPU-seconds in 10 seconds with no call in flight, as a thread or process that a tool left
running does, is stopped the same way (`CPULimitExceeded`); calls that the worker's own
tools make through `call_tool` do not count as calls in flight. A call's CPU time is
bounded only by `worker.timeout_seconds`, so one call can use every core of the host for
that long. Measuring a worker with thousands of processes takes long (about a second for
2,000), and the next measurement then waits ten times as long, about a tenth of a core,
but at most 5 seconds. So for very large process trees the interval between measurements
can reach 5 seconds plus the measurement itself, and the router spends a larger share of a
core on them.

**Gate 1** checks the source for dynamic execution and file access, then runs the
candidate once with no file writes and the `timeout_seconds` budget. Each of its processes
has 2 GiB of address space, and the run is stopped when its processes together hold more
than 2 GiB, measured as for the worker. With the pipeline's `isolation` (the default), it
runs in the same kind of namespaces, with no network at all. The static check only narrows
what a candidate can express; the namespaces are what keep it from the host.

**What the worker does not confine:**

- **What a tool sends out.** With `worker.network: true`, a tool can send what it reads
  (its arguments, images and the credentials it is granted) to any public address on the
  allowed ports. Set `worker.network: false` when the tools do not need the web.
- **Code written to evade the limits.** The memory and CPU rules stop runaway tools: a
  memory blow-up, work left running after a call, a flood of output. Between two
  measurements, a spike can reach the address-space cap in each of the worker's processes.
  Among the memory that is not counted are shared memory (memfd or shared anonymous) whose
  pages no process currently has mapped in (with `write_files`, each memfd file is capped at
  `worker.memory_gib`), SysV shared memory that a tool holds detached before Linux 5.19,
  what the kernel holds for a tool, such as data queued in its sockets and pipes, and files
  it writes to the host's upload directory (with `write_files` and the local image host).
  Where `kcmp` is refused, a forked child less than a second old, whose parent has a thread
  waiting on a `vfork`, can pass for a vforked one: it is not counted, and lowers its
  parent's share of the pages they share. A process whose main thread has exited and whose
  other threads never live long enough to be read counts its last known memory, or nothing
  when it was never read. Children whose parent ignores SIGCHLD take their CPU time with
  them. This list is not complete.
- **Other secrets.** The environment filter goes by name. Secrets under other names, such
  as `DATABASE_URL` or a password in a proxy URL, reach the worker; keep them out of the
  router's environment.
- **Without isolation**, none of the file, process and network confinement applies. A
  tool reaches every service the host can, the router's HTTP API among them (calls that
  arrive through it keep the worker from ever looking idle), and processes that detach
  from the worker escape the memory count and outlive it. The workers that read tool
  descriptions and Gate 1 share the host's network too.

For catalogs you do not trust, run the router and the pipeline in a container or a
virtual machine that holds only the files, credentials and network access they need,
and set hard limits on its memory and CPU there.

## The packaged catalog

The packaged catalog in [openvlharness/toolsmith/tools](../openvlharness/toolsmith/tools)
holds tools written in the style of generated tools. Several are building blocks that the
others call. It is a small set of example tools to try ToolSmith with; the paper's setup
is a catalog grown with the pipeline on the training pools (see
[Grow your own catalog](#grow-your-own-catalog)).

| Tool | What it does |
| --- | --- |
| `aircraft_type_by_image` | Names the aircraft type in an image from its reverse image search matches |
| `animal_typical_size_by_image` | Identifies an animal in an image, then reads its typical adult size from its encyclopedia page |
| `bird_named_after_from_image` | Identifies a bird in an image, then reports whom or what its name honors, from Wikidata |
| `event_at_location_from_image` | Identifies a place in an image, then reports a significant event there, from Wikidata |
| `fungi_substrate_from_image` | Identifies a fungus in an image, then reports what it grows on, from its encyclopedia page |
| `gazetteer_lookup` | Geocodes a place name with OpenStreetMap Nominatim |
| `image_manufacturer_lookup` | Identifies a product, vehicle or aircraft model in an image, then reports its manufacturer, from Wikidata |
| `image_scenic_location_identifier` | Identifies the city or place in a scenic image, resolved with the gazetteer |
| `image_similarity` | Ranks candidate text labels for an image with CLIP |
| `landmark_name_from_image` | Names the landmark or structure in an image |
| `landmark_operator_from_image` | Identifies a landmark in an image, then reports who operates it, from Wikidata |
| `media_catalog` | Looks up music in MusicBrainz, and films, TV series and people in TMDB (`TMDB_API_KEY`) |
| `park_latest_renovation_date_from_image` | Identifies a park in an image, then reports its latest renovation from its encyclopedia page |
| `place_elevation_from_image` | Identifies a place or mountain in an image, then reports its elevation, from Wikidata |
| `place_naming_origin_from_image` | Identifies a named place in an image, then reports where its name comes from, from Wikidata |
| `plant_original_genus_from_image` | Identifies a plant in an image, then reports the genus it was first described in |
| `species_id` | Scores candidate species for an organism photo with BioCLIP |
| `unit_compute` | Converts units, does arithmetic and computes date differences, offline |
| `vehicle_brand_by_image` | Identifies a vehicle in an image, then reports its brand, from Wikidata |
| `visual_matches` | Returns the Google Lens matches of an image (Serper and an image host) |
| `wikidata_query` | Looks up an entity's properties in Wikidata |

The image tools identify their subject with `visual_matches`, so they need the Serper key
and an image host, like the base `Image_Search_Tool`. `image_similarity` and `species_id`
need `open_clip_torch` and their models in the cache (see [Installation](#installation)).
`gazetteer_lookup` and `media_catalog` send Nominatim and MusicBrainz at most one request
a second from each worker process, as their terms ask. The limit is per process: several
routers, or a worker that is still finishing calls next to the one that replaced it, each
send up to that rate, so together they can exceed it. The licenses and terms of the
models and services these tools use are listed in
[THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md). `media_catalog` uses the TMDB API
but is not endorsed or certified by TMDB.

## Troubleshooting

**"the tool router at ... serves ToolSmith catalog ..., but this run's catalog is ..."**
The router loaded a different catalog than your configuration names, or the catalog
changed after the router started (a pipeline admitted a tool, say). Restart the router
with the same preset. Both
processes read a relative catalog path from their own working directory, so start them
from the same directory. "serves no ToolSmith catalog" means the router was started with
a preset that has no `toolsmith` section, such as the baseline. "cannot read which
ToolSmith catalog" means the evaluator could not reach the router at `TOOL_SERVER_URL`.

**"the ToolSmith catalog '...' is not readable here"** The evaluator cannot read the
catalog `toolsmith.catalog` names, so it cannot check that the router serves it. Fix the
path: it is read from the evaluator's working directory, so a relative path must hold
from there. On a host that has no copy of the catalog, set
`toolsmith.trust_router_catalog: true` to evaluate on whatever catalog the router serves;
the run records that catalog's digest.

**"Generated tools run confined by Linux user namespaces, which this host does not allow"**
Run the [check](#installation) from the installation section to see the reason.
On some distributions an administrator can enable unprivileged user namespaces with a
sysctl, such as `kernel.unprivileged_userns_clone=1` on Debian. Some container runtimes
block them; inside such a container, set `worker.isolation: false` and `isolation: false`
and rely on the container. Running as root is refused, because a root process would keep
its capabilities inside the namespace.

**Ubuntu 24.04** (and 23.10) restricts unprivileged user namespaces with AppArmor
(`kernel.apparmor_restrict_unprivileged_userns=1`): a program without an AppArmor profile
that allows them gets a namespace without the permissions the confinement needs, so the
check fails. An administrator can choose one of:

- **Allow them for one interpreter (safer).** Print the path of the Python that runs the
  router and the pipeline with
  `python -c "import os, sys; print(os.path.realpath(sys.executable))"`, write a profile
  for it to `/etc/apparmor.d/openvlharness-python`:

  ```
  abi <abi/4.0>,
  include <tunables/global>

  profile openvlharness-python /path/to/bin/python3.10 flags=(unconfined) {
    userns,
  }
  ```

  and load it with `sudo apparmor_parser -r /etc/apparmor.d/openvlharness-python`. Every
  program that this interpreter runs may then create user namespaces.
- **Lift the restriction for the whole host** with
  `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0` (add the setting to a
  file in `/etc/sysctl.d/` to keep it after a reboot).
- **Use a container or a virtual machine** for the router and the pipeline, set
  `worker.isolation: false` and `isolation: false`, and rely on it.

**A tool gets no network: "Temporary failure in name resolution", "Network is unreachable"
or "Connection refused".** In a confined worker, only HTTP and HTTPS clients that read
`HTTP_PROXY` and `HTTPS_PROXY` reach the network, through the router's outbound proxy, and
only with `worker.network: true`. Use `requests`, `httpx` or `urllib.request` in the tool,
or aiohttp with `trust_env=True`.

**"Tunnel connection failed: 403 Forbidden", or a 403 answer from the proxy.** The outbound
proxy refused the destination. Its answer says why: "port N is not allowed
(toolsmith.worker.network_ports: ...)" for a port that `worker.network_ports` does not
list, or "HOST does not resolve to an address the proxy may reach: ..." for a name that
does not resolve on the router's host, or that resolves to a private, loopback or
link-local address, an address of the host, one on its local networks (a range that a VPN
interface's route covers counts as one) or one in `worker.deny_networks`. These names all get the same answer, so that a tool cannot learn
which names your network knows; with the `openvlharness.toolsmith.proxy` logger at DEBUG,
the router logs which it was. HTTPS clients show only the status line.

**"Tunnel connection failed: 502 Bad Gateway".** The outbound proxy could not reach the
destination ("cannot connect to HOST:PORT"), or the router's own `HTTP_PROXY`,
`HTTPS_PROXY` or `ALL_PROXY` refused the tunnel to it ("the router's proxy refused a tunnel
to HOST:PORT (403)") or is not an `http://` proxy. Many proxies, Squid among them by
default, allow CONNECT only to port 443, so `http://` requests fail through them: allow
CONNECT to port 80, and to IP addresses, there. A 503 means that 256 connections of the
router's generated tools are already open, or that the router cannot read the host's
networks.

**`MemoryLimitExceeded`: "its processes and in-memory files used ... of memory together".**
The worker's processes, with what its `/tmp`, home directory, `/dev/shm` and cache layer
hold, used more than `worker.memory_gib`. The router stopped the worker, its calls in
flight failed, and the next call starts a new worker. Raise `worker.memory_gib`, or have
the tool hold less, for example by deleting its temporary files when it is done with them.

**The router logs "a tool left work running after its call returned".** A thread or process
that a tool started kept using CPU with no call in flight (more than 2 CPU-seconds in 10
seconds), and the router stopped the worker (`CPULimitExceeded`); the next call starts a
new one. Stop background threads and child processes before `execute` returns.

**A tool fails for lack of a credential.** At startup, the router logs "Generated tools run
without these router environment variables: ...". Add the variables your tools read to
`toolsmith.worker.environment` and restart the router. Reverse image search also needs
an image host; `visual_matches` reports, for example, that `IMAGE_UPLOAD_PUBLIC_BASE_URL`
is not set. See the [search and storage setup](../openvlharness/services/tools/serper_text_search/README.md).

**"toolsmith.catalog '...' is not a directory"** The path does not exist from the
directory where you started the command. Use an absolute path, or start the router, the
evaluator and the pipeline from the repository root.

**`--mode eval` does not find a run.** With a ToolSmith preset as `--config`, evaluation-only
mode picks only runs recorded with the same catalog hash as the configured catalog. If you
grew the catalog since the run, or the catalog path names an environment variable that is
not set, select the run directly with `--run-dir`:

```bash
openvlharness-eval --mode eval --run-dir ./outputs/Qwen3-VL-8B-Instruct/toolsmithagent/<run_id> --data MMSearch
```

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
