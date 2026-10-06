# Reproducible release checks

These checks exercise the released software and the fixed perception subset used
for integration validation. They do not reproduce the paper's full benchmark
results. Run commands from the repository root after installing the package and
the relevant evaluator dependencies.

| Evidence | Reproduction / checked-in inputs |
| --- | --- |
| Complete regression suite | `LMUData="$LMUData" python -m pytest tests -q`; set `LMUData` to a writable dataset directory before collection |
| Runtime, protocol, memory and capability regressions | `python -m pytest tests/test_agent.py tests/test_models.py tests/test_memory.py tests/test_capabilities.py tests/test_services.py tests/test_bootstrap.py -q` |
| Killable Python execution and HTTP retries | `python -m pytest tests/test_python_executor.py tests/test_review_regressions.py tests/test_http.py -q` |
| Raw trace reconstruction and export failures | `python -m pytest tests/test_raw_trace.py tests/evaluation/test_raw_trace_reporting.py -q` |
| Saved-run evaluation, resume identity and scoring | `python -m pytest tests/evaluation -q` with `LMUData` set |
| Stable provider/capability fixtures | [`tests/fixtures`](../tests/fixtures/README.md); no live service calls |
| Fixed ODinW/FSC147 selection | [`perception_samples.json`](perception_samples.json): 13 positive ODinW images and 12 FSC147 images, with IDs and image/question hashes |
| Reviewed commit results | [`review_summary.json`](review_summary.json): test/replay counts, live metrics, retained failures and SHA256 identities of external reports |
| Reviewed results and limits | [`docs/validation.md`](../docs/validation.md), including the tested source revision and live comparison scope |

Prepare the dataset TSVs and image files under a local directory, then check that
they match the frozen selection without initializing any inference clients:

```bash
python validation/run_perception.py --data-root "$LMUData" --check-only
```

Configure the model and tool services using [model setup](../docs/models.md) and
[tool setup](../src/openvlharness/services/tools/README.md). Run the fixed set with
one worker and direct metrics, using a fresh output directory for each backbone:

```bash
python validation/run_perception.py --data-root "$LMUData" \
  --config configs/agent/qwen3_paper.yaml --output-dir "$QWEN_VALIDATION_DIR"
python validation/run_perception.py --data-root "$LMUData" \
  --config configs/agent/gpt6_luna_paper.yaml --output-dir "$GPT_VALIDATION_DIR"
```

`LMUData` is your prepared dataset root; the two validation-directory variables
must name empty/new directories. ODinW annotations are read from `GLIP_ROOT`
(defaulting to `$LMUData/GLIP` in this runner). The runner verifies every selected
image and question, saves the model/configuration identity, uses constant-score
grounding evaluation without an LLM judge, and writes predictions, full
trajectories, images, detection sidecars and HTML. Live predictions can vary.

Evaluation can subsequently run with the model and tools offline:

```bash
python run_benchmark.py --mode eval --run-dir "$QWEN_VALIDATION_DIR" \
  --data ODinW13_bbox_test_qwen3 FSC147_test
```

Raw live trajectories and provider captures are retained outside Git. The checked-in
selection, fixtures, commands and compact results identify the validation scope
without distributing full traces, credentials or machine-specific launch paths.

The five supplemental chains exercise grounding/zoom, depth/coding, OCR/coding,
ordered camera/coding and reverse-image/text/webpage search. Their archived images
and full service captures are external evidence; the portable runner above covers
the 25 public-dataset examples. It does not claim to reproduce those supplemental
chains or the paper's complete benchmark suite.
