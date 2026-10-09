# Test fixtures

These files are frozen reference outputs captured from the implementation used
for the paper runs. Tests compare the current code against them and never
regenerate them. All inputs are synthetic: tiny generated PNGs and arrays,
invented URLs and nonfunctional API keys. Downloads, service replies and model
requests were replaced with fixed responses during capture.

- `memory_old_release.json`: memory store behaviour for image and URL
  references, lazy downloads, typed value bindings and environment messages.
- `capabilities/`: capability results for fixed service replies, and the nine
  model-facing tool schemas in `public_schemas.json` (the current Text Search
  description differs; `test_capabilities.py` allows only that change).
- `model_protocol/`: request bodies, argument normalization and tool-output
  encoding for the Chat Completions and Responses protocols, and the auxiliary
  model's request arguments.
