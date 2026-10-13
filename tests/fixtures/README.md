# Test fixtures

These files are frozen reference outputs captured from the implementation used
for the paper runs. Tests compare the current code against them and never
regenerate them. All inputs are synthetic: tiny generated PNGs and arrays,
invented URLs and nonfunctional API keys. Downloads, service replies and model
requests were replaced with fixed responses during capture. The `provenance`
entry of each `*old_release.json` file records the source commit and file hashes
in the research codebase, which is not part of this repository.

- `memory_old_release.json`: memory store behaviour for image and URL
  references, lazy downloads, typed value bindings and environment messages.
- `capabilities/`: capability results for fixed service replies, and the nine
  model-facing tool schemas in `public_schemas.json`. `test_capabilities.py`
  allows three differences: the current Text Search description, the Depth
  Estimation `size` descriptor, which now reads `[height, width]`, and OCR
  results, which are now stored as a text binding. The Grounding DINO adapter
  has no reference output.
- `model_protocol/`: request bodies, argument normalization and tool-output
  encoding for the Chat Completions and Responses protocols, and the auxiliary
  model's request arguments.
