"""SAM3 baseline runner for counting and grounding benchmarks.

Standalone package: bypasses the agent loop in `run_benchmark.py` and calls
the SAM3 segmentation server directly, then writes prediction files in the
exact shape the existing dataset evaluators consume.

CLIs:
  python -m baselines.sam3.preview ...   # Stage 1: dump SAM3 prompts (no model calls)
  python -m baselines.sam3.run ...       # Stage 2+3: SAM3 inference + dataset.evaluate
"""
