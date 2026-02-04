# Regular Inspection Notes

Inspection procedure (brief): Review all non-`mini_vlm_toolkit/tool_infra/` code for evaluation correctness risks and tool-usage agent design gaps; note concrete issues with file refs and list any open questions for maintainers. When updating this doc, replace the top block's Date with the run date and add new items under the existing sections; move resolved items into a "Resolved (from <date>)" block at the bottom labeled with the original top-block date.

---

Date: Monday, January 26, 2026
Scope: Evaluation codebase (everything outside `mini_vlm_toolkit/tool_infra/`)

## Findings — Evaluation Correctness
1) Video eval results can be stale or wrong because `infer_video_job` writes prediction files without encoding fps/nframe/pack settings. Re-running with different sampling settings in the same `work_dir` silently reuses older outputs. This is a correctness blocker for video benchmarks.
   - Files: `mini_vlm_toolkit/inference_video.py`
2) Video inference skips when a result file exists, without completeness checks. If a prior run produced a partial file (or a file from a different sampling config), inference is skipped and evaluation uses incomplete/wrong data.
   - Files: `mini_vlm_toolkit/inference_video.py`
3) Global `EVAL_FORMAT` is mutated in MathCanvas eval without restoration. In multi-dataset runs, later evaluations may emit JSON unexpectedly and break eval loading/aggregation.
   - Files: `mini_vlm_toolkit/dataset/image_vqa.py`
4) `SPLIT_THINK` env parsing is truthy-string buggy: any non-empty string (e.g., "0" or "false") triggers thinking split, potentially truncating predictions and changing scores.
   - Files: `mini_vlm_toolkit/inference_image.py`
5) `dump(...)` falls back to `.pkl` on write failure but the pipeline still uses the original path for evaluation. That can desync eval (stale or missing file) with no hard failure.
   - Files: `mini_vlm_toolkit/smp/file.py`
6) `PRED_FORMAT=json` is allowed, but many evaluators assume a DataFrame and will break when `load(...)` returns dict/list for JSON outputs.
   - Files: `mini_vlm_toolkit/smp/file.py` and dataset evaluators

## Findings — Tool-Usage Agent Training/Inference Design Risks
1) No tool-call support in `BaseAgent`: it ignores tool schema, tool calls, and tool messages. Tool-using training/inference cannot work without new agent implementation.
   - Files: `mini_vlm_toolkit/agent/base.py`
2) Prompt formatting only handles `text`/`image` inputs; tool messages and other modalities are not represented.
   - Files: `mini_vlm_toolkit/agent/base.py`
3) Agent registry only exposes `BaseAgent`, so tool-aware agents cannot be built from config.
   - Files: `mini_vlm_toolkit/agent/__init__.py`
4) Agent config silently drops unknown kwargs, so tool-related config fields are ignored with no warning.
   - Files: `mini_vlm_toolkit/agent/config.py`
5) Inference loop assumes single-shot `generate(...)` with no tool-calling loop; tool execution isn’t supported.
   - Files: `mini_vlm_toolkit/inference_image.py`, `mini_vlm_toolkit/inference_video.py`

## Questions to Raise With Maintainers
1) Are video datasets ever run with multiple fps/nframe configurations in the same `work_dir`? If yes, the filename collision is a correctness bug (not just a footgun).
2) Is `PRED_FORMAT=json` intended to be fully supported for evaluation? If yes, evaluator loading paths need harmonization (DataFrame vs dict/list).
3) For tool usage: do maintainers want a new tool-aware agent subclass and inference loop, or should `BaseAgent` be retrofitted?
