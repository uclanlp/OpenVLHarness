# Regular Inspection Notes

Inspection procedure (brief): Review all non-`mini_vlm_toolkit/tool_infra/` code for evaluation correctness risks and tool-usage agent design gaps; note concrete issues with file refs and list any open questions for maintainers. When updating this doc, replace the top block's Date with the run date and add new items under the existing sections; move resolved items into a "Resolved (from <date>)" block at the bottom labeled with the original top-block date.

---

Date: Monday, February 9, 2026
Scope: Evaluation codebase (everything outside `mini_vlm_toolkit/tool_infra/`); scanned commits: `269eea2`, `4d24a07`, `7ac95d9`, `577b4c4`, `c1fdbdc`, `fe4c7c8`, `bf50703`
Validation note: Runtime agent tests were partially blocked in this environment (`numpy` missing), so this update is based on static code inspection plus existing unit-test coverage in repo.

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
1) `/execute` non-JSON `200 OK` responses are treated as success (`ok=True`) with raw text results. Backends that encode failures in plain text while returning HTTP 200 can be misclassified as successful tool calls.
   - Files: `mini_vlm_toolkit/utils/api_utils.py`

## Questions to Raise With Maintainers
1) Are video datasets ever run with multiple fps/nframe configurations in the same `work_dir`? If yes, the filename collision is a correctness bug (not just a footgun).
2) Is `PRED_FORMAT=json` intended to be fully supported for evaluation? If yes, evaluator loading paths need harmonization (DataFrame vs dict/list).
3) Should non-JSON `200 OK` tool-execute responses default to `ok=False` unless explicitly marked successful by envelope?

## Resolved (from Monday, January 26, 2026)
1) No tool-call support in `BaseAgent`: a tool-aware implementation now exists (`BaseAgentWithTools`) and is exposed in agent config/build flow.
   - Files: `mini_vlm_toolkit/agent/base.py`, `mini_vlm_toolkit/agent/__init__.py`
2) Agent registry only exposed `BaseAgent`; tool-aware agents could not be built from config.
   - Files: `mini_vlm_toolkit/agent/__init__.py`
3) Agent config silently dropped unknown kwargs.
   - Files: `mini_vlm_toolkit/agent/config.py`, `mini_vlm_toolkit/agent/__init__.py`
4) Inference path lacked any tool-calling loop support.
   - Files: `mini_vlm_toolkit/agent/base.py`

## Resolved (from Friday, February 6, 2026)
1) Tool execution no longer fails from positional/keyword mismatch; `_run(...)` now calls `tool_execute_request` with keyword arguments and reaches `/execute`.
   - Files: `mini_vlm_toolkit/agent/base.py`, `mini_vlm_toolkit/utils/api_utils.py`
2) Tool-call schema mismatch (prompt vs execution keys) is removed by adopting native OpenAI-style `assistant.tool_calls` (`function.name` + JSON `arguments`) end-to-end.
   - Files: `mini_vlm_toolkit/agent/base.py`, `mini_vlm_toolkit/agent/constant.py`
3) Tool execute feedback is now normalized and consumable; structured envelopes are preserved (including non-2xx envelope errors), and results are parsed into tool messages.
   - Files: `mini_vlm_toolkit/utils/api_utils.py`, `mini_vlm_toolkit/agent/base.py`
4) Tool-loop undefined-variable crash path is fixed (`tool_response` no longer referenced before assignment).
   - Files: `mini_vlm_toolkit/agent/base.py`
5) Tool outputs are now appended as proper `role="tool"` messages with `tool_call_id` (including multimodal image parts when applicable), replacing synthetic user-message feedback.
   - Files: `mini_vlm_toolkit/agent/base.py`
