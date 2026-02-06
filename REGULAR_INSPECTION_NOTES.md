# Regular Inspection Notes

Inspection procedure (brief): Review all non-`mini_vlm_toolkit/tool_infra/` code for evaluation correctness risks and tool-usage agent design gaps; note concrete issues with file refs and list any open questions for maintainers. When updating this doc, replace the top block's Date with the run date and add new items under the existing sections; move resolved items into a "Resolved (from <date>)" block at the bottom labeled with the original top-block date.

---

Date: Friday, February 6, 2026
Scope: Evaluation codebase (everything outside `mini_vlm_toolkit/tool_infra/`); scanned commits: `acd7742`, `e50cc17`, `d62fdf7`

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
1) `BaseAgentWithTools` cannot execute tools as implemented: `_run(...)` calls `tool_execute_request(self.tool_base, tool_call, ...)` positionally, but `tool_execute_request` only accepts keyword args. This raises `TypeError` on every tool call, so `/execute` is never reached.
   - Files: `mini_vlm_toolkit/agent/base.py`, `mini_vlm_toolkit/utils/api_utils.py`
2) Tool-call schema is internally inconsistent. Prompts require `{"action":"tool_call","name":...,"parameters":...}`, but execution reads `action["tool_name"]` and `action["tool_args"]`. Prompt-conformant outputs will fail with missing-key errors.
   - Files: `mini_vlm_toolkit/agent/constant.py`, `mini_vlm_toolkit/agent/base.py`
3) Tool execution feedback is not consumable by the model. `tool_execute_request` returns a raw `requests.Response`, and `_update_msg_after_execution` injects it as string text (e.g., `<Response [200]>`) instead of parsed tool output.
   - Files: `mini_vlm_toolkit/utils/api_utils.py`, `mini_vlm_toolkit/agent/base.py`
4) The tool loop can crash on non-tool/non-final actions because `tool_response` is referenced before assignment.
   - Files: `mini_vlm_toolkit/agent/base.py`
5) `DVDAgent` is non-functional in the registry path: it calls `super().__init__` without required `tool_base`, then references undefined names (`config`, `frame_inspect_tool`, `clip_search_tool`, etc.) in constructor/runtime logic.
   - Files: `mini_vlm_toolkit/agent/dvd.py`, `mini_vlm_toolkit/agent/base.py`, `mini_vlm_toolkit/agent/__init__.py`
6) Tool outputs are represented as a synthetic user message (`"Tool Execution Results: ..."`), not as tool-role messages. This diverges from function-calling conversation format and weakens iterative tool use reliability.
   - Files: `mini_vlm_toolkit/agent/base.py`

## Questions to Raise With Maintainers
1) Are video datasets ever run with multiple fps/nframe configurations in the same `work_dir`? If yes, the filename collision is a correctness bug (not just a footgun).
2) Is `PRED_FORMAT=json` intended to be fully supported for evaluation? If yes, evaluator loading paths need harmonization (DataFrame vs dict/list).
3) Which action JSON schema is canonical for tool calls: `name`/`parameters` or `tool_name`/`tool_args`? The code and prompt must converge on one.
4) Should tool feedback be fed back via structured `assistant.tool_calls` + `tool` messages instead of plain user text?
5) Is `DVDAgent` intended to be production-ready now, or should it be hidden from `AVAILABLE_AGENTS` until its dependencies/runtime wiring are complete?

## Resolved (from Monday, January 26, 2026)
1) No tool-call support in `BaseAgent`: a tool-aware implementation now exists (`BaseAgentWithTools`) and is exposed in agent config/build flow.
   - Files: `mini_vlm_toolkit/agent/base.py`, `mini_vlm_toolkit/agent/__init__.py`
2) Agent registry only exposed `BaseAgent`; tool-aware agents could not be built from config.
   - Files: `mini_vlm_toolkit/agent/__init__.py`
3) Agent config silently dropped unknown kwargs.
   - Files: `mini_vlm_toolkit/agent/config.py`, `mini_vlm_toolkit/agent/__init__.py`
4) Inference path lacked any tool-calling loop support.
   - Files: `mini_vlm_toolkit/agent/base.py`
