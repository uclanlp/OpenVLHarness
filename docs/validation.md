# Release validation

The baseline is `opensource` commit
`b98e5a540c4867eab20e07166c274e7ca8b8b9fe`. Validation uses the same frozen
examples and model/raw-tool checkpoints before and after extraction:

- 13 positive ODinW-13 images, one per domain.
- 12 FSC147 images across count/density strata, selected with seed 42.
- Five supplemental chains per backbone: grounding/zoom, depth/coding,
  OCR/coding, ordered camera/coding, reverse-image/text-search/webpage visiting.

The two backbones are Qwen3-VL-8B-Instruct (Chat Completions) and GPT-6-Luna
(Responses). Qwen multimodal search summarization is shared. ODinW uses COCO box
metrics; FSC147 uses direct count parsing. No LLM judge is used. These small
sets validate wiring and regressions; they are not full benchmark estimates.

## Deterministic replay

All 60 baseline trajectories were replayed with saved model and raw tool
responses, network access forbidden. The check covers 1,452 boundaries,
including all 152 model request histories/settings, resolved tool arguments,
processed evidence, memory images/arrays/descriptors, environment emissions,
detections and predictions.

No unexplained differences remain. The retained mismatches are:

- OCR's approved new binding and its downstream effects when replaying old
  generated Python that never had the binding. Suppressing only that binding
  in an external replay experiment restores exact equality for both OCR rows.
- Two non-model-facing Python function `repr` addresses in one counting row.
  The strict records are retained; normalizing only process addresses explains
  those mismatches.

Checked-in fixtures preserve public schemas, provider requests, memory and tool
results independently of current implementation. The offline suite also covers
shared depth/camera capacity, responsive health while queued, failure cleanup,
worker-thread execution timeouts, optional imports and copied numerical bindings.

## Fresh comparison

Fresh runs use separate capture, router and output directories; the baseline
predictions and observations are immutable. Live response text, web results and
SAM3 overlay colors may vary. Exact behavior claims come from saved-response
replay; every live disagreement is retained for inspection.

All 60 original cases completed with valid native tool/result cycles, complete
trajectories and sidecars. There were no request/capture/protocol failures,
generation truncations, or FSC count parse failures. Both OCR chains now read the
new binding and compute 19. Qwen's camera chain recovered one missing NumPy import
in generated code, a failure type also present in the baseline.

| Backbone | ODinW AP before → after | FSC MAE before → after | FSC RMSE before → after |
| --- | --- | --- | --- |
| Qwen3-VL-8B | 66.99 → 70.94 | 16.00 → 16.00 | 31.27 → 31.27 |
| GPT-6-Luna | 74.27 → 67.73 | 5.00 → 14.67 | 11.16 → 30.42 |

Dataset predictions match exactly on 39/50 cases, including all 12 Qwen counts.
Luna's ODinW decrease is concentrated in one first-call category choice
(stingray versus starfish). Its largest FSC difference follows the query
apple versus apples and a changed zoom sequence on a dense image. The original
results remain in the comparison; they were not replaced by reruns.

Six Luna trajectories covering the changed cases were replayed in reverse,
through the immutable old runtime using new saved replies. All 164 boundary
comparisons passed, and all six initial native request bodies match the original
baseline exactly. Thus the old runtime reproduces those new trajectories given
the same responses. This rules out refactor handling differences on those cases;
it does not establish statistical performance equivalence.

A further fixed 25-example Luna slice, excluding all original IDs, was selected
before inference to diagnose the material score variance: one additional positive
ODinW image/domain and one FSC image/original count-density stratum. Old/new
controls use separate logs and the same model/raw services. Results are reported
separately from the original set:

| Luna slice | ODinW AP before → after | FSC MAE before → after | FSC RMSE before → after |
| --- | --- | --- | --- |
| Additional 13 ODinW / 12 FSC | 67.89 → 69.47 | 31.83 → 27.75 | 94.11 → 72.74 |
| Combined 26 ODinW / 24 FSC | 68.90 → 67.12 | 18.42 → 21.21 | 67.01 → 55.75 |

Combined AP was recomputed over the combined annotations and predictions, not
averaged from the two slice scores. Additional-set predictions agree on 19/25
cases. Both completed controls have no request/capture/protocol/parse failures or
truncations; the separately preserved failed provider attempt is described below.
The added slice shifts in the opposite direction from the initial gap. These
observations support the trajectory-level diagnosis of model variation, without
claiming statistical equivalence from a small sample.

The final offline suite passes 123 tests and 215 subtests, including retained
real evaluator/reporting tests after removing old runtime modules. An actual
public CLI run on a frozen FSC image produced predictions, direct metrics,
full trajectories, HTML, copied images and detection sidecars with judge=None.
An evaluation-only resume after retirement preserved all prior CLI artifacts and
made zero new model requests. A clean installed wheel completes a fixed native-protocol fixture outside the
checkout with no legacy runtime, Torch, Transformers, OpenAI SDK or FastAPI.

Raw SAM3 and DA3 were refreshed from the extracted package. SAM3 boxes/scores/masks
and depth arrays match exactly; camera extrinsics differ by at most 2.3544e-05
after restart while preserving dtype/shape/convention. DINO's adapter is tested
with fixtures; Kimi and live DINO inference are outside this validation pair.
Fresh GPU dependency installations have not been reproduced from scratch.

Original sample manifest SHA256:
`bd0edb4cac8f927d0c4052799c4c8157b1e536d047ad282e1f3a5bf7e5d84eb2`.
Diagnostic extension manifest SHA256:
`44e0890d56c577ea0ac24dfb5fa1f6d1e2d928764f5a81834ec3f5857e64c90d`.

Live search audit: both reverse-image calls sent eight image parts to Qwen and
returned parsed summaries/retained indices. Text summaries also parsed. One Qwen
webpage extraction returned valid JSON with empty content, triggering the existing
raw-page content fallback; a subsequent Taobao visit produced a nonempty extraction.
Luna's two webpage extractions succeeded. The content fallback is retained and
reported explicitly; it was not a provider substitution or JSON parse failure.

## Retained limitations found during validation

The additional old-code control encountered one upstream HTTP500 after four
successful turns. That failed attempt remains recorded separately; the same
fixed examples were recovered under a new control label. An offline reproducer
confirmed a numeric-host retry-classification bug in both implementations at
that time. Subsequent review fixes replace text parsing with structured HTTP
status checks; numeric-host 503 and misleading-body 400 cases have regression
coverage. The saved experiment artifacts retain their original behavior.

Sixty existing dataset/evaluation utility tests also pass after retirement, once
LMUData is configured. Their stale-lock concurrency test emits a thread-level
FileExistsError warning from the unchanged file_db implementation. This is a
retained evaluator issue, separate from the new runtime's per-question memory
and coding-output isolation tests. The original test-collection attempt without
LMUData and the successful configured run are both retained in local evidence.
