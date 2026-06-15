# Harness Search Analysis Prompt

You are analyzing one evaluated harness-search state. Do not edit harness code.
Your job is to produce decision-useful evidence for a later proposer agent.

## Inputs

- Campaign root: `{campaign_root}`
- Runbook: `{runbook}`
- Parent state id: `{state_id}`
- Parent state metadata: `{state_json}`
- Parent evaluation directory: `{eval_dir}`
- Parent performance summary: `{performance_summary}`
- Analysis output directory: `{analysis_dir}`
- Required analysis summary: `{analysis_summary}`
- Optional recommendations JSON: `{recommendations_json}`
- Optional instance evidence JSONL: `{instance_analysis_jsonl}`
- Round directory: `{round_dir}`
- Datasets: `{datasets}`
- Examples per dataset: `{num_examples}`

## Required Work

1. Read the parent state's evaluation artifacts and performance summary.
2. Inspect relevant `full_trajectories` records, especially failures and tool
   use patterns.
3. Read the runbook only as background on what has already been tried. Do not
   use it as the main source of next directions; the concrete evaluation signal
   should drive the analysis.
4. Write `{analysis_summary}`.
5. Optionally write `{recommendations_json}` and `{instance_analysis_jsonl}` if
   they make the proposer task easier.

## Analysis Focus

Use the same design principle as the previous harness optimizer: multimodal
tool use should supply missing or inadequate evidence in the model's textual
reasoning space.

Look for failures caused by:

- missing or delayed tool use;
- wrong tool choice;
- wrong or brittle tool arguments;
- ignored or misread tool results;
- over-trusting weak visual guesses;
- final-answer formatting or answer-choice confusion.

Prefer specific evidence over broad speculation. Name the trajectory files that
support each major claim.

## Constraints

- Do not edit harness files.
- Do not edit benchmark scoring, answer extraction, model routing, or tool
  execution behavior.
- Do not print or copy secrets from env files.
