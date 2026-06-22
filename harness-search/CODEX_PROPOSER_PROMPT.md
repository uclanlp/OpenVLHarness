# Harness Search Proposer Prompt

You are proposing exactly `{max_proposals}` distinct harness-text patches for
one evaluated parent state. You may inspect and temporarily edit the scratch
checkout, but the persistent output must be proposal files under
`{proposals_dir}` and runbook edits in `{runbook}`.

## Inputs

- Campaign root: `{campaign_root}`
- Runbook: `{runbook}`
- Parent state id: `{state_id}`
- Parent state metadata: `{state_json}`
- Parent evaluation directory: `{eval_dir}`
- Analysis summary: `{analysis_summary}`
- Proposal output directory: `{proposals_dir}`
- Round directory: `{round_dir}`
- Datasets: `{datasets}`
- Examples per dataset: `{num_examples}`
- Required proposals: `{max_proposals}`
- Playground max tasks: `{playground_max_tasks}`
- Playground trials per task: `{playground_trials_per_task}`
- Replay script: `{playground_replay_script}`
- Replay router args:
  `--router-env-file {router_env_file} --router-state-dir {router_state_dir} --router-health-timeout {router_health_timeout}`
- Agent config for replay: `{agent_config}`

## Allowed Candidate Edit Surface

Each proposal may edit any of these model-facing harness-text surfaces:

- `mini_vlm_toolkit/agent/constant.py`
- `mini_vlm_toolkit/tool_infra/src/tools/**/tool.py`
- `mini_vlm_toolkit/tool_infra/src/tools/**/process_tool_output.py`

Do not propose changes outside this surface. Do not alter benchmark scoring,
answer extraction, model routing, network behavior, image-processing
algorithms, or tool execution semantics.

## Generalization Constraints (read carefully)

The evaluation slice is a small sample (`{num_examples}` per dataset) drawn from a
much larger held-out test distribution. The search measures accuracy only on that
small slice, so any change that exploits quirks of these specific instances will
look good here and regress on the real test set. Selection now also applies a
paired lower-confidence-bound gate, so changes that merely shuffle a few instances
around for a tiny net gain will be rejected. Propose only changes that improve the
harness for the *distribution*, not for the visible examples.

Hard rules — a proposal that violates any of these is invalid:

- Do NOT condition behavior on the dataset name, split, instance id, file name,
  question text, or any identifier that ties logic to specific examples.
- Do NOT hardcode, pattern-match, or bias toward specific answers, answer strings,
  multiple-choice letters, or numeric values seen in the trajectories.
- Do NOT encode example-specific facts, lookups, or "if you see X say Y" shortcuts.
- Do NOT special-case a handful of failing instances; if a fix would only help the
  specific instances you inspected, it is overfitting — discard it.

Prefer instead:

- Minimal, general edits with a clear mechanism that would plausibly help unseen
  questions of the same kind (clearer instructions, better tool descriptions,
  more robust output formatting, fixing a genuinely misleading prompt).
- The smallest diff that achieves the intended general effect. When two proposals
  are otherwise comparable, prefer the simpler/smaller one.
- Changes whose benefit you can justify by a reusable reason, not by "it fixed
  these N trajectories."

## Required Process

1. Read `{analysis_summary}` first.
2. Inspect the parent evaluation artifacts and relevant trajectories.
3. Use the required playground before finalizing proposals.
   - Select up to `{playground_max_tasks}` relevant saved tasks from the
     parent state's `full_trajectories`.
   - Write `{proposals_dir}/playground/HYPOTHESES.md`.
   - Run up to `{playground_trials_per_task}` replay trials per selected task.
     A trial is one replay of a selected task under a temporary candidate edit
     or hypothesis.
   - Use `{playground_replay_script}` with the replay router args shown above,
     `--agent-config {agent_config}`, one `--trajectory`, and an output
     directory under `{proposals_dir}/playground/`.
   - Restore the scratch checkout to the parent state before finalizing output
     files.
   - Write `{proposals_dir}/playground/PLAYGROUND_SUMMARY.md`.
4. Edit `{runbook}` directly under `Tried Ideas` with a concise hypothesis
   entry for this parent. These are proposed ideas only; the bookkeeper will
   add measured results later.
5. Emit exactly `{max_proposals}` proposal directories:
   - `{proposals_dir}/p001`
   - `{proposals_dir}/p002`
   - continue through exactly `{max_proposals}` proposals

## Required Proposal Layout

Write `{proposals_dir}/proposals.json` with a list of all proposals. Use this
shape:

```json
{{
  "parent_state_id": "{state_id}",
  "proposal_count": {max_proposals},
  "proposals": [
    {{
      "proposal_id": "p001",
      "title": "short name",
      "summary": "one sentence",
      "edited_files": ["repo/relative/path.py"]
    }}
  ]
}}
```

For each proposal directory, write:

```text
pNNN/
  PROPOSAL.md
  metadata.json
  patch.diff
  files/<repo-relative-path>
```

`files/` must contain the full proposed file contents for every edited file,
not just a diff. The controller stores full-file overlays as state. `patch.diff`
is for human review.

`metadata.json` should include:

- `proposal_id`
- `title`
- `summary`
- `edited_files`
- `expected_effect`
- `playground_evidence`
- `generalization_rationale` — one or two sentences explaining the general
  mechanism by which this change helps unseen instances of the same kind, not the
  specific examples you inspected. Proposals that can only justify themselves by
  the visible trajectories must not be emitted.

## Proposal Guidance

You must use the full proposal budget. Some naive-looking text changes may be
surprisingly useful for a downstream LM, so produce exactly `{max_proposals}`
small, diverse proposals unless the parent state cannot be analyzed at all.

Keep each proposal focused. Different proposals may edit different allowed
files, or different parts of the same file, if that best explores the analysis
signal.

Every proposal must obey the Generalization Constraints above. Diversity of
proposals should come from different *general* hypotheses about the harness, never
from tuning to different subsets of the visible instances.
