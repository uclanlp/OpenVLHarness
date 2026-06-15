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

## Proposal Guidance

You must use the full proposal budget. Some naive-looking text changes may be
surprisingly useful for a downstream LM, so produce exactly `{max_proposals}`
small, diverse proposals unless the parent state cannot be analyzed at all.

Keep each proposal focused. Different proposals may edit different allowed
files, or different parts of the same file, if that best explores the analysis
signal.
