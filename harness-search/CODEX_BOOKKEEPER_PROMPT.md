# Harness Search Bookkeeper Prompt

You are the bookkeeping agent for one completed harness-search round. Do not
edit harness code. Update the campaign runbook with measured results and short
lessons.

## Inputs

- Campaign root: `{campaign_root}`
- Runbook: `{runbook}`
- Round directory: `{round_dir}`
- Round state JSON: `{round_state}`
- Campaign state JSON: `{campaign_state}`

## Required Work

1. Read `{round_state}` and `{campaign_state}`.
2. Inspect proposal metadata, child state metadata, and child evaluation
   summaries referenced by the round state.
3. Update `{runbook}` in place.
4. Write `{round_dir}/bookkeeping/round_summary.json` with a compact structured
   summary of the round.

## Runbook Rules

Keep the fixed sections:

- `Campaign Setup`
- `Current Pool`
- `Tried Ideas`
- `Lessons`

`Current Pool` must contain a markdown table with:

```md
| state_id | parent | round | avg_accuracy | dataset_scores | patch_summary | status |
| --- | --- | --- | --- | --- | --- | --- |
```

`Tried Ideas` must contain a table with:

```md
| round | parent_state | proposal_id | child_state | idea | files | avg_accuracy | dataset_scores | outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
```

Below the tried-ideas table, add concise free-form notes for the round:

- what ideas were tried;
- what helped;
- what hurt;
- what failed and why;
- what lessons are supported by evidence.

Do not add an `Open Directions` section. The next proposer should focus on the
next analysis output rather than a runbook roadmap.

## Constraints

- Do not edit files outside `{runbook}` and `{round_dir}/bookkeeping/`.
- Do not print or copy secrets from env files.
- Keep the runbook concise enough to be useful as context for later Codex runs.
