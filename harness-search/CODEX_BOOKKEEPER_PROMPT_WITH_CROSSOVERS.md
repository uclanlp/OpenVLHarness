# Harness Search Bookkeeper Prompt With Crossovers

You are the bookkeeping agent for one completed harness-search round. Do not
edit harness code. Update the campaign runbook with measured results and short
lessons.

This run may evaluate both single proposals and crossover combinations of
multiple non-conflicting proposals from the same parent state.

## Inputs

- Campaign root: `{campaign_root}`
- Runbook: `{runbook}`
- Round directory: `{round_dir}`
- Round state JSON: `{round_state}`
- Campaign state JSON: `{campaign_state}`

## Required Work

1. Read `{round_state}` and `{campaign_state}`.
2. Inspect proposal metadata, child state metadata, skipped crossover entries,
   and child evaluation summaries referenced by the round state.
3. Update `{runbook}` in place.
4. Write `{round_dir}/bookkeeping/round_summary.json` with a compact structured
   summary of the round.

## Crossover Handling

Child states may include:

- single proposals: `combo_size = 1`, `is_crossover = false`;
- crossovers: `combo_size > 1`, `is_crossover = true`, with
  `source_proposal_ids` listing the component proposals.

`round_state.skipped_crossovers` lists crossover combinations that were
considered but not evaluated because their edits conflicted. Record these as
tried/skipped ideas, not as evaluated failures.

When summarizing performance, compare:

- each single proposal against its parent;
- each crossover against its parent and its component single proposals when
  those singles were evaluated;
- skipped crossovers separately from evaluated children.

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

For crossover rows, use the combined proposal id such as `p001+p003` and mention
the component proposal ids in the idea or outcome. For skipped crossovers, leave
`child_state`, `avg_accuracy`, and `dataset_scores` empty and set the outcome to
`skipped_conflict: <reason>`.

Below the tried-ideas table, add concise free-form notes for the round:

- what single ideas were tried;
- what crossover combinations were tried;
- what helped;
- what hurt;
- which crossovers were skipped and why;
- what lessons are supported by evidence.

Do not add an `Open Directions` section. The next proposer should focus on the
next analysis output rather than a runbook roadmap.

## Constraints

- Do not edit files outside `{runbook}` and `{round_dir}/bookkeeping/`.
- Do not print or copy secrets from env files.
- Keep the runbook concise enough to be useful as context for later Codex runs.
