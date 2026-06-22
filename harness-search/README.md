# Harness Search

`harness-search` is a research controller for tree-search optimization of the
mini-vlm-toolkit harness. It is derived from the older `harness-optimization`
flow, but instead of applying one linear chain of edits it maintains a pool of
promising harness states, asks Codex to analyze and propose edits, evaluates the
resulting states, and keeps the best states for the next round.

The implementation is intentionally simple: states are represented as full-file
overlays, scratch repositories are temporary copies, and each campaign keeps all
metadata and logs under one campaign directory.

## High-Level Algorithm

Each campaign starts from the current repository snapshot as state `s0000`.

For every round:

1. Start from the previous round's state pool.
2. For each parent state in the pool:
   - materialize the parent state into a scratch repository;
   - run the Codex analysis agent;
   - run the Codex proposer agent, which must emit exactly `M` proposals;
   - create candidate child states from the proposals;
   - evaluate child states.
3. Rank the previous pool and all newly evaluated children by average accuracy.
4. Keep the top `N` states for the next round.
5. Run the Codex bookkeeper to update the campaign runbook.

The objective is the unweighted mean of dataset accuracies. Ties are broken by
state creation order, so older states win exact ties.

## Launch Scripts

Run from the repository root.

### Debug Run

```bash
bash harness-search/run_harness_search_debug.sh
```

Defaults:

- datasets: `ERQA`, `MMSIBench_wo_circular`
- examples per dataset: `20`
- pool size: `3`
- proposals per parent: `3`
- rounds: `5`
- per-state benchmark workers: `50`
- parallel child states: `4`
- crossover max size: `1` through the controller default
- router base port: `8101`

With `PARALLEL_STATES=4`, child evaluation may use ports `8101..8104`.

### Large Run

```bash
bash harness-search/run_harness_search_large.sh
```

Current defaults:

- examples per dataset: `200`
- pool size: `4`
- proposals per parent: `4`
- rounds: `10`
- per-state benchmark workers: `50`
- parallel child states: `4`
- crossover max size: `1` through the controller default
- router base port: `8104`

With `PARALLEL_STATES=4`, child evaluation may use ports `8104..8107`.

### Large Run With Crossovers

```bash
bash harness-search/run_harness_search_large_w_crossovers.sh
```

This mirrors the current large-run settings, but enables crossovers:

- crossover max size: `4`
- router base port: `8110`

With `PARALLEL_STATES=4`, child evaluation may use ports `8110..8113`. This is
separate from the default large-run port range so it can run alongside a large
run if the other shared services can handle the load.

## Common Environment Overrides

The launch scripts expose the main operational knobs as environment variables:

```bash
NUM_PROC=25 PARALLEL_STATES=2 ROUTER_PORT=8120 \
  bash harness-search/run_harness_search_large_w_crossovers.sh
```

Important knobs:

- `NUM_PROC`: passed to `run_benchmark.py --num-proc`; this is per evaluated
  state, not total campaign parallelism.
- `PARALLEL_STATES`: number of child states evaluated concurrently.
- `ROUTER_PORT`: base router port. Parallel child evaluation uses
  `ROUTER_PORT + slot_index`.
- `CROSSOVER_MAX_SIZE`: only used by the crossover launcher unless passed
  manually to `run_search.py`.
- `CODEX_USAGE_THRESHOLD_PERCENT`: record usage after each controller-side
  `codex exec`, then sleep before the next Codex call when the primary Codex
  usage window reaches this percent. Defaults to `90`; set to `0` to disable.
- `CODEX_USAGE_SLEEP_CUSHION_SECONDS`: extra sleep after the reported primary
  usage reset time. Defaults to `60`.
- `CODEX_STATUS_FAILURE_MODE`: behavior when the backend PTY `/status` probe
  cannot parse usage. Defaults to `warn`; set to `fail` for strict quota
  protection.
- `CODEX_STATUS_STARTUP_DELAY`: seconds to wait before sending `/status` in the
  backend PTY probe. Defaults to `8`; increase this if Codex is still booting
  MCP servers when the probe runs.
- `CODEX_EXEC_TIMEOUT_SECONDS`: hard timeout for each analysis/proposer/
  bookkeeper `codex exec` phase. Defaults to `3600`; set to `0` to disable.
- `CODEX_EXEC_RETRIES`: retries after the first failed or timed-out Codex
  phase attempt. Defaults to `1`.

Total benchmark worker pressure during child evaluation is roughly:

```text
NUM_PROC * PARALLEL_STATES
```

For example, `NUM_PROC=50` and `PARALLEL_STATES=4` means up to 200 benchmark
workers across four candidate states.

## Direct Controller Usage

Cheap controller-only dry run:

```bash
conda run --no-capture-output -n dpr-lite python harness-search/scripts/run_search.py \
  --datasets ERQA MMSIBench_wo_circular \
  --num-examples 2 \
  --num-rounds 1 \
  --pool-size 3 \
  --max-proposals 3 \
  --num-proc 50 \
  --parallel-states 4 \
  --crossover-max-size 1 \
  --work-root /tmp/harness_search_dryruns \
  --campaign-id dryrun \
  --router-env-file .env.latest \
  --router-port 8120 \
  --dry-run
```

The default controller values are:

- `--num-proc 50`
- `--parallel-states 4`
- `--crossover-max-size 1`
- `--pool-size 3`
- `--max-proposals 3`
- `--num-rounds 1`
- `--num-examples 20`

## Visualizing Campaign Results

Generate a self-contained HTML report for a campaign directory:

```bash
conda run --no-capture-output -n dpr-lite python harness-search/scripts/visualize_search.py \
  /local/<user>/tool_vlm/harness_search_logs/harness_search_20260514_155538
```

By default this writes:

```text
<campaign_root>/search_report.html
```

The report reads `campaign_state.json`, per-state `state.json` files, completed
`round_state.json` files, and any bookkeeping summaries. This means it can show
newly completed evaluation rounds even while campaign bookkeeping is still
catching up.

## Router Ports And Runtime Config

Each campaign creates campaign-local runtime config files:

```text
<campaign_root>/configs/router.env
<campaign_root>/configs/agent.yaml
```

The router env is copied from the source router env, usually `.env.latest`, with
`PORT` overridden. The agent config is copied from
`harness-search/configs/run_benchmark_agent.yaml`, with `agent_kwargs.tool_base`
set to the matching router URL.

For parallel child evaluation, each child state gets its own eval-local runtime
config:

```text
<campaign_root>/states/<state_id>/eval/runtime/router.env
<campaign_root>/states/<state_id>/eval/runtime/agent.yaml
<campaign_root>/states/<state_id>/eval/runtime/router_state/
```

Before baseline evaluation and before every parallel child-evaluation batch, the
controller checks that the relevant port range is available. If a port is
occupied, the run fails before launching that batch. The router manager also
checks that an existing server is actually compatible by requiring:

- `/tools` returns a JSON object;
- `/llm-schema` returns a JSON list.

Do not run two campaigns with overlapping port ranges. The work directories are
isolated, but overlapping router ports will cause one campaign to fail or talk
to the wrong server.

## State Representation

The base repository snapshot is stored at:

```text
<campaign_root>/base_repo/
```

Each state lives under:

```text
<campaign_root>/states/<state_id>/
```

Important files:

- `state.json`: metadata, parent id, score, proposal id, edited files, status.
- `overlay/`: full-file copies for files changed relative to the parent/base.
- `patch.diff`: unified diff against the parent state for the changed files.
- `eval/`: benchmark outputs, logs, runtime config, score summaries.

Scratch repositories are created under:

```text
<campaign_root>/scratch/
```

They are deleted after each analysis, proposer, or evaluation step.

## Edit Surface

Proposal files are validated before state creation. The allowed edit surface is:

- `mini_vlm_toolkit/agent/constant.py`
- `mini_vlm_toolkit/tool_infra/src/tools/**/tool.py`
- `mini_vlm_toolkit/tool_infra/src/tools/**/process_tool_output.py`

Disallowed paths fail candidate creation.

## Proposal Format

The proposer writes into:

```text
<round_dir>/parents/<parent_state>/proposals/
```

Required structure:

```text
proposals.json
playground/HYPOTHESES.md
playground/PLAYGROUND_SUMMARY.md
p001/
  metadata.json
  PROPOSAL.md
  files/<repo-relative edited files>
p002/
...
```

The controller requires exactly `M` proposal directories named `p001..p00M`.
Support directories such as `playground/` are not counted as proposals.

## Crossover Evaluation

`--crossover-max-size 1` preserves the standard behavior: evaluate each single
proposal independently.

When `--crossover-max-size K` is greater than `1`, the controller evaluates:

- all single proposals;
- all valid 2-way combinations;
- all valid 3-way combinations;
- all valid combinations up to size `K`.

For example, with `M=4` and `K=4`, the controller considers:

```text
4 singles + 6 pairs + 4 triples + 1 quadruple = 15 candidates
```

Crossovers are generated only within the same parent state.

### Conflict Policy

The merge policy is deliberately conservative:

- proposals editing disjoint files are compatible;
- proposals editing the same file are compatible only when their changed parent
  line ranges do not overlap;
- insertions at the same parent line conflict;
- ambiguous same-file overlaps are skipped.

Skipped crossovers are recorded in `round_state.json` under
`skipped_crossovers`. They are not evaluated and do not consume a state id.

Evaluated crossover states include metadata such as:

```json
{
  "proposal_id": "p001+p003",
  "source_proposal_ids": ["p001", "p003"],
  "combo_size": 2,
  "is_crossover": true,
  "conflict_status": "evaluated"
}
```

## Parallel Evaluation

Parallelism applies only to child proposal/crossover evaluation. Baseline
evaluation, analysis, proposer, and bookkeeper remain sequential.

The controller creates child states sequentially for deterministic state ids,
then evaluates them in batches of size `--parallel-states`.

For a batch with base port `8110` and `--parallel-states 4`:

```text
slot 0 -> 8110
slot 1 -> 8111
slot 2 -> 8112
slot 3 -> 8113
```

Each slot has its own router env, agent config, router state directory, scratch
repo, artifacts, and logs.

## Runbook And Bookkeeping

Every campaign has:

```text
<campaign_root>/RUNBOOK.md
```

The runbook contains:

- `Campaign Setup`
- `Current Pool`
- `Tried Ideas`
- `Lessons`

At the end of a round, the bookkeeper prompt is chosen by crossover setting:

- `CODEX_BOOKKEEPER_PROMPT.md` when `--crossover-max-size <= 1`;
- `CODEX_BOOKKEEPER_PROMPT_WITH_CROSSOVERS.md` when crossovers are enabled.

The crossover prompt asks the bookkeeper to distinguish single proposals,
evaluated crossovers, and skipped conflicting crossovers.

## Output Layout

Typical campaign layout:

```text
<campaign_root>/
  RUNBOOK.md
  campaign_state.json
  configs/
    router.env
    agent.yaml
  base_repo/
  logs/router/
  rounds/
    r001/
      round_state.json
      parents/<state_id>/
        analysis/
        proposals/
        logs/
      bookkeeping/
  states/
    s0000/
    s0001/
      state.json
      patch.diff
      overlay/
      eval/
        performance_summary.json
        score.json
        eval_state.json
        logs/
        runtime/
```

Useful files while debugging:

- `campaign_state.json`: campaign-level status and all known states.
- `rounds/<round>/round_state.json`: parent expansions, children, skipped
  crossovers, next pool.
- `states/<state>/eval/logs/benchmark.log`: benchmark process output.
- `states/<state>/eval/logs/router_restart.log`: router startup status.
- `rounds/<round>/parents/<state>/logs/codex_analysis.log`
- `rounds/<round>/parents/<state>/logs/codex_proposer.log`
- `rounds/<round>/bookkeeping/codex_bookkeeper.log`

The usage guard checks Codex before analysis, proposer, and bookkeeper calls by
running a short-lived backend PTY session, sending `/status`, parsing the 5-hour
limit and reset time, and sleeping until reset when usage reaches
`--codex-usage-threshold-percent`. The older `codex exec --json` token-count
parser remains as a fallback when Codex emits that event.

Probe Codex status manually:

```bash
python harness-search/scripts/check_codex_status_pty.py
```

Resume an interrupted campaign:

```bash
python harness-search/scripts/run_search.py \
  --resume-campaign-root /path/to/harness_search_<timestamp>
```

Resume skips completed rounds, accepts valid analysis/proposer/bookkeeper
artifacts even if their phase state JSON is stale, and continues the first
incomplete round.

## Tests And Sanity Checks

Compile the controller:

```bash
conda run -n dpr-lite python -m py_compile \
  harness-search/scripts/run_search.py \
  harness-search/scripts/state_store.py \
  harness-search/scripts/manage_router.py
```

Run the lightweight tests:

```bash
conda run -n dpr-lite python harness-search/tests/test_state_store.py
conda run -n dpr-lite python harness-search/tests/test_ranking.py
```

Check shell syntax:

```bash
bash -n harness-search/run_harness_search_debug.sh \
  harness-search/run_harness_search_large.sh \
  harness-search/run_harness_search_large_w_crossovers.sh
```

## Operational Notes

- Use the `dpr-lite` conda environment for controller runs.
- Use `conda run --no-capture-output` when launching through conda so progress
  logs stream to the terminal.
- Keep router port ranges non-overlapping across concurrent campaigns.
- The tool router is restarted for each evaluated state so edited tool metadata
  is loaded fresh.
- Large runs can be costly because proposals, crossovers, parallel states, and
  per-state workers multiply quickly.
