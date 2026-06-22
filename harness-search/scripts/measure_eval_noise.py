#!/usr/bin/env python3
"""Measure the eval noise floor by re-running ONE baseline harness N times.

Why this exists
---------------
The promotion gate compares a child against its parent per-instance (paired):
``net = fixed - broke`` over the instances whose correctness *flipped*, promoting
only when ``net - z*sqrt(fixed + broke) > 0``. If every child gets gated, the
usual cause is not a broken gate but a noisy eval: at ``temperature > 0`` the
*same* harness disagrees with itself across runs, so much of every
``fixed``/``broke`` count is sampling luck, not a real effect of the edit.

This script quantifies that directly. It evaluates the current repo harness
(the baseline) ``--repeats`` times on the same deterministic head(N) slice, then
reports the self-discordance between runs. That self-discordance is the noise
floor: any real prompt effect has to clear it to be detectable, and the gate's
``sqrt(discordant)`` term is measuring exactly this.

It does NOT need a campaign. It just starts the router and runs ``run_benchmark.py``
against the repo, the same way a single search eval does, reusing run_search's
config helpers so the runtime matches.

Usage
-----
    python harness-search/scripts/measure_eval_noise.py \
        --datasets ERQA_val_train --num-examples 100 --repeats 3

Override --router-env-file / --agent-config / --judge-config to profile a
non-default harness. Results print and save to
``<work-dir>/eval_noise.json`` (default work dir: ./harness_eval_noise).
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent


def load_run_search():
    """Import run_search.py to reuse its small config + parsing helpers."""
    module_path = SCRIPT_DIR / "run_search.py"
    spec = importlib.util.spec_from_file_location("harness_run_search", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def paired_stats(a: Dict[str, bool], b: Dict[str, bool]) -> Dict[str, Any]:
    """McNemar-style paired comparison treating `a` as parent and `b` as child."""
    shared = sorted(set(a) & set(b))
    fixed = sum(1 for k in shared if b[k] and not a[k])
    broke = sum(1 for k in shared if a[k] and not b[k])
    return {
        "n_shared": len(shared),
        "fixed": fixed,
        "broke": broke,
        "net": fixed - broke,
        "discordant": fixed + broke,
        "flip_rate": ((fixed + broke) / len(shared)) if shared else 0.0,
    }


def _vote_correct_prob(p: float, k: int) -> float:
    """P(majority of k independent samples is correct) for true pass-rate p.

    Odd k has no ties. For even k an exact split counts as half a vote (random
    tie-break), which is why voting is conventionally done with odd k.
    """
    from math import comb
    half = k / 2.0
    prob = 0.0
    for j in range(k + 1):
        weight = 1.0 if j > half else (0.5 if j == half else 0.0)
        if weight:
            prob += weight * comb(k, j) * (p ** j) * ((1.0 - p) ** (k - j))
    return prob


def samples_per_instance_curve(
    counts: Dict[str, int], runs: int, k_values: List[int], z: float
) -> List[Dict[str, Any]]:
    """Project the paired noise floor vs k for averaging and majority voting.

    `counts[task] = #correct out of `runs` independent samples`, so p_hat = c/runs.
    Averaging needs only S = sum p(1-p) (unbiased estimator below); voting needs the
    per-instance p_hat. SE is the standard error (in instance-count units, i.e. out
    of N) of the paired difference between two independent k-sample evaluations; the
    gate promotion bar is z*SE.
    """
    n = len(counts) or 1
    single_acc = sum(counts.values()) / (runs * n)
    # Unbiased estimate of S = sum_i p_i(1-p_i):  c(R-c)/(R(R-1)).
    if runs >= 2:
        s_total = sum(c * (runs - c) / (runs * (runs - 1)) for c in counts.values())
    else:
        s_total = sum((c / runs) * (1 - c / runs) for c in counts.values())
    # Small-sample (R/(R-1)) correction so the voting variance uses the same
    # unbiased per-instance basis as averaging; this makes the two columns agree
    # exactly at k=1 (where "majority of 1" == "mean of 1" == the single sample).
    bias_correction = runs / (runs - 1) if runs >= 2 else 1.0
    rows: List[Dict[str, Any]] = []
    for k in k_values:
        avg_se = math.sqrt(2.0 * s_total / k) if s_total > 0 and k > 0 else 0.0
        vote_var = 0.0
        vote_acc_sum = 0.0
        for c in counts.values():
            p = c / runs
            q = _vote_correct_prob(p, k)
            vote_var += 2.0 * q * (1.0 - q)
            vote_acc_sum += q
        vote_se = math.sqrt(vote_var * bias_correction)
        rows.append({
            "k": k,
            "avg_se": avg_se,
            "avg_bar": z * avg_se,
            "vote_se": vote_se,
            "vote_bar": z * vote_se,
            "vote_acc": vote_acc_sum / n,
            "single_acc": single_acc,
        })
    return rows


def main() -> None:
    run_search = load_run_search()
    repo_root = run_search.REPO_ROOT

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", required=True, help="Datasets to evaluate (same slice the search trains on, e.g. ERQA_val_train).")
    parser.add_argument("--num-examples", type=int, default=100, help="Examples per dataset (head(N) slice). Match the search value.")
    parser.add_argument("--repeats", type=int, default=3, help="How many times to re-evaluate the baseline (>= 2). More repeats = tighter estimate.")
    parser.add_argument("--num-proc", type=int, default=8, help="Eval parallelism passed to run_benchmark.py.")
    parser.add_argument("--router-env-file", default="harness-search/configs/.env.latest", help="Router env file (defines the tool router). Default matches the search scripts.")
    parser.add_argument("--agent-config", default=run_search.DEFAULT_AGENT_CONFIG, help="Agent config = the harness under test. Default is the baseline benchmark agent.")
    parser.add_argument("--judge-config", default=run_search.DEFAULT_JUDGE_CONFIG, help="Judge config for scoring.")
    parser.add_argument("--router-port", type=int, default=None, help="Router port. Defaults to PORT in the env file, else 8120.")
    parser.add_argument("--router-health-timeout", type=float, default=90.0, help="Seconds to wait for the router to become healthy.")
    parser.add_argument("--gate-z", type=float, default=1.0, help="z used to report the promotion bar implied by the measured noise.")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3, 4, 5, 7], help="Samples-per-instance values to measure the noise (SE) for.")
    parser.add_argument("--work-dir", type=Path, default=Path("harness_eval_noise"), help="Where to write per-run artifacts and the summary JSON.")
    parser.add_argument("--crn", action="store_true", help="Common Random Numbers: fix a per-instance sampling seed so identical harnesses sample identically. Re-running the same harness then exposes only the residual (batching/tool) noise floor.")
    parser.add_argument("--reuse-existing", action="store_true", help="Skip re-running; analyze noise_run_* artifacts already in --work-dir.")
    args = parser.parse_args()

    if args.repeats < 2:
        raise SystemExit("--repeats must be >= 2 to measure run-to-run discordance")

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (repo_root / path)

    source_router_env = resolve(args.router_env_file)
    source_agent_config = resolve(args.agent_config)
    judge_config = resolve(args.judge_config)
    if not source_router_env.exists():
        raise SystemExit(f"router env file not found: {source_router_env}")
    if not source_agent_config.exists():
        raise SystemExit(f"agent config not found: {source_agent_config}")

    env_values = run_search.parse_env_file(source_router_env)
    router_port = args.router_port or int(env_values.get("PORT") or 8120)
    tool_base = f"http://localhost:{router_port}"

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # Per-port router env + agent config pointing at this router (reuses search helpers).
    runtime_dir = work_dir / "runtime"
    router_env_path = runtime_dir / "router.env"
    agent_config_path = runtime_dir / "agent.yaml"
    run_search.write_router_env_with_port(source_router_env, router_env_path, router_port)
    run_search.write_agent_config_with_tool_base(source_agent_config, agent_config_path, tool_base)
    if args.crn:
        import yaml  # local import; pyyaml is already a dependency of the eval stack
        cfg = yaml.safe_load(agent_config_path.read_text(encoding="utf-8"))
        cfg.setdefault("agent_kwargs", {})["paired_sampling_seed"] = True
        agent_config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print("CRN enabled: paired_sampling_seed=True injected into the eval agent config.")
    benchmark_env = run_search.build_benchmark_env(repo_root, tool_base)
    router_state_dir = runtime_dir / "router_state"

    def manage_router(action: str) -> None:
        cmd = [
            sys.executable, "harness-search/scripts/manage_router.py", action,
            "--env-file", str(router_env_path),
            "--state-dir", str(router_state_dir),
            "--health-timeout", str(args.router_health_timeout),
        ]
        subprocess.run(cmd, cwd=str(repo_root), check=True)

    print("=" * 72)
    mode = "CRN / paired seeds (residual floor)" if args.crn else "independent sampling (full floor)"
    print(f"Eval noise-floor probe (baseline harness, repeated runs) — {mode}")
    print("=" * 72)
    print(f"agent_config  : {source_agent_config}")
    print(f"router env    : {source_router_env}  (port {router_port})")
    print(f"datasets      : {', '.join(args.datasets)}")
    print(f"num_examples  : {args.num_examples}  |  num_proc: {args.num_proc}")
    print(f"repeats       : {args.repeats}  |  gate_z: {args.gate_z}")
    print(f"work_dir      : {work_dir}")
    print()

    correctness_per_run: List[Tuple[str, Dict[str, bool]]] = []
    try:
        for k in range(args.repeats):
            name = f"noise_run_{k}"
            artifacts = work_dir / name / "artifacts"
            if not (args.reuse_existing and artifacts.exists()):
                print(f"[run {k}] restarting router + evaluating baseline -> {name}/ ...")
                manage_router("restart")
                cmd = [
                    sys.executable, "run_benchmark.py",
                    "--config", str(agent_config_path),
                    "--judge-config", str(judge_config),
                    "--data", *args.datasets,
                    "--num-proc", str(args.num_proc),
                    "--num-examples", str(args.num_examples),
                    "--work-dir", str(artifacts),
                    "--run-tag", name,
                    "--save-html", "0",
                ]
                subprocess.run(cmd, cwd=str(repo_root), env=benchmark_env, check=True)
            else:
                print(f"[run {k}] reusing existing artifacts in {name}/")
            correct = run_search.load_instance_correctness(artifacts)
            if not correct:
                raise SystemExit(f"no per-instance correctness found under {artifacts}")
            acc = sum(1 for v in correct.values() if v) / len(correct)
            print(f"[run {k}] accuracy = {acc:.4f} over {len(correct)} instances")
            correctness_per_run.append((name, correct))
    finally:
        if not args.reuse_existing:
            try:
                manage_router("stop")
            except Exception as exc:  # best-effort cleanup
                print(f"warning: router stop failed: {exc}")

    print()
    print("-" * 72)
    print("Pairwise self-discordance (same harness, different runs)")
    print("-" * 72)
    print(f"{'pair':<24}{'n':>5}{'fixed':>7}{'broke':>7}{'net':>6}{'discord':>9}{'flip%':>8}")

    pairwise: List[Dict[str, Any]] = []
    for (na, ca), (nb, cb) in itertools.combinations(correctness_per_run, 2):
        st = paired_stats(ca, cb)
        st["pair"] = f"{na} vs {nb}"
        pairwise.append(st)
        print(
            f"{st['pair']:<24}{st['n_shared']:>5}{st['fixed']:>7}{st['broke']:>7}"
            f"{st['net']:>6}{st['discordant']:>9}{100 * st['flip_rate']:>7.1f}%"
        )

    mean_discordant = sum(p["discordant"] for p in pairwise) / len(pairwise)
    mean_abs_net = sum(abs(p["net"]) for p in pairwise) / len(pairwise)
    mean_flip = sum(p["flip_rate"] for p in pairwise) / len(pairwise)
    noise_se = math.sqrt(mean_discordant) if mean_discordant else 0.0
    promo_bar = args.gate_z * noise_se

    # ---- Samples-per-instance projection: SE-vs-k for averaging vs voting ----
    shared_keys = set(correctness_per_run[0][1])
    for _, c in correctness_per_run[1:]:
        shared_keys &= set(c)
    counts = {key: sum(1 for _, c in correctness_per_run if c.get(key)) for key in shared_keys}
    runs = len(correctness_per_run)
    n_instances = len(counts)
    # Pass-rate distribution over the R runs: the "partial" mass is the noise driver.
    n_always = sum(1 for c in counts.values() if c == runs)
    n_never = sum(1 for c in counts.values() if c == 0)
    n_partial = n_instances - n_always - n_never

    curve = samples_per_instance_curve(counts, runs, sorted(set(args.k_values)), args.gate_z)
    se1 = curve[0]["avg_se"] if curve else 0.0
    vote_floor = curve[-1]["vote_se"] if curve else 0.0

    print()
    print("-" * 72)
    print("Does sampling each instance k times stabilize the comparison? (SE = noise on")
    print("a same-harness paired comparison; lower = more stable. avg = pass-rate mean,")
    print("vote = majority label.)")
    print(f"N={n_instances}; pass-rate over {runs} runs: always-right {n_always}, "
          f"always-wrong {n_never}, ambiguous {n_partial}")
    print("-" * 72)
    print(f"{'k':>3}{'avg SE':>9}{'vs k=1':>9}{'vote SE':>9}{'vs k=1':>9}")
    for row in curve:
        a_mult = (se1 / row["avg_se"]) if row["avg_se"] else float("inf")
        v_mult = (se1 / row["vote_se"]) if row["vote_se"] else float("inf")
        print(f"{row['k']:>3}{row['avg_se']:>9.2f}{a_mult:>8.2f}x"
              f"{row['vote_se']:>9.2f}{v_mult:>8.2f}x")
    if runs < 5:
        print("note: voting column uses a coarse p_hat at this --repeats; raise --repeats (>=5) to sharpen it.")

    print()
    print("=" * 72)
    print("Verdict")
    print("=" * 72)
    print(f"mean discordant pairs (noise floor) : {mean_discordant:.1f} of {args.num_examples}  ({100 * mean_flip:.1f}% of instances flip between identical runs)")
    print(f"mean |net| between identical runs   : {mean_abs_net:.2f}  (pure noise; a real edit must beat it)")
    print(f"noise standard error  sqrt(discord) : {noise_se:.2f} accuracy-points-equivalent")
    print(f"gate promotion bar  net > z*SE      : net must exceed {promo_bar:.2f} (z={args.gate_z}) to promote")
    print()
    if args.crn:
        # Under CRN the same harness should sample identically; whatever flips
        # remains is irreducible (batch nondeterminism, nondeterministic tools).
        print("[CRN mode] This is the RESIDUAL floor with shared per-instance seeds.")
        if mean_flip <= 0.08:
            print("=> CRN cancels most of the noise. Turn on paired_sampling_seed for the search:")
            print("   child-vs-parent flips will then reflect the edit, not sampling luck, and the")
            print("   gate can promote real wins. Compare this number to the non-CRN run.")
        elif mean_flip <= 0.15:
            print("=> CRN helps but a meaningful residual remains (batching/tool nondeterminism).")
            print("   Worth enabling, but pair it with a couple of repeats per instance for safety.")
        else:
            print("=> CRN barely helps — the floor is dominated by nondeterminism the seed can't")
            print("   control (vLLM continuous batching, nondeterministic tools). Pursue repeat-")
            print("   and-average instead, and/or batch-invariant serving settings.")
    elif promo_bar >= 4:
        print("=> High noise floor. A child needs a very large, repeatable net gain to clear")
        print("   the gate. Re-run with --crn to see how much a fixed per-instance seed cancels.")
    elif promo_bar >= 2:
        print("=> Moderate noise floor. Borderline real gains may be lost. Re-run with --crn to")
        print("   measure how much paired seeding recovers before loosening z.")
    else:
        print("=> Low noise floor. The gate should be able to promote genuine improvements;")
        print("   if children still gate out, the proposed edits likely aren't real wins.")

    # Does sample-per-instance stabilize the search?
    print()
    last = curve[-1]
    print(f"Sample-per-instance effect (k=1 -> k={last['k']}):")
    print(f"  AVERAGING: SE {se1:.2f} -> {last['avg_se']:.2f}  "
          f"({se1 / last['avg_se']:.1f}x more stable; keeps falling ~1/sqrt(k))")
    print(f"  VOTING:    SE {curve[0]['vote_se']:.2f} -> {last['vote_se']:.2f}  "
          f"(plateaus near {vote_floor:.2f}; the {n_partial} ambiguous instances stay ~50%)")
    if se1 > 0 and last["avg_se"] < 0.7 * se1:
        print("=> Yes: averaging over k samples meaningfully stabilizes the comparison.")
        print("   Voting saturates, so use pass-rate averaging (compose with CRN in the search).")
    else:
        print("=> Sampling barely helps here: the per-instance noise is already low, or the")
        print("   ambiguous mass is small. Re-check whether the floor is worth the k-fold cost.")

    out = {
        "agent_config": str(source_agent_config),
        "router_env_file": str(source_router_env),
        "datasets": list(args.datasets),
        "num_examples": args.num_examples,
        "repeats": args.repeats,
        "gate_z": args.gate_z,
        "crn": bool(args.crn),
        "per_run_accuracy": [
            {"run": name, "accuracy": sum(1 for v in c.values() if v) / len(c), "n": len(c)}
            for name, c in correctness_per_run
        ],
        "pairwise": pairwise,
        "mean_discordant": mean_discordant,
        "mean_abs_net": mean_abs_net,
        "mean_flip_rate": mean_flip,
        "noise_se": noise_se,
        "promotion_bar": promo_bar,
        "instances": n_instances,
        "passrate_distribution": {"always_right": n_always, "always_wrong": n_never, "ambiguous": n_partial},
        "samples_per_instance_curve": curve,
        "voting_se_floor": vote_floor,
    }
    out_path = work_dir / "eval_noise.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
