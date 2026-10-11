# Adapted from VLMEvalKit (https://github.com/open-compass/VLMEvalKit), Apache License 2.0; modified by the OpenVLHarness authors.
"""Inference runner: feeds benchmark rows to an agent and writes predictions, trajectories and reports."""

import copy
import itertools
import os
import warnings
import pandas as pd
from .utils.reporting_utils import (
    build_trajectory_records,
    write_input_snapshot_json,
    write_raw_response_jsons,
    write_raw_trace_jsons,
    build_complete_messages_from_trajectory,
    render_html_report,
    _replace_data_uris_with_files,
)
from .utils import track_progress_rich
from .utils import file_db
from .smp import *
from .smp.file import NumpyEncoder

# Constants
FAIL_MSG = 'Failed to obtain answer via API.'
# The agent records an exhausted request as MainAgent.fail_msg and a question that used every
# iteration as MainAgent.limit_msg; --ignore-failed must treat all of them as failed rows.
FAIL_MSGS = (FAIL_MSG, 'Error: Unable to get response from the server.',
             'Error: Reached the iteration limit without a final answer.')


def _is_failed_prediction(pred):
    text = str(pred)
    return any(m in text for m in FAIL_MSGS)


def _failure_cause(result, limit=300):
    """Return the error the agent recorded for a failed sample, shortened to ``limit`` characters."""
    extra = result.get("extra_records") if isinstance(result, dict) else None
    trajectory = extra.get("trajectory") if isinstance(extra, dict) else None
    if not (isinstance(trajectory, dict) and trajectory.get("error")):
        return "no error recorded"
    cause = str(trajectory["error"])
    return cause if len(cause) <= limit else cause[:limit] + "..."


# ============================================================================
# API-based Inference Functions
# ============================================================================

def _build_full_trajectory_payload(record, save_images, images_subdir):
    """Build the per-task ``full_trajectories/<idx>.json`` payload.

    Mirrors ``write_full_trajectory_jsons`` (reporting_utils.py:679-729) but
    returns the dict instead of writing it, so the caller can land it
    atomically alongside other commit-time sidecars.
    """
    trajectory = record.get("trajectory")
    payload = {
        "task_id": copy.deepcopy(record.get("task_id")),
        "dataset": copy.deepcopy(record.get("dataset")),
        "model_name": copy.deepcopy(record.get("model_name")),
        "agent_class": copy.deepcopy(record.get("agent_class")),
        "prediction": copy.deepcopy(record.get("prediction")),
    }
    gt = record.get("ground_truth")
    if gt is not None:
        payload["ground_truth"] = copy.deepcopy(gt)
    if isinstance(trajectory, dict) and trajectory.get("error"):
        payload["error"] = copy.deepcopy(trajectory["error"])
    reasoning = (record.get("extra_records") or {}).get("reasoning")
    if reasoning:
        payload["reasoning"] = reasoning
    usage = (record.get("extra_records") or {}).get("usage")
    if isinstance(usage, dict) and usage:
        payload["usage"] = copy.deepcopy(usage)
    complete_messages = build_complete_messages_from_trajectory(trajectory)
    if complete_messages is not None:
        payload["messages"] = complete_messages
    elif isinstance(trajectory, dict) and isinstance(trajectory.get("messages"), list):
        payload["messages"] = copy.deepcopy(trajectory["messages"])

    if save_images == "files" and payload.get("messages") is not None:
        os.makedirs(images_subdir, exist_ok=True)
        _replace_data_uris_with_files(
            payload["messages"], payload["task_id"], images_subdir,
        )
    return payload


def _make_committed_gen(
    original_gen,
    *,
    work_dir,
    dataset,
    dataset_name,
    model_name,
    agent_class,
    save_images,
    save_raw_trace,
    html_limit,
    full_trajectories_limit=-1,
):
    """Wrap ``model.generate`` with per-example claim → run → commit.

    Each successful invocation atomically writes:
      - ``results/<shard>/<idx>/record.json`` (canonical, with done flag)
      - ``full_trajectories/<idx>.json`` (gated by ``full_trajectories_limit``;
        -1 = write all (image default), N = first-N by completion order,
        0 = never)
      - ``tool_detections/<idx>.json`` (only if non-empty)
      - ``htmls/<idx>.html`` (first-N by completion order, gated by html_limit)
      - ``raw_traces/<idx>.json`` or ``raw_traces/errors/<idx>.json``
        (if raw export was requested)
    """
    # Disk escape hatch: full_trajectories/ (mostly saved images) is ~99.7% of run output.
    # FULL_TRAJECTORIES_LIMIT caps it: 0 = save none, N = first N per dataset, unset/-1 =
    # unlimited (default). Metrics are unaffected (they come from the xlsx/csv).
    _ftl = os.environ.get("FULL_TRAJECTORIES_LIMIT", "").strip()
    if _ftl:
        try:
            full_trajectories_limit = int(_ftl)
        except ValueError:
            pass
    file_db.ensure_layout(work_dir)
    gt_key = None
    for key in ("answer", "GT", "answers", "ground_truth"):
        if key in dataset.data:
            gt_key = key
            break
    answer_map = dict(zip(dataset.data["index"], dataset.data[gt_key])) if gt_key else {}

    html_committed = itertools.count()
    full_committed = itertools.count()
    images_subdir = osp.join(work_dir, "full_trajectories", "images")

    def gen(message=None, dataset=None, task_id=None, **kwargs):
        idx = task_id
        with file_db.claim_and_heartbeat(work_dir, idx) as got:
            if not got:
                return None
            result = original_gen(message=message, dataset=dataset, task_id=task_id, **kwargs)

            records = build_trajectory_records(
                [idx], {idx: result},
                dataset_name=dataset_name, model_name=model_name,
                agent_class=agent_class, scrub_images=False,
                strip_assistant_content=False,
            )
            if records:
                rec0 = records[0]
                gt = answer_map.get(idx)
                if gt is not None:
                    rec0["ground_truth"] = str(gt)

                # 1) full_trajectories/<idx>.json — first-N gating
                if full_trajectories_limit != 0:
                    if full_trajectories_limit < 0 or next(full_committed) < full_trajectories_limit:
                        payload = _build_full_trajectory_payload(rec0, save_images, images_subdir)
                        file_db.atomic_write(
                            osp.join(work_dir, "full_trajectories", f"{idx}.json"),
                            payload, json_encoder=NumpyEncoder,
                        )

                # 2) tool_detections/<idx>.json — only if non-empty
                extra = rec0.get("extra_records") or {}
                detections = extra.get("tool_detections") if isinstance(extra, dict) else None
                if detections:
                    file_db.atomic_write(
                        osp.join(work_dir, "tool_detections", f"{idx}.json"),
                        detections, json_encoder=NumpyEncoder,
                    )

                # 3) htmls/<idx>.html — first-N gating
                if html_limit != 0:
                    if html_limit < 0 or next(html_committed) < html_limit:
                        html_text = render_html_report(rec0)
                        file_db.atomic_write(
                            osp.join(work_dir, "htmls", f"{idx}.html"),
                            html_text,
                        )

                # 4) Requested raw export, including explicit per-sample failures.
                if save_raw_trace:
                    write_raw_trace_jsons(
                        [rec0], osp.join(work_dir, "raw_traces"), limit=-1,
                    )

            prediction = result.get("prediction") if isinstance(result, dict) else result
            if _is_failed_prediction(prediction):
                get_logger("RUN").warning("Sample %s failed: %s", idx, _failure_cause(result))

            # 5) canonical record + done flag — last, so a crash before this
            # leaves no done flag and the boot scan re-dispatches.
            file_db.commit_atomic(work_dir, idx, result, json_encoder=NumpyEncoder)
            return result

    return gen


def infer_data_api(
    model, work_dir, model_name, dataset, *,
    index_set=None, api_nproc=4, ignore_failed=False,
    enable_incremental_json=False,
    html_limit=0, save_images="files", save_raw_trace=False,
):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    file_db.ensure_layout(work_dir)

    # MMBench_V11 reuse: pre-seed sidecars from the cached predictions.
    if dataset_name in ['MMBench', 'MMBench_CN']:
        pred_format = get_pred_file_format()
        v11_pred = f'{work_dir}/{model_name}_V11.{pred_format}'
        if osp.exists(v11_pred):
            try:
                reuse_inds = load('http://opencompass.openxlab.space/utils/mmb_reuse.pkl')
                v11_data = load(v11_pred)
                ans_map = {
                    x: y for x, y in zip(v11_data['index'], v11_data['prediction'])
                    if x in reuse_inds
                }
                for k, v in ans_map.items():
                    if str(k) not in file_db.load_done_set(work_dir):
                        file_db.commit_atomic(work_dir, k, v)
            except Exception as err:
                print(type(err), err)

    # Boot scan: skip already-committed and live-claimed examples.
    done = file_db.load_done_set(work_dir)
    fresh = file_db.load_fresh_claims(work_dir)
    if ignore_failed:
        # Re-run examples whose committed prediction is FAIL_MSG.
        for idx_str in list(done):
            try:
                rec = file_db.load_record(work_dir, idx_str)
            except (FileNotFoundError, ValueError):
                continue
            pred = rec.get("prediction") if isinstance(rec, dict) else rec
            if _is_failed_prediction(pred):
                done.discard(idx_str)
                try:
                    os.unlink(file_db._done_path(work_dir, idx_str))
                except FileNotFoundError:
                    pass

    skip = done | fresh
    structs = []
    indices_to_run = []
    for i in range(len(data)):
        item = data.iloc[i]
        idx = item['index']
        if str(idx) in skip:
            continue
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)
        indices_to_run.append(idx)

    structs = [
        dict(message=s, dataset=dataset_name, task_id=idx)
        for s, idx in zip(structs, indices_to_run)
    ]

    if structs:
        agent_class = model.__class__.__name__
        committed_gen = _make_committed_gen(
            model.generate,
            work_dir=work_dir, dataset=dataset, dataset_name=dataset_name,
            model_name=model_name, agent_class=agent_class,
            save_images=save_images, save_raw_trace=save_raw_trace,
            html_limit=html_limit,
        )
        incremental_json = (
            osp.join(work_dir, f'{model_name}_incremental.jsonl')
            if enable_incremental_json else None
        )
        track_progress_rich(
            committed_gen, structs, nproc=api_nproc, chunksize=api_nproc,
            save=None, keys=indices_to_run,
            incremental_json_file=incremental_json,
        )

    # Build the result dict from per-example sidecars.
    idx_lookup = {str(x): x for x in dataset.data['index']}
    res = {}
    for idx_str, record in file_db.iter_records(work_dir):
        native_idx = idx_lookup.get(idx_str, idx_str)
        res[native_idx] = record
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    return res


def infer_data(
    model, model_name, work_dir, dataset, *,
    verbose=False, api_nproc=4, enable_incremental_json=False,
    html_limit=0, save_images="files", save_raw_trace=False,
    ignore_failed=False,
):
    """Run inference for ``dataset`` under ``work_dir``.

    Single-rank only (the inner ``infer_data_api`` asserts world_size==1).
    Resume is driven by ``_done/`` flags + per-example sidecars under
    ``work_dir`` rather than the legacy ``_supp.pkl`` / ``_PREV.pkl``
    cumulative pickles.
    """
    rank, world_size = get_rank_and_world_size()
    indices = list(dataset.data['index'])
    infer_data_api(
        model=model, work_dir=work_dir, model_name=model_name,
        dataset=dataset, index_set=set(indices), api_nproc=api_nproc,
        ignore_failed=ignore_failed,
        enable_incremental_json=enable_incremental_json,
        html_limit=html_limit, save_images=save_images,
        save_raw_trace=save_raw_trace,
    )
    return model


# ============================================================================
# Helper Functions
# ============================================================================

def _is_structured_record(v):
    """Check if a value is a structured result record."""
    return isinstance(v, dict) and 'prediction' in v and 'extra_records' in v


def _strip_trajectory(extra_records):
    if isinstance(extra_records, dict) and "trajectory" in extra_records:
        cleaned = extra_records.copy()
        cleaned.pop("trajectory", None)
        return cleaned
    return extra_records


def _split_thinking(s):
    """
    Extract thinking and prediction from model output with <think> tags.
    
    Returns:
        Tuple of (prediction, thinking)
    """
    if '</think>' in s:
        splits = s.split('</think>')
        prediction = splits[-1].strip()
        if len(splits) == 2 and '<think>' in splits[0]:
            thinking = splits[0].split('<think>')[1].strip()
        else:
            thinking = '</think>'.join(splits[:-1])
            thinking += '</think>'
            warnings.warn('Failed to parse thinking, multiple </think> tags or missing <think> tag.')
    else:
        thinking = ''
        prediction = s
    return (prediction, thinking)


def _process_results_with_thinking(data, data_all, model):
    """
    Process results with thinking tags separation.
    
    Returns:
        Processed data with prediction and thinking columns
    """
    if all(_is_structured_record(data_all[x]) for x in data['index']):
        prediction = [data_all[x]['prediction'] for x in data['index']]
        extra_records = [_strip_trajectory(data_all[x]['extra_records']) for x in data['index']]
        data['extra_records'] = extra_records
    else:
        prediction = [str(data_all[x]) for x in data['index']]
    
    split_func = model.split_thinking if hasattr(model, 'split_thinking') else _split_thinking
    tups = [split_func(x) for x in prediction]
    data['prediction'] = [x[0] for x in tups]
    data['thinking'] = [x[1] for x in tups]
    return data


def _process_results_default(data, data_all):
    """
    Process results without thinking tags separation.
    
    Returns:
        Processed data with prediction column
    """
    if all(_is_structured_record(data_all[x]) for x in data['index']):
        data['prediction'] = [data_all[x]['prediction'] for x in data['index']]
        data['extra_records'] = [_strip_trajectory(data_all[x]['extra_records']) for x in data['index']]
    else:
        data['prediction'] = [str(data_all[x]) for x in data['index']]
    return data


# ============================================================================
# Main Inference Job
# ============================================================================

def infer_image_job(
    model,
    work_dir,
    model_name,
    dataset,
    verbose=False,
    api_nproc=4,
    ignore_failed=False,
    html_limit=0,
    enable_incremental_json=False,
    use_dataset_subdir=True,
    save_images="files",
    save_raw_trace=False,
):
    """
    Main inference job wrapper for image datasets with pre/post processing.
    
    Args:
        model: The model with generate() method
        work_dir: Working directory for output files
        model_name: Name of the model
        dataset: Dataset object
        verbose: Whether to print verbose output
        api_nproc: Number of processes for API calls
        ignore_failed: Whether to ignore failed results
        enable_incremental_json: Whether to enable incremental JSONL output
    
    Returns:
        The model object
    """
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    if use_dataset_subdir:
        work_dir = osp.join(work_dir, dataset_name)
    if rank == 0 and not osp.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)
    pred_dataset = dataset_name if use_dataset_subdir else None
    result_file = get_pred_file_path(work_dir, model_name, pred_dataset, use_env_format=True)

    if osp.exists(result_file) and not ignore_failed:
        return model  # already done, skip entirely

    # --ignore-failed: seed _done/ flags + sidecars for non-failed rows from
    # the existing xlsx so the inference loop only re-runs failures.
    # Replaces the legacy _PREV.pkl seed.
    if osp.exists(result_file) and ignore_failed and rank == 0:
        _seed_done_from_xlsx(work_dir, result_file)

    # ========== Inference (per-example sidecars) ==========
    infer_data(
        model=model, model_name=model_name, work_dir=work_dir, dataset=dataset,
        verbose=verbose, api_nproc=api_nproc,
        enable_incremental_json=enable_incremental_json,
        html_limit=html_limit, save_images=save_images,
        save_raw_trace=save_raw_trace, ignore_failed=ignore_failed,
    )

    # ========== Aggregation ==========
    if rank == 0:
        idx_lookup = {str(x): x for x in dataset.data['index']}
        data_all = {
            idx_lookup.get(idx_str, idx_str): record
            for idx_str, record in file_db.iter_records(work_dir)
        }

        data = dataset.data
        for x in data['index']:
            assert x in data_all, f"missing committed record for idx={x!r}"

        if os.getenv('SPLIT_THINK', False):
            data = _process_results_with_thinking(data, data_all, model)
            print('Prediction format: SPLIT_THINK, splitting func enabled')
        else:
            data = _process_results_default(data, data_all)

        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        write_input_snapshot_json(
            data["index"], data_all,
            osp.join(work_dir, "input_snapshot.json"),
        )

        # raw_response is opt-in per agent. The other per-task artifacts
        # (full_trajectories/, tool_detections/, htmls/, raw_traces/)
        # are already written at commit time inside the worker wrapper.
        if getattr(model, 'save_raw_response', False):
            full_records = build_trajectory_records(
                data["index"], data_all,
                dataset_name=dataset_name, model_name=model_name,
                agent_class=model.__class__.__name__,
                scrub_images=False, strip_assistant_content=False,
            )
            write_raw_response_jsons(full_records, osp.join(work_dir, "raw_response"))
    return model


def _seed_done_from_xlsx(work_dir, result_file):
    """Walk an existing prediction xlsx and emit per-example sidecars +
    ``_done/`` flags for rows whose prediction is not ``FAIL_MSG``. Used by
    the ``--ignore-failed`` boot path; replaces the legacy ``_PREV.pkl``
    seed so the worker loop only re-runs the failures."""
    file_db.ensure_layout(work_dir)
    data = load(result_file)
    if isinstance(data, list):
        data = pd.DataFrame(data)
    has_extra = 'extra_records' in data
    already_done = file_db.load_done_set(work_dir)
    seeded = 0
    for i in range(len(data)):
        row = data.iloc[i]
        idx = row['index']
        prediction = row['prediction']
        if _is_failed_prediction(prediction):
            continue
        if str(idx) in already_done:
            continue
        if has_extra:
            record = {"prediction": prediction, "extra_records": row['extra_records']}
        else:
            record = prediction
        file_db.commit_atomic(work_dir, idx, record, json_encoder=NumpyEncoder)
        seeded += 1
    if seeded:
        print(f"[ignore-failed seed] re-imported {seeded} non-failed rows from {result_file}")
