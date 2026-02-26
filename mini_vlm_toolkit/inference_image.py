"""Image inference utilities for multi-modal models."""

import os
import warnings
import torch
import torch.distributed as dist
from .utils.reporting_utils import (
    build_trajectory_records,
    write_trajectory_jsonl,
    write_input_snapshot_json,
    write_html_reports,
    write_full_trajectory_jsons,
)
from .utils import track_progress_rich
from .smp import *

# Constants
FAIL_MSG = 'Failed to obtain answer via API.'


# ============================================================================
# API-based Inference Functions
# ============================================================================

def infer_data_api(model, work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False, enable_incremental_json=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    out_file = f'{work_dir}/{model_name}_supp.pkl'

    # To reuse records in MMBench_V11
    if dataset_name in ['MMBench', 'MMBench_CN']:
        pred_format = get_pred_file_format()
        v11_pred = f'{work_dir}/{model_name}_V11.{pred_format}'
        if osp.exists(v11_pred):
            try:
                reuse_inds = load('http://opencompass.openxlab.space/utils/mmb_reuse.pkl')
                data = load(v11_pred)
                ans_map = {x: y for x, y in zip(data['index'], data['prediction']) if x in reuse_inds}
                dump(ans_map, out_file)
            except Exception as err:
                print(type(err), err)

    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    # Shared-agent threaded inference is safe after stateless-agent refactor.
    gen_func = model.generate
    # Legacy fallback kept for reference only:
    # gen_func = _build_isolated_generate_func(model)
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        if enable_incremental_json:
            incremental_json = osp.join(work_dir, f'{model_name}_incremental.jsonl')
            track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices, incremental_json_file=incremental_json)
        else:
            track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res


def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4, enable_incremental_json=False):
    """
    Run inference on dataset with batching across ranks.
    
    Args:
        model: The model with generate() method
        model_name: Name of the model
        work_dir: Working directory for output files
        dataset: Dataset object
        out_file: Output file for results
        verbose: Whether to print verbose output
        api_nproc: Number of processes for API calls
        enable_incremental_json: Whether to enable incremental JSONL output
    
    Returns:
        The model object
    """
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    ws_bak = os.environ.pop('WORLD_SIZE', None)
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak

    lt, indices = len(data), list(data['index'])
    supp = infer_data_api(
        model=model,
        work_dir=work_dir,
        model_name=model_name,
        dataset=dataset,
        index_set=set(indices),
        api_nproc=api_nproc,
        enable_incremental_json=enable_incremental_json)
    for idx in indices:
        assert idx in supp
    res.update(supp)
    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
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


def _build_prompts(model, dataset, dataset_name, data):
    """
    Build prompts for a dataset.
    
    Returns:
        List of prompt structures
    """
    structs = []
    for i in range(len(data)):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)
    return structs


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
    # set dataset_len = 16 for debugging
    work_dir = osp.join(work_dir, dataset_name)
    if rank == 0 and not osp.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    result_file = get_pred_file_path(work_dir, model_name, dataset_name, use_env_format=True)

    # ========== Pre-processing ==========
    prev_file = f'{work_dir}/{model_name}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    # ========== Distributed Inference ==========
    tmpl = osp.join(work_dir, '{}' + f'{world_size}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data(
        model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
        out_file=out_file, verbose=verbose, api_nproc=api_nproc,
        enable_incremental_json=enable_incremental_json)
    if world_size > 1:
        dist.barrier()

    # ========== Post-processing ==========
    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        
        # Process results with thinking tags if enabled
        if os.getenv('SPLIT_THINK', False):
            data = _process_results_with_thinking(data, data_all, model)
            print(f'Prediction format: SPLIT_THINK, splitting func enabled')
        else:
            data = _process_results_default(data, data_all)
        
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        write_input_snapshot_json(
            data["index"],
            data_all,
            osp.join(work_dir, "input_snapshot.json"),
        )

        records = build_trajectory_records(
            data["index"],
            data_all,
            dataset_name=dataset_name,
            model_name=model_name,
            agent_class=model.__class__.__name__,
            scrub_images=True,
            strip_assistant_content=True,
        )
        trajectories_path = osp.join(work_dir, "trajectories.jsonl")
        write_trajectory_jsonl(records, trajectories_path)

        if html_limit != 0:
            html_dir = osp.join(work_dir, "htmls")
            full_trajectory_dir = osp.join(work_dir, "full_trajectories")
            html_records = build_trajectory_records(
                data["index"],
                data_all,
                dataset_name=dataset_name,
                model_name=model_name,
                agent_class=model.__class__.__name__,
                scrub_images=False,
                strip_assistant_content=False,
            )
            write_html_reports(html_records, html_dir, html_limit)
            write_full_trajectory_jsons(html_records, full_trajectory_dir, html_limit)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    
    if world_size > 1:
        dist.barrier()
    return model
