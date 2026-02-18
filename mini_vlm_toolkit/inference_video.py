import torch
import torch.distributed as dist
import copy as cp
from .utils.reporting_utils import (
    build_trajectory_records,
    write_trajectory_jsonl,
    write_html_reports,
)
from .utils import track_progress_rich
from .smp import *

FAIL_MSG = 'Failed to obtain answer via API.'


def _build_isolated_generate_func(model, api_nproc):
    """Return a generate callable with per-call model isolation for threaded inference."""
    def _isolated_generate(**kwargs):
        model_copy = cp.deepcopy(model)
        return model_copy.generate(**kwargs)

    return _isolated_generate

# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, samples_dict={}, api_nproc=4):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name

    indices = list(samples_dict.keys())
    if getattr(model,'backend', None) == 'genai':
        if dataset.nframe > 0:
            print(
                'Gemini model (with genai backend) does not support nframe, '
                'will set its VIDEO_LLM to False to enable multi-image input for video.'
            )
            setattr(model, 'VIDEO_LLM', False)
        else:
            print('Gemini model (with genai backend) is a video-llm, '
                  'will reset fps setting in model to match the dataset.')
            setattr(model, 'fps', dataset.fps)
            print(f'The fps is set to {dataset.fps} for the model {model_name}.')
    elif getattr(model,'backend', None) == 'vertex':
        print('Gemini model (with vertex backend) does not support video input, '
              'will set its VIDEO_LLM to False to enable multi-image input for video.')
        setattr(model, 'VIDEO_LLM', False)

    packstr = 'pack' if getattr(dataset, 'pack', False) else 'nopack'
    build_prompt_input = [(samples_dict[idx], getattr(model, 'VIDEO_LLM', False)) for idx in indices]
    if dataset.nframe > 0:
        struct_tmp_file = f'{work_dir}/{model_name}_{dataset.nframe}frame_{packstr}_structs.pkl'
    else:
        struct_tmp_file = f'{work_dir}/{model_name}_{dataset.fps}fps_{packstr}_structs.pkl'
    structs = track_progress_rich(
        dataset.build_prompt,
        tasks=build_prompt_input,
        nproc=api_nproc,
        save=struct_tmp_file,
        keys=indices,
    )

    if dataset.nframe > 0:
        out_file = f'{work_dir}/{model_name}_{dataset.nframe}frame_{packstr}_supp.pkl'
    else:
        out_file = f'{work_dir}/{model_name}_{dataset.fps}fps_{packstr}_supp.pkl'
    res = load(out_file) if osp.exists(out_file) else {}

    structs = [s for i, s in zip(indices, structs) if i not in res or res[i] == FAIL_MSG]
    structs = [struct for struct in structs if struct is not None]
    indices = [i for i in indices if i not in res or res[i] == FAIL_MSG]

    # Original behavior (shared agent across worker threads):
    # gen_func = model.generate
    # Use per-call deep-copied agent when api_nproc > 1 to avoid cross-task mutable-state races.
    gen_func = _build_isolated_generate_func(model, api_nproc)
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    return res


def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4):
    res = load(out_file) if osp.exists(out_file) else {}
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name

    sample_indices = list(dataset.videos) if getattr(dataset, 'pack', False) else list(dataset.data['index'])
    samples = list(dataset.videos) if getattr(dataset, 'pack', False) else list(range(len(dataset.data)))
    sample_map = {i: s for i, s in zip(sample_indices, samples)}

    sample_indices_sub = sample_indices[rank::world_size]
    if np.all([idx in res for idx in sample_indices_sub]):
        return model
    sample_indices_subrem = [x for x in sample_indices_sub if x not in res]

    # (25.06.05) In newer version of transformers (after 4.50), with device_map='auto' and torchrun launcher,
    # Transformers automatically adopt TP parallelism, which leads to compatibility problems with VLMEvalKit
    # (In VLMEvalKit, we use torchrun to launch multiple model instances on a single node).
    # To bypass this problem, we unset `WORLD_SIZE` before building the model to not use TP parallel.
    ws_bak = os.environ.pop('WORLD_SIZE', None)
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak

    assert world_size == 1
    supp = infer_data_api(
        model=model,
        work_dir=work_dir,
        model_name=model_name,
        dataset=dataset,
        samples_dict={k: sample_map[k] for k in sample_indices_subrem},
        api_nproc=api_nproc)
    for k in sample_indices_subrem:
        assert k in supp
    res.update(supp)
    dump(res, out_file)
    return model



# A wrapper for infer_data, do the pre & post processing
def infer_video_job(
        model,
        work_dir,
        model_name,
        dataset,
        dataset_name=None,
        verbose=False,
        api_nproc=4,
        html_limit=0,
    ):

    if dataset_name is None:
        dataset_name = dataset.dataset_name
    rank, world_size = get_rank_and_world_size()
    work_dir = osp.join(work_dir, dataset_name)
    if rank == 0 and not osp.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    result_file = get_pred_file_path(work_dir, model_name, dataset_name, use_env_format=True)
    # Dump Predictions to Prev File if result file exists
    if osp.exists(result_file):
        return model

    tmpl = osp.join(work_dir, '{}' + f'{world_size}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data(
        model=model,
        model_name=model_name,
        work_dir=work_dir,
        dataset=dataset,
        out_file=out_file,
        verbose=verbose,
        api_nproc=api_nproc)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        meta = dataset.data
        if dataset_name == 'MMBench-Video' and getattr(dataset, 'pack', False):
            meta, vstats = dataset.load_pack_answers(data_all)
            print(f'Statitics of Pack Video Inference: {vstats}')
        else:
            for x in meta['index']:
                assert x in data_all
            if all(isinstance(data_all[x], dict) and "prediction" in data_all[x] and "extra_records" in data_all[x] for x in meta['index']):
                meta['prediction'] = [data_all[x]['prediction'] for x in meta['index']]
                cleaned_extra = []
                for x in meta['index']:
                    extra = data_all[x]['extra_records']
                    if isinstance(extra, dict) and "trajectory" in extra:
                        extra = extra.copy()
                        extra.pop("trajectory", None)
                    cleaned_extra.append(extra)
                meta['extra_records'] = cleaned_extra
            else:
                meta['prediction'] = [str(data_all[x]) for x in meta['index']]
            if 'image' in meta:
                meta.pop('image')

        dump(meta, result_file)
        records = build_trajectory_records(
            meta["index"],
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
            html_records = build_trajectory_records(
                meta["index"],
                data_all,
                dataset_name=dataset_name,
                model_name=model_name,
                agent_class=model.__class__.__name__,
                scrub_images=False,
                strip_assistant_content=False,
            )
            write_html_reports(html_records, html_dir, html_limit)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    return model
