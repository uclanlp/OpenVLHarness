# Adapted from VLMEvalKit (https://github.com/open-compass/VLMEvalKit), Apache License 2.0; modified by the OpenVLHarness authors.
import json
import threading
from pathlib import Path
from typing import Callable, Iterable, Optional

import os.path as osp
from ..smp import load, dump
from ..smp.file import NumpyEncoder

_incremental_json_lock = threading.Lock()


def _extract_prediction(result) -> str:
    """Extract prediction string from model output (str or dict with 'prediction' key)."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict) and 'prediction' in result:
        return result['prediction']
    return str(result)


def cpu_count():
    # Handle K8s LXCFS setting.
    period = Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
    quota = Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    try:
        if period.exists() and quota.exists():
            return int(quota.read_text()) // int(period.read_text())
    except Exception:
        pass

    import os
    return os.cpu_count()


def new_func(idx: int, func, inputs):
    if not isinstance(inputs, (tuple, list, dict)):
        inputs = (inputs, )
    if isinstance(inputs, dict):
        return idx, func(**inputs)
    else:
        return idx, func(*inputs)


def track_progress_rich(
        func: Callable,
        tasks: Iterable = tuple(),
        nproc: int | None = 1,
        save=None,
        keys=None,
        use_process: bool = False,
        incremental_json_file: Optional[str] = None,
        incremental_record_builder: Optional[Callable] = None,
        **kwargs) -> list:
    """
    incremental_json_file: when set (and PRED_FORMAT=json), append one JSON object per
        line as each result arrives, so you can tail -f and analyze results in real time.
    incremental_record_builder: optional (key, result) -> dict; default uses
        {"index": key, "prediction": <extracted>}.
    """
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    if save is not None:
        assert osp.exists(osp.dirname(save)) or osp.dirname(save) == ''
        if not osp.exists(save):
            dump({}, save)
    if keys is not None:
        assert len(keys) == len(tasks)
    if not callable(func):
        raise TypeError('func must be a callable object')
    if not isinstance(tasks, Iterable):
        raise TypeError(
            f'tasks must be an iterable object, but got {type(tasks)}')
    assert nproc is None or nproc > 0, 'nproc must be a positive number'
    nproc = nproc or cpu_count()
    res = load(save) if save is not None else {}
    results = [None for _ in range(len(tasks))]

    def _append_incremental_json(key, result):
        if incremental_record_builder is not None:
            record = incremental_record_builder(key, result)
        else:
            record = {"index": key, "prediction": _extract_prediction(result)}
        line = json.dumps(record, ensure_ascii=False, cls=NumpyEncoder) + "\n"
        with _incremental_json_lock:
            with open(incremental_json_file, "a", encoding="utf-8") as f:
                f.write(line)

    executor_type = ProcessPoolExecutor if use_process else ThreadPoolExecutor
    with executor_type(max_workers=nproc) as executor:
        futures = []

        for idx, inputs in enumerate(tasks):
            future = executor.submit(new_func, idx, func, inputs)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures)):
            idx, result = future.result()
            results[idx] = result
            if keys is not None and save is not None:
                # Legacy cumulative-pickle path. New per-example sidecar
                # callers pass save=None and persist inside `func` itself.
                res[keys[idx]] = result
                dump(res, save)
            if incremental_json_file is not None and keys is not None:
                _append_incremental_json(keys[idx], result)

    if save is not None:
        dump(res, save)
    return results
