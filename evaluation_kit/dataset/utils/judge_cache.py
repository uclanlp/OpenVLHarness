"""Keep evaluator judge caches consistent with the judge and the predictions.

Evaluators cache judge verdicts and answer extractions next to the prediction
file, either as a dict keyed by row index or as a table that is reused when the
file exists. ``guard_cache`` runs before such a cache is read. It keeps a
sidecar ``<cache>.fingerprint.json`` with the judge model, the judge prompt and
a short hash of each prediction, and removes cached results that no longer
match: the whole cache when the judge changed, otherwise the rows whose
prediction changed.
"""
import hashlib
import os
import os.path as osp

from ...smp.file import dump, load
from ...smp.log import get_logger

FINGERPRINT_SUFFIX = '.fingerprint.json'


def fingerprint_path(cache_path):
    return cache_path + FINGERPRINT_SUFFIX


def prediction_hashes(data, key_col='index'):
    """Map ``str(key)`` to the first 16 hex characters of sha256(str(prediction))."""
    return {
        str(key): hashlib.sha256(str(pred).encode('utf-8', 'surrogatepass')).hexdigest()[:16]
        for key, pred in zip(data[key_col], data['prediction'])
    }


def guard_cache(cache_path, data, judge_model, prompt=None, keyed=True, key_col='index'):
    """Remove stale results from ``cache_path`` before an evaluator reads it.

    Args:
        cache_path: the cache file.
        data: the rows being scored; needs ``key_col`` and ``prediction`` columns.
        judge_model: judge model name from the judge config.
        prompt: name of the judge prompt, or None if the evaluator has only one.
        keyed: True for a pickled dict keyed by ``data[key_col]``; only the keys
            whose prediction changed are dropped. False for tables and for dicts
            with other keys (e.g. circular MCQ groups); any change deletes the file.
        key_col: column holding the cache keys.

    A cache that has no sidecar cannot be checked; it is kept and a warning is
    logged. The sidecar is always rewritten for the current run.
    """
    logger = get_logger('Evaluation')
    sidecar = fingerprint_path(cache_path)
    current = prediction_hashes(data, key_col)
    record = dict(
        judge_model=None if judge_model is None else str(judge_model),
        judge_prompt=prompt,
        predictions=current,
    )
    if osp.exists(cache_path):
        if not osp.exists(sidecar):
            logger.warning(
                f'{cache_path} has no fingerprint, so its judge results are reused as they are. '
                'Delete it to re-judge.'
            )
        else:
            old = load(sidecar)
            old_predictions = old.get('predictions', {})
            old_judge = (old.get('judge_model'), old.get('judge_prompt'))
            if old_judge != (record['judge_model'], prompt):
                logger.info(
                    f'Judge changed from {old_judge} to {(record["judge_model"], prompt)}; deleting {cache_path}'
                )
                os.remove(cache_path)
            elif keyed:
                cache = load(cache_path)
                stale = [
                    k for k in cache
                    if str(k) in current and old_predictions.get(str(k)) != current[str(k)]
                ]
                if stale:
                    logger.info(f'{len(stale)} predictions changed; dropping them from {cache_path}')
                    for k in stale:
                        del cache[k]
                    dump(cache, cache_path)
                # Keep fingerprints of rows not scored in this run, so their
                # cached results can still be checked when they come back.
                record['predictions'] = {**old_predictions, **current}
            elif old_predictions != current:
                logger.info(f'Predictions changed; deleting {cache_path}')
                os.remove(cache_path)
    dump(record, sidecar)

