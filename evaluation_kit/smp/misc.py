# flake8: noqa: F401, F403
import abc
import argparse
import ast
import csv
import multiprocessing as mp
import os
import os.path as osp
from pathlib import Path
import copy as cp
import random as rd
import requests
import shutil
import string
import subprocess
import warnings
import pandas as pd
from collections import OrderedDict, defaultdict
from multiprocessing import Pool, current_process
from tqdm import tqdm
import datetime
from huggingface_hub import scan_cache_dir
from huggingface_hub.utils._cache_manager import _scan_cached_repo


def modelscope_flag_set():
    return os.environ.get('VLMEVALKIT_USE_MODELSCOPE', None) in ['1', 'True']


def process_punctuation(inText):
    import re
    outText = inText
    punct = [
        ';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
        '>', '<', '@', '`', ',', '?', '!'
    ]
    commaStrip  = re.compile(r'(\d)(,)(\d)')
    periodStrip = re.compile(r'(?<!\d)\.(?!\d)')
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(
                commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText

def istype(s, type):
    if isinstance(s, type):
        return True
    try:
        return isinstance(ast.literal_eval(s), type)
    except Exception as _:
        return False

def get_cache_path(repo_id, branch='main', repo_type='datasets'):
    try:
        if modelscope_flag_set():
            from modelscope.hub.file_download import create_temporary_directory_and_cache
            if repo_type == 'datasets':
                repo_type = 'dataset'
            _, cache = create_temporary_directory_and_cache(model_id=repo_id, repo_type=repo_type)
            cache_path = cache.get_root_location()
            return cache_path
        else:
            from .file import HFCacheRoot
            cache_path = HFCacheRoot()
            org, repo_name = repo_id.split('/')
            repo_path = Path(osp.join(cache_path, f'{repo_type}--{org}--{repo_name}/'))
            hf_cache_info = _scan_cached_repo(repo_path=repo_path)
            revs = {r.refs: r for r in hf_cache_info.revisions}
            if branch is not None:
                revs = {refs: r for refs, r in revs.items() if branch in refs}
            rev2keep = max(revs.values(), key=lambda r: r.last_modified)
            return str(rev2keep.snapshot_path)
    except Exception as e:
        import logging
        logging.warning(f'{type(e)}: {e}')
        return None

def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size

def listinstr(lst, s):
    assert isinstance(lst, list)
    for item in lst:
        if item in s:
            return True
    return False

def d2df(D):
    return pd.DataFrame({x: [D[x]] for x in D})

def cn_string(s):
    import re
    if re.search(u'[\u4e00-\u9fff]', s):
        return True
    return False

def timestr(granularity='second'):
    s = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    assert granularity in ['second', 'minute', 'hour', 'day']
    if granularity == 'second':
        return s
    elif granularity == 'minute':
        return s[:-2]
    elif granularity == 'hour':
        return s[:-4]
    elif granularity == 'day':
        return s[:-6]

def dict_merge(dct, merge_dct):
    for k, _ in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict)):  #noqa
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


def toliststr(s):
    if isinstance(s, str) and (s[0] == '[') and (s[-1] == ']'):
        return [str(x) for x in ast.literal_eval(s)]
    elif isinstance(s, str):
        return [s]
    elif isinstance(s, list):
        return [str(x) for x in s]
    raise NotImplementedError
