"""Auto-download of the benchmarks backed up on HuggingFace.

The processed TSVs (prompts + annotations), the images they reference and the
ODinW-13 ground-truth files live in one dataset repo, ``HF_REPO_ID``::

    <split>.tsv                 one file per registered split
    images/<folder>.zip         images for every split sharing img_root_map folder
    GLIP.zip                    ODinW-13 configs + val/test COCO annotations

A missing TSV is fetched by ``ImageBaseDataset.load_data``; missing images are
fetched and unpacked into ``$LMUData/images/<folder>/`` by ``ensure_hf_images``;
the ODinW GT tree is unpacked into ``$LMUData/GLIP`` by ``ensure_hf_glip``.

Counting / grounding splits are built by ``scripts/build_hf_release.py``; the general-VQA
and spatial splits by ``scripts/backup_eval_datasets.py``. Datasets whose TSV embeds its
images only need their DATASET_URL pointed at ``hf_tsv_url``.
"""

import base64
import hashlib
import json
import logging
import os
import os.path as osp
import zipfile

logger = logging.getLogger(__name__)

HF_REPO_ID = 'uclanlp/OpenVLHarness-Evaluation-Datasets'

# split name -> images/<folder>.zip (the split's img_root_map folder)
HF_SPLITS = {
    'FSC147_test': 'FSC147',
    'FSC147_val': 'FSC147',
    'FSCD_147_bbox_test': 'FSC147',
    'FSCD_147_bbox_val': 'FSC147',
    'FSCD_LVIS_bbox_test': 'LVIS',
    'FSCD_LVIS_bbox_val': 'LVIS',
    'ODinW13_bbox_test_qwen3': 'ODinW',
    'ODinW13_bbox_val_qwen3': 'ODinW',
    'FREAK_MCQ': 'FREAK',
    'FREAK_VQA': 'FREAK',
    'MindCubeBench_tiny_raw_qa': 'MindCube',
    'MindCubeBench_raw_qa': 'MindCube',
    'OmniSpatialBench': 'OmniSpatial',
    'OmniSpatialBench_default': 'OmniSpatial',
    'OmniSpatialBench_zeroshot_cot': 'OmniSpatial',
    'OmniSpatialBench_manual_cot': 'OmniSpatial',
    # multimodal search and general VQA (exactly the questions we evaluate on)
    'FVQA_test': 'FVQA',
    'Infoseek_GDM_test': 'Infoseek_GDM',
    'EncyclopedicVQA_test': 'EncyclopedicVQA',
    'FVQA_test_300': 'FVQA',
    'Infoseek_GDM_test_300': 'Infoseek_GDM',
    'EncyclopedicVQA_test_300': 'EncyclopedicVQA',
    'MMSearch': 'MMSearch',
    'MMSearchPlus': 'MMSearchPlus',
    'hle': 'HLE',
    'BrowseCompVL': 'BrowseCompVL',
    'RealXBench': 'RealXBench',
    'MMBrowseComp': 'MMBrowseComp',
    'VDRBench': 'VDRBench',
    'VisBrowseBench': 'VisBrowseBench',
    'SimpleVQA': 'SimpleVQA',
    'SimpleVQA_EN': 'SimpleVQA',
    'WorldVQA_300': 'WorldVQA',
}

# MD5 of each hosted TSV (printed by build_hf_release).
HF_DATASET_MD5 = {
    'FSC147_test': '1e04e1abad214ede87fadea333d16c1b',
    'FSC147_val': '1a1ab15000877cad4d80652da61b2e04',
    'FSCD_147_bbox_test': '5314179f5e6ecacf0c92097f1ffd9f25',
    'FSCD_147_bbox_val': 'cda58fb233fb04b8c060cb3b56f8f79c',
    'FSCD_LVIS_bbox_test': '3d5fd8e77d74dd40453333f5d4f4b5c8',
    'FSCD_LVIS_bbox_val': 'eedde50f9768d19053f62c3a3e4904c5',
    'ODinW13_bbox_test_qwen3': '20b51865305a6ae817d338db096ca278',
    'ODinW13_bbox_val_qwen3': '55ae98c3e4ffa1c3292f4939f1b52012',
    'FREAK_MCQ': '92a97ca0d72e0c47859e865968ddeb42',
    'FREAK_VQA': '3fa7094fad677caf2225a2fea83f1cd0',
    'FVQA_test': '74c81f819890526042102701cb538c77',
    'Infoseek_GDM_test': 'e1aedfd8d6ac8b0af665ed611cf3cc73',
    'EncyclopedicVQA_test': 'defa1c8d92362a9895a661e048af0587',
    'FVQA_test_300': '7655e2cdd2da05965ad64a2d8674fe86',
    'Infoseek_GDM_test_300': '859765d5967e4030853284bb79019fc8',
    'EncyclopedicVQA_test_300': 'dcae0e8fafc034588fc69dee08cad775',
    'MMSearch': '7b69f9c912f32a668228a943f1cb06ef',
    'MMSearchPlus': '042d455af098d0b556307c6c63360c46',
    'hle': 'bf6120b926904d7752f0ff2eca88636b',
    'BrowseCompVL': '0899409decc7a20472633a0f5cf64a47',
    'RealXBench': 'e8485e9b5b8261dd317ed45c038e54c2',
    'MMBrowseComp': 'c7c78ee9fbd7c0321e9f1cba4b29e109',
    'VDRBench': '887e364a005cad5479813872d0350164',
    'VisBrowseBench': '08ee00f73d9bf0d06629ac3e7258b549',
    'SimpleVQA': '87a55db6b42507a80f11e8e1c84c190f',
    'SimpleVQA_EN': '198c569fd852a95cb0fd830cf6e9d650',
    'WorldVQA_300': 'adaf33776d1b16f77782032d989a2a36',
}


def hf_tsv_url(dataset):
    return f'https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{dataset}.tsv'


def _download_and_unpack(filename, dest_dir):
    from filelock import FileLock
    from huggingface_hub import hf_hub_download

    os.makedirs(dest_dir, exist_ok=True)
    with FileLock(osp.join(dest_dir, f'.{osp.basename(filename)}.lock')):
        logger.info('Downloading %s from %s', filename, HF_REPO_ID)
        zip_path = hf_hub_download(HF_REPO_ID, filename, repo_type='dataset')
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)


def ensure_hf_images(dataset, img_root, image_paths):
    """Fetch ``images/<folder>.zip`` if any of ``image_paths`` is missing."""
    folder = HF_SPLITS.get(dataset)
    if folder is None:
        return
    missing = [p for p in image_paths if not osp.exists(osp.join(img_root, p))]
    if not missing:
        return
    logger.info('%s: %d images missing under %s', dataset, len(missing), img_root)
    _download_and_unpack(f'images/{folder}.zip', osp.dirname(img_root))
    still = [p for p in missing if not osp.exists(osp.join(img_root, p))]
    if still:
        raise FileNotFoundError(
            f'{dataset}: {len(still)} images still missing after download, '
            f'e.g. {osp.join(img_root, still[0])}')


def ensure_hf_glip(lmu_root):
    """Unpack the ODinW-13 GT tree to ``<lmu_root>/GLIP`` and return it."""
    glip_root = osp.join(lmu_root, 'GLIP')
    if not osp.isdir(glip_root):
        _download_and_unpack('GLIP.zip', lmu_root)
    return glip_root


# Splits whose questions and answers are hosted encrypted, as in their official releases
# (MMSearch-Plus, MM-BrowseComp, VisBrowse-Bench). Same scheme and passwords as upstream:
# base64 of the UTF-8 text XOR-ed with a SHA-256-derived key; the password is the row's
# ``canary``. Each entry lists the encrypted text columns and the encrypted list columns
# (JSON lists of ciphertexts) with the in-memory form the dataset class expects:
# 'repr' -> str(list), 'json' -> json.dumps(list), 'list' -> a Python list.
HF_ENCRYPTED = {
    'MMSearchPlus': (('question', 'answer'), {'alternative_gt_answers': 'repr'}),
    'MMBrowseComp': (('question', 'answer'), {'checklist': 'list'}),
    'VisBrowseBench': (('question', 'answer'), {'ground_truth_list': 'json'}),
}


def _derive_key(password, length):
    key = hashlib.sha256(password.encode()).digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt_text(ciphertext_b64, password):
    data = base64.b64decode(ciphertext_b64)
    return bytes(a ^ b for a, b in zip(data, _derive_key(password, len(data)))).decode()


def decrypt_hf_split(dataset, data):
    """Decrypt the encrypted columns of a hosted split in place; no-op for plain splits."""
    if dataset not in HF_ENCRYPTED:
        return data
    text_cols, list_cols = HF_ENCRYPTED[dataset]
    passwords = [str(x) for x in data['canary']]
    for col in text_cols:
        data[col] = [decrypt_text(str(v), pw) for v, pw in zip(data[col], passwords)]
    for col, form in list_cols.items():
        values = [[decrypt_text(x, pw) for x in json.loads(str(v))]
                  for v, pw in zip(data[col], passwords)]
        if form == 'repr':
            values = [str(v) for v in values]
        elif form == 'json':
            values = [json.dumps(v) for v in values]
        data[col] = values
    return data.drop(columns=['canary'])


def _absolute_image_paths(value, img_root):
    """Hosted ``image_path`` holds file names relative to ``images/<folder>/``."""
    value = str(value)
    if value.startswith('['):
        return [osp.join(img_root, x) for x in json.loads(value)]
    return osp.join(img_root, value)


def load_hf_split(dataset_obj, dataset):
    """Load a hosted split for ``dataset_obj``: TSV (MD5-checked), decryption and image paths.

    Images are fetched afterwards by ``ImageBaseDataset.__init__`` via ``ensure_hf_images``.
    """
    data = dataset_obj.prepare_tsv(hf_tsv_url(dataset), HF_DATASET_MD5.get(dataset))
    data = decrypt_hf_split(dataset, data)
    data['image_path'] = [_absolute_image_paths(v, dataset_obj.img_root) for v in data['image_path']]
    return data

