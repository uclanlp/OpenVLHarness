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
