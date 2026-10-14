# Adapted from VLMEvalKit (https://github.com/open-compass/VLMEvalKit), Apache License 2.0; modified by the OpenVLHarness authors.
"""Benchmark registry for the OpenVLHarness evaluation layer.

Registers the loaders for the paper's benchmarks together with the VLMEvalKit loaders
they inherit from, so ``SUPPORTED_DATASETS`` also contains dataset names that are not in
the paper; those names are untested. Each class lists the dataset names it serves
through ``supported_datasets()``; ``build_dataset`` maps a name to an instance and falls
back to a custom TSV under ``$LMUData`` when the name is unknown.
"""
import os.path as osp
import warnings

from ..smp import LMUDataRoot, load
from .image_base import img_root_map, ImageBaseDataset
from .image_yorn import ImageYORNDataset
from .image_mcq import ImageMCQDataset, CustomMCQDataset
from .image_vqa import ImageVQADataset, CustomVQADataset, FSC147Dataset
from .image_shortqa import ImageShortQADataset
from .simplevqa import SimpleVQA, SimpleVQA_300
from .mmsearch import MMSearchEnd2End
from .freak import FREAKMCQDataset, FREAKVQADataset
from .pixmo_count import PixmoCount
from .countqa import CountQA
from .lvis import LVISBBoxDataset, LVISMaskDataset
from .odinw import ODinWBBoxDataset
from .fscd import FSCD147BBoxDataset, FSCDLVISBBoxDataset
from .browsecomp_vl import BrowseCompVLDataset
from .realxbench import RealXBench
from .mmsearch_plus import MMSearchPlus
from .mmbrowsecomp import MMBrowseCompDataset
from .vdrbench import VDRBench
from .visbrowse_bench import VisBrowseBenchDataset
from .embspatialbench import EmbSpatialBench
from .erqabench import ERQABench
from .mmsibench import MMSIBench
from .mindcubebench import MindCubeBench
from .sparbench import SparBench

# Resolution order matters when two classes serve the same name; it follows the
# order of the original VLMEvalKit-style registry.
IMAGE_DATASET = [
    ImageYORNDataset, ImageMCQDataset, ImageVQADataset, ImageShortQADataset,
    SimpleVQA, SimpleVQA_300,
    MMSearchEnd2End,
    FSC147Dataset, FREAKMCQDataset, FREAKVQADataset, PixmoCount, CountQA,
    LVISBBoxDataset, LVISMaskDataset, ODinWBBoxDataset, FSCD147BBoxDataset, FSCDLVISBBoxDataset,
    BrowseCompVLDataset, RealXBench, MMSearchPlus, MMBrowseCompDataset, VDRBench, VisBrowseBenchDataset,
    EmbSpatialBench, ERQABench, MMSIBench, MindCubeBench, SparBench,
]
CUSTOM_DATASET = [CustomMCQDataset, CustomVQADataset]
DATASET_CLASSES = IMAGE_DATASET + CUSTOM_DATASET
SUPPORTED_DATASETS = []
for DATASET_CLS in DATASET_CLASSES:
    SUPPORTED_DATASETS.extend(DATASET_CLS.supported_datasets())


def DATASET_TYPE(dataset, *, default: str = 'MCQ') -> str:
    for cls in DATASET_CLASSES:
        if dataset in cls.supported_datasets():
            if hasattr(cls, 'TYPE'):
                return cls.TYPE
    if 'openended' in dataset.lower():
        return 'VQA'
    warnings.warn(f'Dataset {dataset} is a custom one and not annotated as `openended`, will treat as {default}. ')  # noqa: E501
    return default


def DATASET_MODALITY(dataset, *, default: str = 'IMAGE') -> str:
    if dataset is None:
        warnings.warn(f'Dataset is not specified, will treat modality as {default}. ')
        return default
    for cls in DATASET_CLASSES:
        if dataset in cls.supported_datasets():
            if hasattr(cls, 'MODALITY'):
                return cls.MODALITY
    warnings.warn(f'Dataset {dataset} is a custom one, will treat modality as {default}. ')
    return default


def build_dataset(dataset_name, **kwargs):
    for cls in DATASET_CLASSES:
        if dataset_name in cls.supported_datasets():
            return cls(dataset=dataset_name, **kwargs)

    warnings.warn(f'Dataset {dataset_name} is not officially supported. ')
    data_file = osp.join(LMUDataRoot(), f'{dataset_name}.tsv')
    if not osp.exists(data_file):
        warnings.warn(f'Data file {data_file} does not exist. Dataset building failed. ')
        return None

    data = load(data_file)
    if 'question' not in [x.lower() for x in data.columns]:
        warnings.warn(f'Data file {data_file} does not have a `question` column. Dataset building failed. ')
        return None

    if 'A' in data and 'B' in data:
        if 'image' in data or 'image_path' in data:
            warnings.warn(f'Will assume unsupported dataset {dataset_name} as a Custom MCQ dataset. ')
            return CustomMCQDataset(dataset=dataset_name, **kwargs)
        warnings.warn(f'Text-only MCQ dataset {dataset_name} is not supported by this release. ')
        return None
    warnings.warn(f'Will assume unsupported dataset {dataset_name} as a Custom VQA dataset. ')
    return CustomVQADataset(dataset=dataset_name, **kwargs)


def infer_dataset_basename(dataset_name):
    basename = "_".join(dataset_name.split("_")[:-1])
    return basename


__all__ = ['build_dataset', 'img_root_map', 'ImageBaseDataset'] + [cls.__name__ for cls in DATASET_CLASSES]
