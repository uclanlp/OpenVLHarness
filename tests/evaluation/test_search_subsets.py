"""The 300-question search subsets are hosted splits, registered like the full ones."""
import pytest

from evaluation_kit.dataset import image_shortqa
from evaluation_kit.dataset.hf_release import HF_DATASET_MD5, HF_SPLITS

SUBSETS = {"FVQA_test_300": "FVQA", "Infoseek_GDM_test_300": "Infoseek_GDM",
           "EncyclopedicVQA_test_300": "EncyclopedicVQA"}


@pytest.mark.parametrize("name,folder", sorted(SUBSETS.items()))
def test_subset_is_hosted_next_to_its_full_split(name, folder):
    full = name[: -len("_300")]
    assert HF_SPLITS[name] == HF_SPLITS[full] == folder
    assert name in HF_DATASET_MD5 and HF_DATASET_MD5[name] != HF_DATASET_MD5[full]
    assert name in image_shortqa.ImageShortQADataset.supported_datasets()


def test_no_runtime_sampling_remains():
    assert not hasattr(image_shortqa, "_apply_subset")
    assert "WorldVQA_300" in HF_SPLITS and "WorldVQA_VDR300" not in HF_SPLITS
    assert HF_SPLITS["SimpleVQA_300"] == "SimpleVQA" and "SimpleVQA_EN" not in HF_SPLITS


@pytest.mark.parametrize("name", ["FVQA_train", "Infoseek_GDM_val", "EncyclopedicVQA_val", "Gen_Seed_V0",
                                  "LiveMMBench_Reasoning", "MMSearch_Requery", "VDRBench_Full", "SimpleVQA_EN"])
def test_unreleased_splits_are_not_registered(name):
    from evaluation_kit import dataset as datasets
    assert not any(name in c.supported_datasets() for c in datasets.IMAGE_DATASET)
