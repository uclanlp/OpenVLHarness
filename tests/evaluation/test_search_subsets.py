"""FVQA_test, Infoseek_GDM_test and EncyclopedicVQA_test evaluate in full unless a subset is asked for."""
import pandas as pd
import pytest

from evaluation_kit.dataset.image_shortqa import _apply_subset

FULL = pd.DataFrame({"index": range(1000), "question": ["q"] * 1000})


@pytest.mark.parametrize("dataset", ["FVQA_test", "Infoseek_GDM_test", "EncyclopedicVQA_test"])
def test_full_split_by_default(monkeypatch, dataset):
    monkeypatch.delenv(f"{dataset.upper()}_SUBSET", raising=False)
    assert len(_apply_subset(FULL, dataset)) == 1000


def test_subset_is_fixed_and_sized(monkeypatch):
    monkeypatch.setenv("FVQA_TEST_SUBSET", "300")
    first = _apply_subset(FULL, "FVQA_test")
    assert len(first) == 300
    assert first["index"].tolist() == _apply_subset(FULL, "FVQA_test")["index"].tolist()
    monkeypatch.setenv("FVQA_TEST_SUBSET", "5000")
    assert len(_apply_subset(FULL, "FVQA_test")) == 1000


@pytest.mark.parametrize("value", ["0", "-3", "all", "true"])
def test_invalid_subset_is_rejected(monkeypatch, value):
    monkeypatch.setenv("FVQA_TEST_SUBSET", value)
    with pytest.raises(ValueError, match="FVQA_TEST_SUBSET"):
        _apply_subset(FULL, "FVQA_test")


def test_other_datasets_are_untouched(monkeypatch):
    monkeypatch.setenv("HLE_SUBSET", "10")
    assert len(_apply_subset(FULL, "hle")) == 1000
