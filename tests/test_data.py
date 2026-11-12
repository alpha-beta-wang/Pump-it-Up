"""Tests for the cleaning chain and the shared fold split.

Run with: uv run pytest
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, data


@pytest.fixture
def raw_like() -> pd.DataFrame:
    """A tiny frame shaped like the raw competition data."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "funder": ["  Danida ", "0", "Commu", None],
            "installer": ["DWE", "unknown", "commu", "-"],
            "longitude": [37.2, 0.0, 29.6, -3.0],
            "latitude": [-3.2, 0.0, -6.1, 33.0],
            "quantity": ["enough", "dry", "enough", "dry"],
            "quantity_group": ["enough", "dry", "enough", "dry"],
            "recorded_by": ["GeoData"] * 4,
            "public_meeting": [True, False, None, True],
            "permit": [True, None, False, True],
        }
    )


def test_basic_clean_drops_useless_columns(raw_like):
    out = data.basic_clean(raw_like)
    assert "recorded_by" not in out.columns  # constant column
    assert "quantity_group" not in out.columns  # duplicate of `quantity`


def test_placeholders_become_missing(raw_like):
    out = data.normalize_organisations(raw_like)
    assert pd.isna(out.loc[1, "funder"])  # "0"
    assert pd.isna(out.loc[1, "installer"])  # "unknown"
    assert pd.isna(out.loc[3, "installer"])  # "-"
    assert pd.isna(out.loc[3, "funder"])  # None stays None


def test_aliases_are_merged(raw_like):
    out = data.normalize_organisations(raw_like)
    assert out.loc[0, "funder"] == "danida"  # strip + lowercase
    assert out.loc[2, "funder"] == "community"  # Commu -> community
    assert out.loc[2, "installer"] == "community"  # commu -> community
    assert out.loc[0, "installer"] == "dwe"  # DWE stays distinct from RWE


def test_normalisation_is_idempotent(raw_like):
    once = data.normalize_organisations(raw_like)
    twice = data.normalize_organisations(once)
    assert once["funder"].astype(object).equals(twice["funder"].astype(object))
    assert once["installer"].astype(object).equals(twice["installer"].astype(object))


def test_sanitize_coordinates_nulls_impossible_values(raw_like):
    out = data.sanitize_coordinates(raw_like)
    assert pd.isna(out.loc[1, "longitude"])  # longitude == 0
    assert pd.isna(out.loc[1, "latitude"])
    assert pd.isna(out.loc[3, "longitude"])  # negative longitude
    assert pd.isna(out.loc[3, "latitude"])  # latitude in the longitude range
    assert out.loc[0, "longitude"] == 37.2  # valid row untouched
    assert out.loc[0, "latitude"] == -3.2


def test_clean_runs_the_whole_chain(raw_like):
    out = data.clean(raw_like)
    assert out.shape[0] == len(raw_like)
    assert "recorded_by" not in out.columns


def test_folds_are_stratified_and_deterministic():
    y = np.array([0] * 100 + [1] * 30 + [2] * 20)
    folds = data.make_folds(y, n_splits=5, seed=7)
    assert set(np.unique(folds)) == set(range(5))
    # every class must be present in every fold
    for fold in range(5):
        assert set(np.unique(y[folds == fold])) == {0, 1, 2}
    assert np.array_equal(folds, data.make_folds(y, n_splits=5, seed=7))
    assert not np.array_equal(folds, data.make_folds(y, n_splits=5, seed=8))


def test_base_parquet_roundtrip(tmp_path, raw_like):
    path = tmp_path / "base.parquet"
    train = data.clean(raw_like).assign(status_group=["functional"] * 4)
    test = data.clean(raw_like)
    data.save_base(train, test, path=path)

    loaded_train, loaded_test = data.load_base(path=path)
    assert list(loaded_train.columns) == list(loaded_test.columns) + ["status_group"] or set(
        loaded_test.columns
    ) <= set(loaded_train.columns)
    assert len(loaded_train) == len(train)
    assert len(loaded_test) == len(test)
    assert "split" not in loaded_train.columns


def test_real_data_schema_matches():
    """train and test must expose the same feature columns after cleaning."""
    train, test = data.prepare_base()
    assert len(train) == 59400
    assert len(test) == 14850
    assert config.TARGET in train.columns
    assert config.TARGET not in test.columns
    feature_cols = [c for c in train.columns if c != config.TARGET]
    assert feature_cols == list(test.columns)
    assert train["longitude"].isna().sum() == 1819
