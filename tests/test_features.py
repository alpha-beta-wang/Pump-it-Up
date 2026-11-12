"""Tests for feature construction."""

from __future__ import annotations

import pandas as pd

from src import config, data, features


def test_build_features_shapes_and_target():
    train, test = data.prepare_base()
    X_train, X_test, y, class_names = features.build_features(train, test)

    assert len(X_train) == len(train)
    assert len(X_test) == len(test)
    assert config.TARGET not in X_train.columns
    assert config.ID not in X_train.columns
    assert len(y) == len(train)
    assert class_names == config.CLASSES
    assert set(X_train.columns) == set(X_test.columns)
    # derived columns exist
    for col in ("age", "rec_year", "construction_year_is0", "population_is0"):
        assert col in X_train.columns


def test_v2_adds_the_documented_feature_groups():
    train, test = data.prepare_base()
    X_v1, _, _, _ = features.build_features(train, test, version="v1")
    X_v2, X_v2_test, _, _ = features.build_features(train, test, version="v2")

    added = set(X_v2.columns) - set(X_v1.columns)
    assert len(added) > 20
    assert "missing_pattern" in added
    assert "geo_cell_1" in added
    assert "region_x_extraction" in added
    assert "wpt_name_len" in added
    assert "subvillage_size" in added
    # geo_cell_001 was dropped on purpose: too many distinct values
    assert "geo_cell_001" not in added
    assert set(X_v2.columns) == set(X_v2_test.columns)


def test_train_and_test_share_one_frame():
    """Group features (village size, frequencies) must use the same scale."""
    train, test = data.prepare_base()
    X_train, X_test, _, _ = features.build_features(train, test, version="v2")
    combined = pd.concat([X_train["subvillage_size"], X_test["subvillage_size"]])
    assert combined.max() >= 2
    # a village's size should be identical for its train and test rows
    key = "subvillage"
    merged = pd.concat(
        [
            X_train[[key, "subvillage_size"]].assign(split="train"),
            X_test[[key, "subvillage_size"]].assign(split="test"),
        ],
        ignore_index=True,
    )
    spread = merged.groupby(key, observed=True)["subvillage_size"].nunique()
    assert spread.max() == 1


def test_categorical_codes_are_aligned():
    """A category present only in test must still encode consistently."""
    train = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "date_recorded": ["2013-01-01"] * 4,
            "cat": ["a", "b", "a", "b"],
            "amount_tsh": [0, 1, 2, 3],
            "construction_year": [2000, 0, 2001, 0],
            "gps_height": [100, 0, 200, 0],
            "longitude": [30.0, 31.0, 32.0, 33.0],
            "latitude": [-3.0, -4.0, -5.0, -6.0],
            "population": [0, 10, 0, 20],
            "status_group": ["functional"] * 4,
        }
    )
    test = train.drop(columns=["status_group"]).copy()
    test["id"] = [5, 6, 7, 8]
    test.loc[0, "cat"] = "never_seen_in_train"

    X_train, X_test, _, _ = features.build_features(train, test)

    assert list(X_train["cat"].cat.categories) == list(X_test["cat"].cat.categories)
    assert "never_seen_in_train" in X_train["cat"].cat.categories
