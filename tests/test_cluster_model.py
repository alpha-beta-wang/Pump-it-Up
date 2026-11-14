"""Tests for partition-wise ensemble members."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import cluster_model, config


def _toy_frames(
    n_train: int = 40, n_test: int = 12
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Minimal feature frames with the columns the partitions rely on."""
    rng = np.random.default_rng(0)
    regions = ["Arusha", "Kigoma", "Mbeya", "Iringa"]
    types = ["hand pump", "communal standpipe"]
    rows = n_train + n_test
    frame = pd.DataFrame(
        {
            "region": [regions[i % len(regions)] for i in range(rows)],
            "waterpoint_type_group": [types[i % len(types)] for i in range(rows)],
            "lga": [f"lga_{i % 5}" for i in range(rows)],
            "gps_height": rng.normal(1000, 100, rows).round(1),
            "population": rng.integers(0, 500, rows),
            "age": rng.integers(0, 40, rows),
            "amount_tsh": rng.integers(0, 100, rows),
            "longitude": rng.uniform(30, 40, rows).round(3),
            "latitude": rng.uniform(-11, -1, rows).round(3),
        }
    )
    frame["region"] = frame["region"].astype("category")
    frame["waterpoint_type_group"] = frame["waterpoint_type_group"].astype("category")
    frame["lga"] = frame["lga"].astype("category")
    return frame.iloc[:n_train].reset_index(drop=True), frame.iloc[
        n_train:
    ].reset_index(drop=True)


def test_partition_from_column_aligns_both_sides():
    train, test = _toy_frames()
    labels_train, labels_test = cluster_model.make_partition(train, test, by="region")

    assert len(labels_train) == len(train)
    assert len(labels_test) == len(test)
    assert labels_train.dtype.kind in "US"
    # every test label must exist in the training label set
    assert set(labels_test) <= set(labels_train)


def test_kmeans_partition_shares_cluster_ids():
    train, test = _toy_frames()
    labels_train, labels_test = cluster_model.make_partition(
        train, test, by="kmeans", n_clusters=3
    )
    assert len(labels_train) == len(train)
    assert len(labels_test) == len(test)
    assert len(set(labels_train)) <= 3
    assert set(labels_test) <= set(labels_train)


def test_run_member_returns_valid_probabilities_and_coverage():
    train, test = _toy_frames()
    y = np.array(
        [0, 1, 2] * (len(train) // 3) + [0] * (len(train) - 3 * (len(train) // 3))
    )
    folds = np.random.default_rng(1).permutation(np.arange(len(train)) % 2)
    labels_train, labels_test = cluster_model.make_partition(train, test, by="region")

    oof, test_proba, covered = cluster_model.run_member(
        train,
        test,
        y,
        folds,
        labels_train,
        labels_test,
        model_name="rf",
        min_rows=4,
        verbose=False,
    )

    assert oof.shape == (len(train), len(config.CLASSES))
    assert test_proba.shape == (len(test), len(config.CLASSES))
    assert np.allclose(oof.sum(axis=1), 1.0)
    assert np.allclose(test_proba.sum(axis=1), 1.0)
    assert covered.shape == (len(train),)
    # with 4 regions and 40 rows every OOF row should be covered by a partition model
    assert covered.mean() > 0.5


def test_min_rows_falls_back_to_the_global_model():
    train, test = _toy_frames()
    y = np.array(
        [0, 1, 2] * (len(train) // 3) + [0] * (len(train) - 3 * (len(train) // 3))
    )
    folds = np.random.default_rng(1).permutation(np.arange(len(train)) % 2)
    labels_train, labels_test = cluster_model.make_partition(train, test, by="region")

    _, _, covered = cluster_model.run_member(
        train,
        test,
        y,
        folds,
        labels_train,
        labels_test,
        model_name="rf",
        min_rows=10_000,  # nothing is big enough
        verbose=False,
    )
    assert not covered.any()
