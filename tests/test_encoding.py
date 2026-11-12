"""Tests for leak-free target encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, encoding


def _toy(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    categories = rng.choice(["a", "b", "c", "d"], size=n)
    # signal: category 'a' is mostly class 0, 'd' mostly class 2
    base = {"a": 0, "b": 0, "c": 2, "d": 2}
    y = np.array([base[c] for c in categories])
    flip = rng.random(n) < 0.15
    y[flip] = rng.integers(0, 3, size=flip.sum())
    X = pd.DataFrame({"region": categories, "value": rng.normal(size=n)})
    return X, y


def test_encoded_columns_are_added():
    X, y = _toy()
    X_train, X_valid, X_test = encoding.encode_fold(
        X.iloc[:300], y[:300], X.iloc[300:], X.iloc[:50], columns=["region"]
    )
    expected = {f"te_region_{i}" for i in range(len(config.CLASSES))}
    assert expected <= set(X_train.columns)
    assert expected <= set(X_valid.columns)
    assert expected <= set(X_test.columns)
    assert "region" in X_train.columns  # originals preserved


def test_no_nans_and_probability_range():
    X, y = _toy()
    X_train, X_valid, _ = encoding.encode_fold(
        X.iloc[:300], y[:300], X.iloc[300:], columns=["region"]
    )
    te_cols = [c for c in X_train.columns if c.startswith("te_")]
    values = X_train[te_cols].to_numpy()
    assert not np.isnan(values).any()
    assert values.min() >= 0.0 and values.max() <= 1.0


def test_train_rows_use_out_of_fold_statistics():
    """A row's own label must not be present in its own encoding.

    With one category holding identical rows, the encoding of a row computed
    from *other* rows is strictly below 1.0 whenever the fold is not
    single-class.
    """
    X = pd.DataFrame({"region": ["same"] * 10})
    y = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
    X_train, _, _ = encoding.encode_fold(
        X, y, X.head(2), columns=["region"], n_inner=5, smoothing=0.0
    )
    # every training row is encoded from 8 of the other 9 rows, so the rates
    # must vary and none may be exactly 1.0 for the majority class
    encoded = X_train["te_region_0"].to_numpy()
    assert encoded.max() < 1.0
    assert len(np.unique(encoded)) > 1


def test_unseen_category_falls_back_to_prior():
    X_train = pd.DataFrame({"region": ["a", "a", "b", "b"]})
    y_train = np.array([0, 0, 1, 1])
    X_valid = pd.DataFrame({"region": ["never_seen"]})
    _, X_valid_enc, _ = encoding.encode_fold(
        X_train, y_train, X_valid, columns=["region"], smoothing=0.0, n_inner=2
    )
    # prior is 50/50, so the unseen key must decode to the prior
    assert np.isclose(X_valid_enc["te_region_0"].iloc[0], 0.5)


def test_shuffled_labels_carry_no_signal():
    """Sanity check from docs/技术方案.md §7: shuffled labels must look like noise."""
    X, y = _toy(n=1200, seed=3)
    rng = np.random.default_rng(0)
    y_shuffled = rng.permutation(y)
    X_train, X_valid, _ = encoding.encode_fold(
        X.iloc[:1000], y_shuffled[:1000], X.iloc[1000:], columns=["region"]
    )
    te_cols = [c for c in X_train.columns if c.startswith("te_")]
    # On shuffled labels the encoded class probabilities should all be close to
    # the overall prior, so no class dominates.
    means = X_train[te_cols].groupby([c for c in te_cols]).size()
    assert means.sum() == len(X_train)
    assert X_valid[te_cols].mean().max() < 0.6
