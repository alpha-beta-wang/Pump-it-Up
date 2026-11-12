"""Target encoding that cannot leak.

The naive version (encode the whole training fold, then fit on that same fold)
scored 0.7606 in E2 - a 4 point drop - because each row's encoding contains its
own label. Here every training row is encoded from *other* inner folds, while
validation/test rows use the full outer training fold.

Reference: docs/技术方案.md §4.1 (A2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config

# Columns worth encoding: mid-to-high cardinality with real target signal.
# `wpt_name` is deliberately excluded - 37k values, mostly singletons.
TE_COLUMNS = [
    "basin",
    "region",
    "lga",
    "ward",
    "subvillage",
    "extraction_type",
    "management",
    "payment_type",
    "source",
    "waterpoint_type",
    "funder",
    "installer",
    "scheme_name",
]


def _prior(labels: np.ndarray, n_classes: int) -> np.ndarray:
    return np.bincount(labels, minlength=n_classes) / len(labels)


def _class_rates(
    keys: pd.Series, labels: np.ndarray, n_classes: int, smoothing: float
) -> list[pd.Series]:
    """P(class | key), smoothed towards the global prior of ``labels``."""
    prior = _prior(labels, n_classes)
    frame = pd.DataFrame({"key": keys.astype(object).values, "y": labels})
    rates: list[pd.Series] = []
    for cls in range(n_classes):
        stats = (
            frame.assign(hit=(frame.y == cls).astype(float))
            .groupby("key")["hit"]
            .agg(["sum", "count"])
        )
        rate = (stats["sum"] + smoothing * prior[cls]) / (stats["count"] + smoothing)
        rates.append(rate)
    return rates


def _apply_rates(keys: pd.Series, rates: list[pd.Series], prior: np.ndarray) -> np.ndarray:
    """Look up encodings, falling back to the prior for unseen keys."""
    values = keys.astype(object)
    out = np.empty((len(keys), len(rates)), dtype=float)
    for cls, rate in enumerate(rates):
        encoded = rate.reindex(values).to_numpy(dtype=float)
        out[:, cls] = np.where(np.isnan(encoded), prior[cls], encoded)
    return out


def encode_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    columns: list[str] | None = None,
    n_inner: int = 5,
    smoothing: float = 20.0,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Return (encoded train, encoded valid, encoded test).

    Train rows get out-of-fold encodings from the inner split; valid/test rows
    use encodings fitted on the whole outer training fold. Original columns are
    kept, encoded columns are appended as ``te_<column>_<class>``.
    """
    columns = [c for c in (columns or TE_COLUMNS) if c in X_train.columns]
    n_classes = len(config.CLASSES)

    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test = X_test.copy() if X_test is not None else None

    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    prior_full = _prior(y_train, n_classes)

    for col in columns:
        nested = np.zeros((len(X_train), n_classes), dtype=float)
        for fit_idx, apply_idx in inner.split(np.zeros(len(X_train)), y_train):
            y_fit = y_train[fit_idx]
            rates = _class_rates(X_train[col].iloc[fit_idx], y_fit, n_classes, smoothing)
            nested[apply_idx] = _apply_rates(
                X_train[col].iloc[apply_idx], rates, _prior(y_fit, n_classes)
            )

        full_rates = _class_rates(X_train[col], y_train, n_classes, smoothing)
        valid_values = _apply_rates(X_valid[col], full_rates, prior_full)
        test_values = (
            _apply_rates(X_test[col], full_rates, prior_full) if X_test is not None else None
        )

        for cls in range(n_classes):
            name = f"te_{col}_{cls}"
            X_train[name] = nested[:, cls]
            X_valid[name] = valid_values[:, cls]
            if X_test is not None:
                X_test[name] = test_values[:, cls]

    return X_train, X_valid, X_test
