"""Data loading, light cleaning and fold generation.

Design rules (see docs/技术方案.md §5.2):
* training and test data go through exactly the same transformation chain;
* nothing here looks at the target, except ``make_folds`` which only needs y
  to stratify;
* heavy feature engineering lives in ``features.py``, not here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config

# Columns that carry no information or duplicate another column.
# Verified against the raw data: `recorded_by` is constant across train/test,
# `quantity_group` is row-wise identical to `quantity`.
DROPPED_COLUMNS = ["recorded_by", "quantity_group"]

# Columns where a 0 means "not recorded" rather than a real measurement.
PSEUDO_MISSING = [
    "amount_tsh",
    "construction_year",
    "gps_height",
    "longitude",
    "latitude",
    "population",
]


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, test, submission_format) as plain DataFrames."""
    train = pd.read_csv(config.TRAIN_VALUES)
    labels = pd.read_csv(config.TRAIN_LABELS)
    test = pd.read_csv(config.TEST_VALUES)
    submission = pd.read_csv(config.SUBMISSION_FORMAT)

    assert len(train) == len(labels), "training values and labels are misaligned"
    assert set(train[config.ID]) == set(labels[config.ID]), "training ids differ between files"
    assert list(train.columns) == list(test.columns), "train/test schemas differ"

    train = train.merge(labels, on=config.ID, how="left", validate="one_to_one")
    assert train[config.TARGET].notna().all(), "labels missing after merge"
    return train, test, submission


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop useless columns and normalize dtypes. No value imputation here."""
    out = df.drop(columns=[c for c in DROPPED_COLUMNS if c in df.columns]).copy()

    # public_meeting / permit are True/False with NaNs -> keep the third state.
    for col in ("public_meeting", "permit"):
        if col in out.columns:
            out[col] = out[col].astype("boolean").astype("category")

    return out


def sanitize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Null out the (0, 0)-ish coordinates and anything outside Tanzania.

    Not part of the E1 baseline (kept byte-compatible with the recorded run),
    but available for the M2 cleaning pipeline.
    """
    out = df.copy()
    bad = (out["longitude"] <= 0) | (out["latitude"] >= -1.0)
    out.loc[bad, ["longitude", "latitude"]] = np.nan
    return out


def make_folds(
    y: np.ndarray, n_splits: int = config.N_SPLITS, seed: int = config.SEED
) -> np.ndarray:
    """Stratified fold ids, one integer per row."""
    folds = np.full(len(y), -1, dtype=np.int8)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[val_idx] = fold
    assert (folds >= 0).all(), "some rows were never assigned to a fold"
    return folds


def get_folds(y: np.ndarray, cache: bool = True) -> np.ndarray:
    """Load the shared fold assignment, creating it on first use.

    All experiments must reuse the same folds, otherwise CV scores are not
    comparable (see docs/技术方案.md F2).
    """
    if cache and config.FOLDS_FILE.exists():
        folds = np.load(config.FOLDS_FILE)
        if len(folds) == len(y):
            return folds
    folds = make_folds(y)
    if cache:
        config.ensure_dirs()
        np.save(config.FOLDS_FILE, folds)
    return folds
