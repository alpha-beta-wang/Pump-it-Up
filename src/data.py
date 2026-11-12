"""Data loading, cleaning and fold generation.

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

# Free-text columns recording who paid for / installed the water point.
ORGANISATION_COLUMNS = ["funder", "installer"]

# Tokens that mean "nothing recorded" rather than an organisation name.
PLACEHOLDER_VALUES = {"", "0", "-", "n a", "nan", "no", "none", "not known", "unknown"}

# Spelling variants of the same organisation, verified by inspecting the raw
# value counts. Deliberately conservative: DWE (district water engineer) and
# RWE (regional water engineer) are different bodies and must not be merged.
ORGANISATION_ALIASES = {
    "commu": "community",
    "danid": "danida",
    "govt": "government",
}


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
    """Null out coordinates that cannot be in Tanzania.

    Two independent problems were measured in the raw data:
    ``longitude == 0`` (1,812 rows) and ``latitude > -1`` (1,819 rows) - both
    sit in the Gulf of Guinea, i.e. they are missing values, not locations.
    Swapped lat/lon does *not* occur (0 rows), so no repair is attempted.
    """
    out = df.copy()
    bad = (out["longitude"] <= 0) | (out["latitude"] >= -1.0)
    out.loc[bad, ["longitude", "latitude"]] = np.nan
    return out


def normalize_text(series: pd.Series) -> pd.Series:
    """Lowercase, strip punctuation, collapse whitespace.

    ``NaN`` is preserved; empty results become ``NaN`` later via
    ``PLACEHOLDER_VALUES``.
    """
    out = series.astype("string").str.lower().str.strip()
    out = out.str.replace(r"[^\w\s]", " ", regex=True)
    return out.str.replace(r"\s+", " ", regex=True).str.strip()


def normalize_organisations(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise ``funder`` / ``installer`` and turn placeholders into NA.

    Effect measured on the training set: ``installer`` 2,145 -> 1,863 distinct
    values, ``funder`` 1,896 -> 1,877; missing (placeholder or blank) rises to
    4,579 / 4,489 rows. Normalisation is idempotent.
    """
    out = df.copy()
    for col in ORGANISATION_COLUMNS:
        if col not in out.columns:
            continue
        canon = normalize_text(out[col])
        canon = canon.mask(canon.isin(PLACEHOLDER_VALUES))
        canon = canon.replace(ORGANISATION_ALIASES)
        out[col] = canon.astype("category")
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning chain applied identically to train and test."""
    out = basic_clean(df)
    out = normalize_organisations(out)
    out = sanitize_coordinates(out)
    return out


def load_submission_format() -> pd.DataFrame:
    """Official submission template, used to keep row order identical."""
    return pd.read_csv(config.SUBMISSION_FORMAT)


def prepare_base() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw files and return cleaned (train, test) frames."""
    train, test, _ = load_raw()
    return clean(train), clean(test)


def save_base(train: pd.DataFrame, test: pd.DataFrame, path=None) -> None:
    """Persist both splits to one parquet file for downstream stages."""
    path = path or config.BASE_PARQUET
    path.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(
        [train.assign(split="train"), test.assign(split="test")], ignore_index=True
    )
    combined.to_parquet(path, index=False)


def load_base(path=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the persisted base table and split it back into train/test.

    The parquet holds one combined frame, so the test rows carry an all-NaN
    target column; it is dropped here to keep the target out of the features.
    """
    path = path or config.BASE_PARQUET
    combined = pd.read_parquet(path)
    train = combined[combined["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    test = combined[combined["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
    if config.TARGET in test.columns:
        assert test[config.TARGET].isna().all(), "test rows must not carry labels"
        test = test.drop(columns=[config.TARGET])
    return train, test


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


if __name__ == "__main__":
    import time

    started = time.time()
    train_frame, test_frame = prepare_base()
    save_base(train_frame, test_frame)
    print(f"train {train_frame.shape} | test {test_frame.shape}")
    print(f"wrote {config.BASE_PARQUET} in {time.time() - started:.1f}s")
