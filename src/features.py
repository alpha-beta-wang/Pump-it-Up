"""Feature construction.

Only the E1 feature set lives here for now: it deliberately contains almost no
hand-made features, because the measured baseline showed extra geo / frequency
features add nothing over LightGBM's native categorical handling
(docs/技术方案.md §2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import PSEUDO_MISSING


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    recorded = pd.to_datetime(df["date_recorded"])
    df["rec_year"] = recorded.dt.year
    df["rec_month"] = recorded.dt.month
    df["rec_day_of_year"] = recorded.dt.day_of_year
    df = df.drop(columns=["date_recorded"])
    # Age at survey time; construction_year == 0 means unknown, so it becomes NaN.
    built = df["construction_year"].replace(0, np.nan)
    df["age"] = df["rec_year"] - built
    return df


def _add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    for col in PSEUDO_MISSING:
        df[f"{col}_is0"] = (df[col] == 0).astype("int8")
    return df


def _cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object or isinstance(df[col].dtype, pd.StringDtype):
            df[col] = df[col].astype("category")
    return df


def _align_categories(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give both frames one shared category set per column.

    LightGBM/pandas encode categoricals by position, so if test sees a category
    that train never had, the codes silently mean different things. Taking the
    union makes the encoding consistent. This uses feature values only, never
    the labels, so it does not leak anything.
    """
    for col in X_train.columns:
        if isinstance(X_test[col].dtype, pd.CategoricalDtype):
            categories = X_train[col].cat.categories.append(X_test[col].cat.categories).unique()
            X_train[col] = X_train[col].cat.set_categories(categories)
            X_test[col] = X_test[col].cat.set_categories(categories)
    return X_train, X_test


def build_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    """Turn cleaned frames into model matrices.

    Returns (X_train, X_test, y_encoded, class_names).
    The target column is removed from both frames; everything else is shared.
    """
    y_raw = train[config.TARGET]
    train = train.drop(columns=[config.TARGET])
    frames = [train.drop(columns=[config.ID]), test.drop(columns=[config.ID])]

    processed = []
    for frame in frames:
        frame = _add_time_features(frame)
        frame = _add_missing_flags(frame)
        frame = _cast_categoricals(frame)
        processed.append(frame)

    # Align columns in case one side gained a category the other lacks.
    common = [c for c in processed[0].columns if c in processed[1].columns]
    X_train = processed[0][common].copy()
    X_test = processed[1][common].copy()
    X_train, X_test = _align_categories(X_train, X_test)

    y = y_raw.map(config.CLASS_TO_CODE).to_numpy()
    assert not np.isnan(y).any(), "unmapped target values"
    return X_train, X_test, y.astype(int), config.CLASSES
