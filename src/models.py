"""Model factories.

Kept in one place so that ``train.py`` does not grow model-specific branches and
so that later ensemble stages can build the same estimators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from . import config


def build(name: str, seed: int = config.SEED, **overrides):
    """Return an unfitted estimator for the given model name."""
    if name == "lgbm":
        params = dict(
            objective="multiclass",
            num_class=len(config.CLASSES),
            n_estimators=400,
            learning_rate=0.06,
            num_leaves=48,
            min_child_samples=60,
            subsample=0.9,
            colsample_bytree=0.8,
            verbose=-1,
            n_jobs=8,
            random_state=seed,
        )
        params.update(overrides)
        return LGBMClassifier(**params)

    if name == "catboost":
        params = dict(
            loss_function="MultiClass",
            iterations=700,
            learning_rate=0.08,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=seed,
            thread_count=8,
            verbose=False,
            allow_writing_files=False,
        )
        params.update(overrides)
        return CatBoostClassifier(**params)

    if name == "rf":
        # Community solutions report ~0.82 with a plain random forest, so it is
        # worth having as a diverse member even though boosting is stronger.
        params = dict(
            n_estimators=300,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=8,
            random_state=seed,
        )
        params.update(overrides)
        return RandomForestClassifier(**params)

    if name == "xgb":
        params = dict(
            objective="multi:softprob",
            num_class=len(config.CLASSES),
            n_estimators=500,
            learning_rate=0.06,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.8,
            tree_method="hist",
            enable_categorical=True,
            n_jobs=8,
            random_state=seed,
        )
        params.update(overrides)
        return XGBClassifier(**params)

    raise ValueError(f"unknown model: {name}")


def categorical_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


def to_catboost_frame(X: pd.DataFrame) -> pd.DataFrame:
    """CatBoost wants plain strings, and cannot take pd.NA inside categories."""
    out = X.astype({col: "string" for col in categorical_columns(X)})
    for col in out.columns:
        if out[col].dtype == "string":
            out[col] = out[col].fillna("missing")
    return out


def to_ordinal_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Random forests need numbers: encode categories by their code."""
    out = X.copy()
    for col in categorical_columns(X):
        codes = X[col].cat.codes.astype("float32")
        out[col] = codes.replace(-1, np.nan)
    return out


def to_xgb_frame(X: pd.DataFrame) -> pd.DataFrame:
    """XGBoost's categorical support rejects boolean categories; stringify them."""
    out = X.copy()
    for col in categorical_columns(X):
        categories = X[col].cat.categories
        if categories.dtype == bool or str(categories.dtype) == "boolean":
            out[col] = X[col].astype("string").fillna("missing").astype("category")
    return out
