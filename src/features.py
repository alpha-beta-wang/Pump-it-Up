"""Feature construction.

Two feature sets are available:

``v1`` (default)
    Minimal: date parts, missing-value flags, age. This is what the E1/E4
    comparison used, and it is the honest baseline.
``v2``
    Adds the candidates from docs/技术方案.md §3 B2/B3/B6/B7 plus frequency and
    spatial-cell features. They were neutral for LightGBM (E3), but CatBoost can
    turn the extra categorical columns into ordered target statistics, so they
    get a second chance with the main model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data import PSEUDO_MISSING

TEXT_COLUMNS = ["wpt_name", "scheme_name"]
FREQ_COLUMNS = ["funder", "installer", "subvillage", "ward", "scheme_name", "wpt_name", "lga"]
NULLABLE_COLUMNS = [
    "funder",
    "installer",
    "public_meeting",
    "scheme_management",
    "scheme_name",
    "permit",
]

# Columns that the importance ranking (docs/feature_importance.csv) showed to be
# dead weight: two indicators are constant after coordinate sanitation, and the
# rest contribute <0.5% each while costing CatBoost a lot of capacity
# (wpt_name alone has 37,399 categories for 0.17% importance).
PRUNED_COLUMNS = [
    "wpt_name",
    "num_private",
    "region_code",
    "district_code",
    "rec_month",
    "longitude_is0",
    "latitude_is0",
]

KEYWORD_PATTERNS = {
    "kw_well": r"\bwell\b",
    "kw_borehole": r"borehole|bore hole",
    "kw_shallow": r"shallow",
    "kw_spring": r"spring",
    "kw_pump": r"pump",
    "kw_tank": r"tank",
    "kw_school": r"school",
    "kw_church": r"church|mission",
    "kw_health": r"health|dispensary",
}


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


def _add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap semantics from the two free-text name columns (B2)."""
    for col in TEXT_COLUMNS:
        if col not in df.columns:
            continue
        text = df[col].astype("string").str.lower().fillna("")
        df[f"{col}_len"] = text.str.len().astype("int16")
        df[f"{col}_words"] = text.str.count(r"\s+").astype("int16") + 1
        for name, pattern in KEYWORD_PATTERNS.items():
            df[f"{col}_{name}"] = text.str.contains(pattern, regex=True).astype("int8")
    return df


def _add_missing_fingerprint(df: pd.DataFrame) -> pd.DataFrame:
    """Which of the nullable fields are absent (B3)."""
    bits = []
    for col in NULLABLE_COLUMNS:
        if col in df.columns:
            bits.append(df[col].isna().astype("int8").astype(str))
    df["missing_pattern"] = bits[0].str.cat(bits[1:], sep="")
    return df


def _add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Coarse location cells and population density per village (B6).

    Only the 0.1-degree cell is used. A 0.01-degree cell was tried and dropped:
    it has ~29k distinct values (a near-unique id), which made CatBoost's target
    statistics both slow and prone to memorising single water points.
    """
    lat = df["latitude"]
    lon = df["longitude"]
    df["geo_cell_1"] = (lat.round(1).astype("string") + "_" + lon.round(1).astype("string")).mask(
        lat.isna() | lon.isna()
    )

    village_size = (
        df.groupby("subvillage", observed=True)["id"].transform("size") if "id" in df else 1
    )
    df["subvillage_size"] = village_size
    df["population_per_well"] = df["population"] / df["subvillage_size"].clip(lower=1)
    return df


def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df["region_x_extraction"] = (
        df["region"].astype("string") + "|" + df["extraction_type"].astype("string")
    )
    df["mgmt_x_payment"] = (
        df["management"].astype("string") + "|" + df["payment_type"].astype("string")
    )
    df["quality_x_source"] = (
        df["water_quality"].astype("string") + "|" + df["source"].astype("string")
    )
    return df


def _add_frequency(df: pd.DataFrame) -> pd.DataFrame:
    for col in FREQ_COLUMNS:
        if col in df.columns:
            counts = df[col].value_counts()
            df[f"{col}_freq"] = df[col].map(counts).fillna(0).astype("int32")
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
    train: pd.DataFrame, test: pd.DataFrame, version: str = "v1"
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    """Turn cleaned frames into model matrices.

    Train and test are concatenated before transformations so that group-based
    features (village size, frequency counts) share one scale, and so both
    sides get identical category sets by construction.

    Returns (X_train, X_test, y_encoded, class_names).
    """
    if version not in ("v1", "v2", "v3"):
        raise ValueError(f"unknown feature version: {version}")

    y_raw = train[config.TARGET]
    n_train = len(train)
    combined = pd.concat(
        [train.drop(columns=[config.TARGET]), test],
        ignore_index=True,
    )

    combined = _add_time_features(combined)
    combined = _add_missing_flags(combined)
    if version == "v2":
        combined = _add_text_features(combined)
        combined = _add_missing_fingerprint(combined)
        combined = _add_spatial_features(combined)
        combined = _add_interactions(combined)
        combined = _add_frequency(combined)
    combined = _cast_categoricals(combined)
    combined = combined.drop(columns=[config.ID])
    if version == "v3":
        combined = combined.drop(columns=[c for c in PRUNED_COLUMNS if c in combined.columns])

    X_train = combined.iloc[:n_train].reset_index(drop=True)
    X_test = combined.iloc[n_train:].reset_index(drop=True)
    X_train, X_test = _align_categories(X_train, X_test)

    y = y_raw.map(config.CLASS_TO_CODE).to_numpy()
    assert not np.isnan(y).any(), "unmapped target values"
    return X_train, X_test, y.astype(int), config.CLASSES


def save_feature_table(version: str = "v1", path=None) -> Path:
    """Persist the feature matrix plus target and fold id (stage 3 deliverable).

    Writes one parquet holding train rows (with ``status_group_code`` and
    ``fold``) followed by test rows (those two columns are NA there), so the
    matrix can be inspected without re-running the pipeline.
    """
    from . import data  # local import keeps the module import graph simple

    train, test = data.load_base()
    X_train, X_test, y, _ = build_features(train, test, version=version)
    folds = data.get_folds(y)

    train_out = X_train.assign(
        status_group_code=y,
        fold=folds,
        split="train",
    )
    test_out = X_test.assign(
        status_group_code=pd.NA,
        fold=pd.NA,
        split="test",
    )
    combined = pd.concat([train_out, test_out], ignore_index=True)

    path = path or (config.DATA_PROCESSED / f"features_{version}.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Persist a feature table.")
    parser.add_argument("--version", default="v1", choices=["v1", "v2", "v3"])
    args = parser.parse_args()
    out = save_feature_table(args.version)
    print(f"wrote {out}")
