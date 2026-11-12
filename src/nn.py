"""Entity-embedding style neural network, built with scikit-learn.

Community solutions for this competition are tree ensembles; a neural net is
useful mainly because its errors are structurally different, which helps a
blend (docs/技术方案.md C5). This implements the cheapest version that is still
a real embedding model: one-hot for the low/medium-cardinality categoricals,
rank-gauss style scaling for numerics, then a two-layer MLP.

High-cardinality columns (`wpt_name`, `subvillage`, `scheme_name`) are reduced
to frequency counts instead of one-hot, otherwise the input is ~40k wide.

Usage:
    uv run python -m src.nn --name N1
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config, data, evaluate, features

# One-hot these; keep the width sane.
ONE_HOT_MAX_CARDINALITY = 200
# Beyond this, replace the raw category with its frequency.
HIGH_CARDINALITY_AS_FREQ = ["wpt_name", "subvillage", "scheme_name", "funder", "installer", "ward"]


def build_matrix(train: pd.DataFrame, test: pd.DataFrame):
    """Return (X_train, X_test, y, cat_cols, num_cols) with NN-friendly dtypes."""
    X_train, X_test, y, _ = features.build_features(train, test, version="v1")

    for col in HIGH_CARDINALITY_AS_FREQ:
        if col in X_train.columns:
            counts = pd.concat([X_train[col], X_test[col]]).astype(object).value_counts()
            X_train[f"{col}_freq"] = X_train[col].astype(object).map(counts).astype(float)
            X_test[f"{col}_freq"] = X_test[col].astype(object).map(counts).astype(float)
            X_train = X_train.drop(columns=[col])
            X_test = X_test.drop(columns=[col])

    # Keep only categoricals small enough to one-hot.
    for col in list(X_train.columns):
        if isinstance(X_train[col].dtype, pd.CategoricalDtype):
            if X_train[col].nunique() > ONE_HOT_MAX_CARDINALITY:
                X_train = X_train.drop(columns=[col])
                X_test = X_test.drop(columns=[col])

    cat_cols: list[str] = []
    num_cols: list[str] = []
    for col in X_train.columns:
        if isinstance(X_train[col].dtype, pd.CategoricalDtype):
            # Booleans and strings must not be mixed inside one category dtype.
            X_train[col] = X_train[col].astype(str).replace({"nan": "missing", "<NA>": "missing"})
            X_test[col] = X_test[col].astype(str).replace({"nan": "missing", "<NA>": "missing"})
            cat_cols.append(col)
        else:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
            num_cols.append(col)

    return X_train, X_test, y, cat_cols, num_cols


def make_pipeline(cat_cols: list[str], num_cols: list[str], seed: int) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat_cols),
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                num_cols,
            ),
        ],
        remainder="drop",
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        alpha=1e-4,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=60,
        early_stopping=True,
        n_iter_no_change=5,
        validation_fraction=0.1,
        random_state=seed,
    )
    return Pipeline([("pre", pre), ("mlp", mlp)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural net member for the blend.")
    parser.add_argument("--name", default="N1")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    started = time.time()
    train_df, test_df = data.load_base()
    X_train, X_test, y, cat_cols, num_cols = build_matrix(train_df, test_df)
    folds = data.get_folds(y)
    print(
        f"[{args.name}] {X_train.shape[1]} input columns ({len(cat_cols)} categorical, {len(num_cols)} numeric)"
    )

    oof = np.zeros((len(X_train), len(config.CLASSES)))
    test_probas = []
    for fold in np.unique(folds):
        model = make_pipeline(cat_cols, num_cols, args.seed + int(fold))
        model.fit(X_train[folds != fold], y[folds != fold])
        oof[folds == fold] = model.predict_proba(X_train[folds == fold])
        test_probas.append(model.predict_proba(X_test))
        print(
            f"  fold {fold}: accuracy={accuracy_score(y[folds == fold], oof[folds == fold].argmax(1)):.4f}"
        )

    score = evaluate.accuracy(y, oof)
    print(f"\n[{args.name}] CV accuracy = {score:.4f}  ({time.time() - started:.0f}s)")

    config.ensure_dirs()
    np.save(config.DATA_PROCESSED / f"oof_{args.name}.npy", oof)
    np.save(config.DATA_PROCESSED / f"test_proba_{args.name}.npy", np.mean(test_probas, axis=0))

    pred = np.array(config.CLASSES)[np.mean(test_probas, axis=0).argmax(axis=1)]
    submission = data.load_submission_format()
    out = config.SUBMISSIONS / f"sub_{args.name}.csv"
    evaluate.write_submission(submission, pred, out)
    evaluate.validate_submission(out, submission)
    print(f"submission written to {out}")


if __name__ == "__main__":
    main()
