"""Pseudo-labelling experiment (docs/技术方案.md D5).

Takes the confident test predictions produced by the blend, adds them to the
training folds as extra rows, and measures the effect on held-out folds. The
pseudo-labels come from model outputs only - never from the official test
labels, which are not available - so the fold-honest score stays an honest
estimate.

Usage:
    uv run python -m src.pseudo --members E5b E7 E9 E10 E11 --threshold 0.95
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config, data, evaluate, features, models


def blend_test_proba(members: list[str]) -> np.ndarray:
    """Fit the stacker on OOF probabilities and predict the test set."""
    oofs = [np.load(config.DATA_PROCESSED / f"oof_{m}.npy") for m in members]
    test_probas = [np.load(config.DATA_PROCESSED / f"test_proba_{m}.npy") for m in members]
    train_df, _ = data.load_base()
    y = train_df[config.TARGET].map(config.CLASS_TO_CODE).to_numpy()

    X_oof = np.concatenate(oofs, axis=1)
    X_test = np.concatenate(test_probas, axis=1)
    stacker = LogisticRegression(max_iter=2000).fit(X_oof, y)
    return stacker.predict_proba(X_test), y


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo-label the confident test rows.")
    parser.add_argument("--members", nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--model", default="lgbm", choices=["lgbm", "catboost"])
    parser.add_argument("--name", default="P1")
    args = parser.parse_args()

    train_df, test_df = data.load_base()
    X_train, X_test, y, class_names = features.build_features(train_df, test_df, version="v1")
    folds = data.get_folds(y)

    test_proba, _ = blend_test_proba(args.members)
    confidence = test_proba.max(axis=1)
    keep = confidence >= args.threshold
    pseudo_X = X_test[keep]
    pseudo_y = test_proba.argmax(axis=1)[keep]
    print(
        f"pseudo-labelled test rows: {keep.sum()} / {len(keep)} ({keep.mean():.1%}) at p>={args.threshold}"
    )
    print("pseudo label distribution:", np.bincount(pseudo_y, minlength=3).tolist())

    oof = np.zeros((len(X_train), len(config.CLASSES)))
    for fold in np.unique(folds):
        train_mask = folds != fold
        X_fold = pd.concat([X_train[train_mask], pseudo_X], ignore_index=True)
        y_fold = np.concatenate([y[train_mask], pseudo_y])

        estimator = models.build(args.model, seed=config.SEED)
        if args.model == "catboost":
            cat_features = models.categorical_columns(X_fold)
            X_fold_cb = models.to_catboost_frame(X_fold)
            X_valid_cb = models.to_catboost_frame(X_train[folds == fold])
            estimator.fit(X_fold_cb, y_fold, cat_features=cat_features)
            oof[folds == fold] = estimator.predict_proba(X_valid_cb)
        else:
            estimator.fit(X_fold, y_fold)
            oof[folds == fold] = estimator.predict_proba(X_train[folds == fold])

    score = evaluate.accuracy(y, oof)
    print(f"[{args.name}] CV accuracy with pseudo-labels = {score:.4f}")
    np.save(config.DATA_PROCESSED / f"oof_{args.name}.npy", oof)


if __name__ == "__main__":
    main()
