"""Cross-validated training and experiment logging.

Usage:
    uv run python -m src.train --name E1 --model lgbm

Every run appends one row to docs/experiments.csv so that the experiment
history stays reviewable in git (docs/技术方案.md F3).
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config, data, evaluate, features


def make_lgbm(seed: int = config.SEED, **overrides) -> lgb.LGBMClassifier:
    """Baseline LightGBM, matching the recorded E1 run."""
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
    return lgb.LGBMClassifier(**params)


MODELS: dict[str, Callable[..., object]] = {"lgbm": make_lgbm}


def run_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    model_factory: Callable[[], object],
    X_test: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit one model per fold; return out-of-fold probabilities and test mean."""
    n_classes = len(config.CLASSES)
    oof = np.zeros((len(X), n_classes), dtype=float)
    fold_test_probas: list[np.ndarray] = []

    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        valid_mask = folds == fold
        model = model_factory()
        model.fit(X[train_mask], y[train_mask])
        oof[valid_mask] = model.predict_proba(X[valid_mask])
        if X_test is not None:
            fold_test_probas.append(model.predict_proba(X_test))
        fold_acc = evaluate.accuracy(y[valid_mask], oof[valid_mask])
        print(f"  fold {fold}: n_train={train_mask.sum()} accuracy={fold_acc:.4f}")

    test_proba = np.mean(fold_test_probas, axis=0) if fold_test_probas else None
    return oof, test_proba


def log_experiment(record: dict) -> None:
    """Append a row to the shared experiment log."""
    config.ensure_dirs()
    row = pd.DataFrame([record])
    header = not config.EXPERIMENTS_FILE.exists()
    row.to_csv(config.EXPERIMENTS_FILE, mode="a", header=header, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model with fixed folds.")
    parser.add_argument("--name", default="E1", help="experiment id, e.g. E1")
    parser.add_argument("--model", default="lgbm", choices=sorted(MODELS))
    parser.add_argument("--note", default="", help="free-text note for the experiment log")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--no-test", action="store_true", help="skip fitting on the test set")
    args = parser.parse_args()

    started = time.time()
    submission = data.load_submission_format()
    if config.BASE_PARQUET.exists():
        train_df, test_df = data.load_base()
        source = "cached base.parquet"
    else:
        train_df, test_df = data.prepare_base()
        source = "raw csv (run `python -m src.data` to cache)"
    print(f"[{args.name}] data: {source}")

    X_train, X_test, y, class_names = features.build_features(train_df, test_df)
    folds = data.get_folds(y)

    print(
        f"[{args.name}] {X_train.shape[0]} rows x {X_train.shape[1]} features, {len(np.unique(folds))} folds"
    )
    model_factory = lambda: MODELS[args.model](seed=args.seed)  # noqa: E731
    oof, test_proba = run_cv(X_train, y, folds, model_factory, None if args.no_test else X_test)

    accuracy = evaluate.accuracy(y, oof)
    report = evaluate.classification_report(y, oof, class_names)
    print(f"\n[{args.name}] CV accuracy = {accuracy:.4f}")
    print(report)

    config.ensure_dirs()
    np.save(config.DATA_PROCESSED / f"oof_{args.name}.npy", oof)
    if test_proba is not None:
        pred = np.array(class_names)[test_proba.argmax(axis=1)]
        out = config.SUBMISSIONS / f"sub_{args.name}.csv"
        evaluate.write_submission(submission, pred, out)
        evaluate.validate_submission(out, submission)
        print(f"submission written to {out}")

    log_experiment(
        {
            "id": args.name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": args.model,
            "n_features": X_train.shape[1],
            "cv_accuracy": round(accuracy, 4),
            "minutes": round((time.time() - started) / 60, 1),
            "params": json.dumps(
                {"n_estimators": 400, "learning_rate": 0.06, "num_leaves": 48, "seed": args.seed}
            ),
            "note": args.note,
        }
    )


if __name__ == "__main__":
    main()
