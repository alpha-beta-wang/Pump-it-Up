"""Cross-validated training and experiment logging.

Usage:
    uv run python -m src.train --name E1p --model lgbm
    uv run python -m src.train --name E5  --model catboost
    uv run python -m src.train --name E6  --model lgbm --encoding te

Every run appends one row to docs/experiments.csv (kept local, not committed)
so the experiment history stays reviewable.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data, encoding, evaluate, features, models

ENCODINGS = ("none", "te")


def _fit_one_fold(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame | None,
    use_te: bool,
    seed: int,
    y_valid: np.ndarray | None = None,
    model_params: dict | None = None,
    early_stop_rounds: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit a single fold, applying target encoding inside the fold only.

    ``early_stop_rounds`` is meant for hyper-parameter search only: stopping on
    the validation fold makes the resulting score optimistic, so the chosen
    configuration must be re-run without it before being reported.
    """
    if use_te:
        X_train, X_valid, X_test = encoding.encode_fold(
            X_train, y_train, X_valid, X_test, seed=seed
        )

    estimator = models.build(model_name, seed=seed, **(model_params or {}))
    if model_name == "catboost":
        cat_features = models.categorical_columns(X_train)
        X_train_cb = models.to_catboost_frame(X_train)
        X_valid_cb = models.to_catboost_frame(X_valid)
        X_test_cb = models.to_catboost_frame(X_test) if X_test is not None else None
        if early_stop_rounds:
            estimator.fit(
                X_train_cb,
                y_train,
                cat_features=cat_features,
                eval_set=(X_valid_cb, y_valid),
                early_stopping_rounds=early_stop_rounds,
                verbose=False,
            )
        else:
            estimator.fit(X_train_cb, y_train, cat_features=cat_features)
        X_valid, X_test = X_valid_cb, X_test_cb
    elif model_name == "xgb":
        X_train = models.to_xgb_frame(X_train)
        X_valid = models.to_xgb_frame(X_valid)
        X_test = models.to_xgb_frame(X_test) if X_test is not None else None
        estimator.fit(X_train, y_train)
    else:
        estimator.fit(X_train, y_train)

    valid_proba = estimator.predict_proba(X_valid)
    test_proba = estimator.predict_proba(X_test) if X_test is not None else None
    return valid_proba, test_proba


def run_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    model_name: str = "lgbm",
    X_test: pd.DataFrame | None = None,
    use_te: bool = False,
    seed: int = config.SEED,
    model_params: dict | None = None,
    early_stop_rounds: int | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit one model per fold; return OOF probabilities and the test mean."""
    n_classes = len(config.CLASSES)
    oof = np.zeros((len(X), n_classes), dtype=float)
    fold_test_probas: list[np.ndarray] = []

    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        valid_mask = folds == fold
        valid_proba, test_proba = _fit_one_fold(
            model_name,
            X[train_mask],
            y[train_mask],
            X[valid_mask],
            X_test,
            use_te,
            seed,
            y_valid=y[valid_mask],
            model_params=model_params,
            early_stop_rounds=early_stop_rounds,
        )
        oof[valid_mask] = valid_proba
        if test_proba is not None:
            fold_test_probas.append(test_proba)
        if verbose:
            print(
                f"  fold {fold}: n_train={train_mask.sum()} "
                f"accuracy={evaluate.accuracy(y[valid_mask], oof[valid_mask]):.4f}"
            )

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
    parser.add_argument("--model", default="lgbm", choices=["lgbm", "catboost", "xgb"])
    parser.add_argument("--encoding", default="none", choices=ENCODINGS)
    parser.add_argument("--features", default="v1", choices=["v1", "v2"])
    parser.add_argument("--note", default="", help="free-text note for the experiment log")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--no-test", action="store_true", help="skip fitting on the test set")
    parser.add_argument("--params", default="{}", help="JSON overrides for the model constructor")
    parser.add_argument("--params-file", default=None, help="read the JSON overrides from a file")
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

    X_train, X_test, y, class_names = features.build_features(
        train_df, test_df, version=args.features
    )
    folds = data.get_folds(y)
    use_te = args.encoding == "te"
    raw_params = Path(args.params_file).read_text() if args.params_file else args.params
    model_params = json.loads(raw_params)

    print(
        f"[{args.name}] model={args.model} encoding={args.encoding} features={args.features} "
        f"{X_train.shape[0]} rows x {X_train.shape[1]} features, {len(np.unique(folds))} folds"
    )
    oof, test_proba = run_cv(
        X_train,
        y,
        folds,
        model_name=args.model,
        X_test=None if args.no_test else X_test,
        use_te=use_te,
        seed=args.seed,
        model_params=model_params,
    )

    accuracy = evaluate.accuracy(y, oof)
    print(f"\n[{args.name}] CV accuracy = {accuracy:.4f}")
    print(evaluate.classification_report(y, oof, class_names))

    config.ensure_dirs()
    np.save(config.DATA_PROCESSED / f"oof_{args.name}.npy", oof)
    if test_proba is not None:
        # Saved so later stages can blend models without retraining.
        np.save(config.DATA_PROCESSED / f"test_proba_{args.name}.npy", test_proba)
        pred = np.array(class_names)[test_proba.argmax(axis=1)]
        out = config.SUBMISSIONS / f"sub_{args.name}.csv"
        evaluate.write_submission(submission, pred, out)
        evaluate.validate_submission(out, submission)
        print(f"submission written to {out}")

    log_experiment(
        {
            "id": args.name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": f"{args.model}+{args.encoding}",
            "n_features": X_train.shape[1],
            "cv_accuracy": round(accuracy, 4),
            "minutes": round((time.time() - started) / 60, 1),
            "params": json.dumps(
                {
                    "seed": args.seed,
                    "encoding": args.encoding,
                    "features": args.features,
                    **model_params,
                }
            ),
            "note": args.note,
        }
    )


if __name__ == "__main__":
    main()
