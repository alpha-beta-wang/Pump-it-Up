"""Partition-wise ensemble members.

Instead of one global model, train one model per partition (administrative region,
water-point type, or an unsupervised cluster), then let the member participate in a
family-equal blend. Measured on this dataset: single partition models are *not*
better than a global model, but as blend members they add +0.0044 (see docs/技术方案.md §2.3).

Design rules (same as the rest of ``src/``):
* partitions are derived from features only, never from labels;
* a group whose training rows fall below ``--min-rows`` falls back to the
  per-fold global model, so every OOF row and test row still has a prediction;
* test probabilities are the average of the fold models, matching ``src.train``.

Usage:
    uv run python -m src.cluster_model --name CLRF_region_k10 --by region --folds 10
    uv run python -m src.cluster_model --name CLRF_wptype_k10 --by wptype --folds 10
    uv run python -m src.cluster_model --name CLRF_kmeans12_k10 --by kmeans --clusters 12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from . import config, data, evaluate, features, models, train

# Columns used by the unsupervised partition (continuous only, no labels involved).
CLUSTER_COLUMNS = [
    "gps_height",
    "population",
    "age",
    "amount_tsh",
    "longitude",
    "latitude",
]
LOG_COLUMNS = ("population", "amount_tsh", "age")

# Partition keys that come straight from a categorical column.
PARTITION_COLUMNS = {
    "region": "region",
    "wptype": "waterpoint_type_group",
    "lga": "lga",
}


def _kmeans_labels(
    X_train: pd.DataFrame, X_test: pd.DataFrame, n_clusters: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster on continuous features; fit on train+test so both sides share centres.

    Fitting on the concatenation is legitimate: it uses feature values only, so no
    label information can leak, and it guarantees the test rows map to the same
    cluster ids as the training rows.
    """

    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[[c for c in CLUSTER_COLUMNS if c in frame.columns]].astype(float)
        for col in LOG_COLUMNS:
            if col in out.columns:
                out[col] = np.log1p(out[col].clip(lower=0))
        return out

    combined = pd.concat([prepare(X_train), prepare(X_test)], ignore_index=True)
    combined = combined.fillna(combined.median())
    scaled = StandardScaler().fit_transform(combined)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(
        scaled
    )
    return labels[: len(X_train)].astype(str), labels[len(X_train) :].astype(str)


def make_partition(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    by: str = "region",
    n_clusters: int = 12,
    seed: int = config.SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(labels_train, labels_test)`` as string arrays.

    ``by`` is one of ``region`` / ``wptype`` / ``lga`` / ``kmeans``. Strings are used
    instead of integer codes so that train and test always agree on the mapping.
    """
    if by == "kmeans":
        return _kmeans_labels(X_train, X_test, n_clusters=n_clusters, seed=seed)
    if by not in PARTITION_COLUMNS:
        raise ValueError(
            f"unknown partition: {by!r}; expected one of {sorted(PARTITION_COLUMNS)} + kmeans"
        )
    column = PARTITION_COLUMNS[by]
    return (
        np.asarray(X_train[column].astype(str).to_numpy(), dtype=str),
        np.asarray(X_test[column].astype(str).to_numpy(), dtype=str),
    )


def _aligned_proba(estimator, frame: pd.DataFrame, n_classes: int) -> np.ndarray:
    """``predict_proba`` padded to ``n_classes`` columns.

    A small partition (or a rare class inside a fold) can miss a label entirely,
    in which case sklearn returns fewer columns. Re-indexing by ``classes_`` keeps
    the columns aligned with ``config.CLASSES``.
    """
    proba = estimator.predict_proba(frame)
    if proba.shape[1] == n_classes:
        return proba
    aligned = np.zeros((proba.shape[0], n_classes))
    aligned[:, np.asarray(estimator.classes_, dtype=int)] = proba
    return aligned


def run_member(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    labels_train: np.ndarray,
    labels_test: np.ndarray,
    model_name: str = "rf",
    min_rows: int = 300,
    seed: int = config.SEED,
    model_params: dict | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train one model per (fold, partition); return OOF, test proba and coverage.

    ``coverage`` marks the OOF rows predicted by a partition model rather than by
    the global fallback, which is what the reported accuracy actually measures.
    """
    n_classes = len(config.CLASSES)
    overrides = model_params or {}

    # Each model family wants a different input frame: CatBoost takes strings plus
    # an explicit ``cat_features`` list, the sklearn forests need ordinal codes.
    if model_name == "catboost":
        X = models.to_catboost_frame(X_train)
        X_test_ord = models.to_catboost_frame(X_test)
        fit_kwargs = {"cat_features": models.categorical_columns(X_train)}
    elif model_name == "lgbm":
        # LightGBM reads pandas category dtype directly.
        X = X_train.copy()
        X_test_ord = X_test.copy()
        fit_kwargs = {}
    else:
        X = models.to_ordinal_frame(X_train)
        X_test_ord = models.to_ordinal_frame(X_test)
        fit_kwargs = {}

    oof = np.zeros((len(y), n_classes))
    covered = np.zeros(len(y), dtype=bool)
    test_sum = np.zeros((len(labels_test), n_classes))
    test_count = np.zeros(len(labels_test))

    for fold in sorted(np.unique(folds)):
        train_mask, valid_mask = folds != fold, folds == fold

        # Global model for this fold: fallback for groups that are too small.
        fallback = models.build(model_name, seed=seed, **overrides)
        fallback.fit(X[train_mask], y[train_mask], **fit_kwargs)
        oof[valid_mask] = _aligned_proba(fallback, X[valid_mask], n_classes)
        test_sum += _aligned_proba(fallback, X_test_ord, n_classes)
        test_count += 1

        for group in np.unique(labels_train):
            fit_idx = np.where(train_mask & (labels_train == group))[0]
            if len(fit_idx) < min_rows:
                continue
            pred_idx = np.where(valid_mask & (labels_train == group))[0]
            estimator = models.build(model_name, seed=seed, **overrides)
            estimator.fit(X.iloc[fit_idx], y[fit_idx], **fit_kwargs)
            if len(pred_idx):
                oof[pred_idx] = _aligned_proba(estimator, X.iloc[pred_idx], n_classes)
                covered[pred_idx] = True

            test_mask = labels_test == group
            if test_mask.any():
                test_sum[test_mask] += _aligned_proba(
                    estimator, X_test_ord[test_mask], n_classes
                )
                test_count[test_mask] += 1

        if verbose:
            accuracy = float((oof[valid_mask].argmax(1) == y[valid_mask]).mean())
            print(f"  fold {fold}: accuracy={accuracy:.4f}")

    test_proba = test_sum / np.maximum(test_count, 1)[:, None]
    return oof, test_proba, covered


def main() -> None:
    parser = argparse.ArgumentParser(description="Partition-wise ensemble member.")
    parser.add_argument("--name", required=True, help="member id, e.g. CLRF_region_k10")
    parser.add_argument(
        "--by", default="region", choices=[*PARTITION_COLUMNS, "kmeans"]
    )
    parser.add_argument(
        "--folds", type=int, default=config.N_SPLITS, help="cross-validation folds"
    )
    parser.add_argument(
        "--model", default="rf", choices=["rf", "et", "catboost", "lgbm"]
    )
    parser.add_argument("--features", default="v1", choices=["v1", "v2", "v3"])
    parser.add_argument(
        "--clusters", type=int, default=12, help="only used by --by kmeans"
    )
    parser.add_argument(
        "--min-rows", type=int, default=300, help="smallest group that gets a model"
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    started = time.time()
    train_df, test_df = data.load_base()
    X_train, X_test, y, class_names = features.build_features(
        train_df, test_df, version=args.features
    )
    folds = (
        data.get_folds(y)
        if args.folds == config.N_SPLITS
        else data.make_folds(y, n_splits=args.folds, seed=config.SEED)
    )
    labels_train, labels_test = make_partition(
        X_train, X_test, by=args.by, n_clusters=args.clusters, seed=args.seed
    )
    model_params = (
        json.loads(Path(args.params_file).read_text()) if args.params_file else {}
    )

    print(
        f"[{args.name}] by={args.by} model={args.model} folds={args.folds} "
        f"groups={len(np.unique(labels_train))} min_rows={args.min_rows}"
    )
    oof, test_proba, covered = run_member(
        X_train,
        X_test,
        y,
        folds,
        labels_train,
        labels_test,
        model_name=args.model,
        min_rows=args.min_rows,
        seed=args.seed,
        model_params=model_params,
    )

    accuracy = float((oof.argmax(1) == y).mean())
    print(f"\n[{args.name}] CV accuracy = {accuracy:.4f}")
    print(f"[{args.name}] 分区模型覆盖率 = {covered.mean():.1%}")

    config.ensure_dirs()
    np.save(config.DATA_PROCESSED / f"oof_{args.name}.npy", oof)
    np.save(config.DATA_PROCESSED / f"test_proba_{args.name}.npy", test_proba)

    submission = data.load_submission_format()
    pred = np.array(class_names)[test_proba.argmax(1)]
    out = config.SUBMISSIONS / f"sub_{args.name}.csv"
    evaluate.write_submission(submission, pred, out)
    evaluate.validate_submission(out, submission)
    print(f"[{args.name}] submission written to {out}")

    train.log_experiment(
        {
            "id": args.name,
            "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
            "model": f"{args.model}-partition+{args.by}",
            "n_features": X_train.shape[1],
            "cv_accuracy": round(accuracy, 4),
            "minutes": round((time.time() - started) / 60, 1),
            "params": (
                f'{{"folds": {args.folds}, "features": "{args.features}", '
                f'"clusters": {args.clusters}, "min_rows": {args.min_rows}, "seed": {args.seed}}}'
            ),
            "note": f"partition={args.by}, coverage={covered.mean():.3f}",
        }
    )
    print(f"[{args.name}] {time.time() - started:.0f}s total")


if __name__ == "__main__":
    main()
