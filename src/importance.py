"""Feature importance from the final model (stage 3 deliverable).

Trains one CatBoost on all training rows with the E5 configuration and dumps a
ranked importance table to ``docs/feature_importance.csv``. Uses ``PredictionValuesChange``
(CatBoost's default) because it is comparable across categorical and numeric
features, unlike split counts.

Usage:
    uv run python -m src.importance
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import config, data, features, models


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank features with the final model.")
    parser.add_argument("--features", default="v1", choices=["v1", "v2"])
    parser.add_argument("--out", default=str(config.DOCS / "feature_importance.csv"))
    args = parser.parse_args()

    train_df, test_df = data.load_base()
    X_train, _, y, _ = features.build_features(train_df, test_df, version=args.features)

    cat_features = models.categorical_columns(X_train)
    frame = models.to_catboost_frame(X_train)
    estimator = models.build("catboost", seed=config.SEED)
    estimator.fit(frame, y, cat_features=cat_features)

    importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": estimator.get_feature_importance(),
            "kind": ["categorical" if c in cat_features else "numeric" for c in X_train.columns],
            "n_unique": [X_train[c].nunique() for c in X_train.columns],
        }
    ).sort_values("importance", ascending=False)

    importance["importance_pct"] = (
        importance["importance"] / importance["importance"].sum() * 100
    ).round(2)
    importance["cum_pct"] = importance["importance_pct"].cumsum().round(2)
    importance.to_csv(args.out, index=False)

    print(importance.head(20).to_string(index=False))
    print(
        f"\nfeatures summing to 90% of importance: "
        f"{int((importance['cum_pct'] < 90).sum()) + 1} / {len(importance)}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
