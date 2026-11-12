"""Random-search hyper-parameter tuning.

Deliberately dependency-free (no Optuna) and deliberately small: CatBoost on
59k rows takes minutes per fold, so a 100-trial search is not affordable. The
search uses early stopping to cut cost, which makes its scores optimistic -
always confirm the winner with ``src/train.py`` afterwards.

Usage:
    uv run python -m src.tune --trials 8 --features v2
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np

from . import config, data, evaluate, features, train

SEARCH_SPACE = {
    "depth": [6, 7, 8],
    "learning_rate": [0.04, 0.06, 0.08],
    "l2_leaf_reg": [1.0, 3.0, 6.0],
    "max_ctr_complexity": [2, 4],
    "one_hot_max_size": [2, 10],
    "iterations": [2000],
}


def sample_params(rng: np.random.Generator) -> dict:
    return {key: values[rng.integers(len(values))] for key, values in SEARCH_SPACE.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Random search for CatBoost params.")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--features", default="v2", choices=["v1", "v2"])
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--early-stop", type=int, default=100)
    args = parser.parse_args()

    train_df, test_df = data.load_base()
    X_train, _, y, _ = features.build_features(train_df, test_df, version=args.features)
    folds = data.get_folds(y)

    rng = np.random.default_rng(args.seed)
    results: list[tuple[float, dict]] = []

    for trial in range(1, args.trials + 1):
        params = sample_params(rng)
        started = time.time()
        oof, _ = train.run_cv(
            X_train,
            y,
            folds,
            model_name="catboost",
            use_te=False,
            seed=args.seed,
            model_params=params,
            early_stop_rounds=args.early_stop,
            verbose=False,
        )
        accuracy = evaluate.accuracy(y, oof)
        results.append((accuracy, params))
        print(f"trial {trial:>2}  acc={accuracy:.4f}  ({time.time() - started:.0f}s)  {params}")

        train.log_experiment(
            {
                "id": f"T{trial}",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": "catboost-tune",
                "n_features": X_train.shape[1],
                "cv_accuracy": round(accuracy, 4),
                "minutes": round((time.time() - started) / 60, 1),
                "params": json.dumps(params),
                "note": f"random search on features={args.features}, early stopping (optimistic)",
            }
        )

    best = max(results, key=lambda item: item[0])
    print("\nbest (optimistic, early-stopped):", f"{best[0]:.4f}")
    print(json.dumps(best[1], indent=2))
    print(
        "\nre-run with: uv run python -m src.train --name E9 --model catboost "
        f"--features {args.features} with the params above"
    )


if __name__ == "__main__":
    main()
