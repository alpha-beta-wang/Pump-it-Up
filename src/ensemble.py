"""Blend already-trained models without retraining.

Each ``src.train`` run saves ``oof_<name>.npy`` (out-of-fold probabilities) and
``test_proba_<name>.npy`` (5-fold averaged test probabilities). This module
searches blend weights on the OOF matrix and reports an honest estimate: the
weights are chosen on four folds and scored on the held-out one.

Usage:
    uv run python -m src.ensemble --members E5 E7 --name ENS1
"""

from __future__ import annotations

import argparse
from itertools import product

import numpy as np

from . import config, data, evaluate


def _load_members(names: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    oofs, test_probas = [], []
    for name in names:
        oof_path = config.DATA_PROCESSED / f"oof_{name}.npy"
        test_path = config.DATA_PROCESSED / f"test_proba_{name}.npy"
        if not oof_path.exists():
            raise FileNotFoundError(f"missing OOF for {name}: {oof_path}")
        oofs.append(np.load(oof_path))
        test_probas.append(np.load(test_path) if test_path.exists() else None)
    return oofs, test_probas


def _grid(n_members: int, step: float = 0.1) -> list[np.ndarray]:
    steps = int(round(1 / step))
    weights = []
    for combo in product(range(steps + 1), repeat=n_members):
        if sum(combo) == steps:
            weights.append(np.array(combo, dtype=float) / steps)
    return weights


def _blend(probas: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return sum(w * p for w, p in zip(weights, probas, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend saved model probabilities.")
    parser.add_argument("--members", nargs="+", required=True, help="experiment ids, e.g. E5 E7")
    parser.add_argument("--name", default="ENS", help="experiment id for the blend")
    parser.add_argument("--step", type=float, default=0.1, help="weight grid step")
    parser.add_argument(
        "--stack", action="store_true", help="also fit logistic-regression stacking"
    )
    parser.add_argument(
        "--stack-only", action="store_true", help="skip the weight grid, run stacking only"
    )
    parser.add_argument(
        "--equal", action="store_true", help="uniform weights, nothing fitted on OOF"
    )
    args = parser.parse_args()

    train_df, test_df = data.load_base()
    y = train_df[config.TARGET].map(config.CLASS_TO_CODE).to_numpy()
    folds = data.get_folds(y)
    oofs, test_probas = _load_members(args.members)

    missing = [n for n, p in zip(args.members, test_probas, strict=True) if p is None]
    if missing:
        raise FileNotFoundError(f"missing test probabilities for: {missing}")

    single = {name: evaluate.accuracy(y, oof) for name, oof in zip(args.members, oofs, strict=True)}
    print("single models:", {k: round(v, 4) for k, v in single.items()})

    if args.equal:
        uniform = np.full(len(oofs), 1.0 / len(oofs))
        score = evaluate.accuracy(y, _blend(oofs, uniform))
        print(f"equal-weight blend: {score:.4f} (no parameters fitted)")
        blended_test = _blend(test_probas, uniform)
        pred = np.array(config.CLASSES)[blended_test.argmax(axis=1)]
        submission = data.load_submission_format()
        out = config.SUBMISSIONS / f"sub_{args.name}.csv"
        evaluate.write_submission(submission, pred, out)
        evaluate.validate_submission(out, submission)
        print(f"submission written to {out}")
        return

    if args.stack_only:
        _stack(oofs, test_probas, y, folds, name=f"{args.name}_stack")
        return

    weights = _grid(len(oofs), args.step)
    best_all = max(weights, key=lambda w: evaluate.accuracy(y, _blend(oofs, w)))
    print(
        f"best weights (in-sample): {best_all.round(2).tolist()} -> {evaluate.accuracy(y, _blend(oofs, best_all)):.4f}"
    )

    # honest estimate: pick weights on 4 folds, score the 5th
    fold_scores = []
    for fold in np.unique(folds):
        fit_mask = folds != fold
        score_mask = folds == fold
        candidate = max(
            weights,
            key=lambda w: evaluate.accuracy(y[fit_mask], _blend([o[fit_mask] for o in oofs], w)),
        )
        fold_scores.append(
            evaluate.accuracy(y[score_mask], _blend([o[score_mask] for o in oofs], candidate))
        )
    mean_score = float(np.mean(fold_scores))
    print(f"fold-honest blend score: {mean_score:.4f} (+/- {np.std(fold_scores):.4f})")
    print(f"best single member : {max(single.values()):.4f}")

    blended_test = _blend(test_probas, best_all)
    pred = np.array(config.CLASSES)[blended_test.argmax(axis=1)]
    submission = data.load_submission_format()
    out = config.SUBMISSIONS / f"sub_{args.name}.csv"
    evaluate.write_submission(submission, pred, out)
    evaluate.validate_submission(out, submission)
    print(f"submission written to {out}")

    if args.stack:
        _stack(oofs, test_probas, y, folds, name=f"{args.name}_stack")


def _stacked_features(probas: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(probas, axis=1)


def _stack(
    oofs: list[np.ndarray],
    test_probas: list[np.ndarray],
    y: np.ndarray,
    folds: np.ndarray,
    name: str,
) -> None:
    """Logistic-regression stacking on OOF probabilities.

    Evaluated fold-by-fold (fit on the other four folds) so the reported score
    is not the usual optimistic in-sample stacking number.
    """
    from sklearn.linear_model import LogisticRegression

    X_oof = _stacked_features(oofs)
    scores = []
    for fold in np.unique(folds):
        fit_mask = folds != fold
        model = LogisticRegression(max_iter=2000)
        model.fit(X_oof[fit_mask], y[fit_mask])
        scores.append(evaluate.accuracy(y[~fit_mask], model.predict_proba(X_oof[~fit_mask])))
    print(f"stack (fold-honest): {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    final = LogisticRegression(max_iter=2000)
    final.fit(X_oof, y)
    test_pred = final.predict_proba(_stacked_features(test_probas))
    pred = np.array(config.CLASSES)[test_pred.argmax(axis=1)]
    submission = data.load_submission_format()
    out = config.SUBMISSIONS / f"sub_{name}.csv"
    evaluate.write_submission(submission, pred, out)
    evaluate.validate_submission(out, submission)
    print(f"stacked submission written to {out}")


if __name__ == "__main__":
    main()
