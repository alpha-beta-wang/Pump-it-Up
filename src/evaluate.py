"""Metrics, reports and submission validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from . import config


def accuracy(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Accuracy of the argmax of predicted probabilities."""
    return float(accuracy_score(y_true, proba.argmax(axis=1)))


def classification_report(y_true: np.ndarray, proba: np.ndarray, class_names: list[str]) -> str:
    """Per-class precision/recall/f1 plus the confusion matrix, as text."""
    pred = proba.argmax(axis=1)
    rows = ["class                      precision  recall    f1     support"]
    cm = confusion_matrix(y_true, pred, labels=range(len(class_names)))
    for code, name in enumerate(class_names):
        tp = cm[code, code]
        support = cm[code].sum()
        pred_n = cm[:, code].sum()
        precision = tp / pred_n if pred_n else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(f"{name:<26} {precision:8.3f} {recall:8.3f} {f1:8.3f} {support:9d}")
    rows.append("")
    rows.append("confusion matrix (rows = truth, cols = prediction)")
    header = "".join(f"{n[:12]:>14}" for n in class_names)
    rows.append(f"{'':<26}{header}")
    for code, name in enumerate(class_names):
        cells = "".join(f"{v:>14d}" for v in cm[code])
        rows.append(f"{name:<26}{cells}")
    return "\n".join(rows)


def write_submission(submission_format: pd.DataFrame, predictions: np.ndarray, path: Path) -> None:
    """Write predictions using the row order of the official template."""
    out = pd.DataFrame({config.ID: submission_format[config.ID], config.TARGET: predictions})
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def validate_submission(path: Path, submission_format: pd.DataFrame) -> None:
    """Fail loudly if the file would be rejected by the leaderboard."""
    df = pd.read_csv(path)
    assert list(df.columns) == [config.ID, config.TARGET], f"unexpected columns: {list(df.columns)}"
    assert len(df) == len(submission_format), f"row count {len(df)} != {len(submission_format)}"
    assert df[config.ID].tolist() == submission_format[config.ID].tolist(), (
        "id order does not match template"
    )
    assert df[config.ID].is_unique, "duplicate ids"
    unknown = set(df[config.TARGET]) - set(config.CLASSES)
    assert not unknown, f"unknown labels: {unknown}"
    assert df[config.TARGET].notna().all(), "missing predictions"
