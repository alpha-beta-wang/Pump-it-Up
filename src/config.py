"""Project-wide constants and paths.

Every module imports paths from here so that a single place controls where
data is read from and written to.
"""

from __future__ import annotations

from pathlib import Path

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
SUBMISSIONS = ROOT / "submissions"
DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"

TRAIN_VALUES = DATA_RAW / "TrainingSetValues.csv"
TRAIN_LABELS = DATA_RAW / "TrainingSetLabels.csv"
TEST_VALUES = DATA_RAW / "TestSetValues.csv"
SUBMISSION_FORMAT = DATA_RAW / "SubmissionFormat.csv"

FOLDS_FILE = DATA_PROCESSED / "folds.npy"
EXPERIMENTS_FILE = DOCS / "experiments.csv"

# --- modelling -------------------------------------------------------------
TARGET = "status_group"
ID = "id"

CLASSES = ["functional", "functional needs repair", "non functional"]
CLASS_TO_CODE = {name: code for code, name in enumerate(CLASSES)}

SEED = 42
N_SPLITS = 5


def ensure_dirs() -> None:
    """Create output directories used by the pipeline."""
    for path in (DATA_PROCESSED, SUBMISSIONS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
