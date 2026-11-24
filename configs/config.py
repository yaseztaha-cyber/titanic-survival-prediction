"""Central configuration for the Titanic ML project.

All randomness, paths, and modeling choices flow from here so that
experiments are reproducible and easy to review.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Paths (resolved relative to the repository root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_PATH = DATA_RAW_DIR / "train.csv"
TEST_PATH = DATA_RAW_DIR / "test.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission.csv"
FINAL_PIPELINE_PATH = MODELS_DIR / "final_pipeline.joblib"

# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------
TARGET_COL: str = "Survived"
ID_COL: str = "PassengerId"

EXPECTED_TRAIN_ROWS: int = 891
EXPECTED_TEST_ROWS: int = 418

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
CV_FOLDS: int = 5
CV_SHUFFLE: bool = True

# ---------------------------------------------------------------------------
# Feature groups (post-feature-engineering column names)
# ---------------------------------------------------------------------------
# Binary indicators are treated as numeric (passthrough after imputation).
NUMERIC_FEATURES: list[str] = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize",
    "TicketGroupSize",
    "FarePerPerson",
    "IsAlone",
    "CabinKnown",
    "Mother",
]

CATEGORICAL_FEATURES: list[str] = [
    "Pclass",
    "Sex",
    "Embarked",
    "Title",
    "CabinDeck",
    "FamilySizeCategory",
]

# Columns that must never reach the model.
DROP_COLUMNS: list[str] = [
    TARGET_COL,
    ID_COL,
    "Name",
    "Surname",
    "Ticket",
    "Cabin",
    "AgeGroup",  # kept for EDA only; raw Age is used for modeling
]

# ---------------------------------------------------------------------------
# Hyperparameter search spaces (kept small and defensible)
# ---------------------------------------------------------------------------
RF_PARAM_GRID: dict = {
    "model__n_estimators": [300, 500],
    "model__max_depth": [None, 8, 12],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
    "model__max_features": ["sqrt", None],
}

GB_PARAM_GRID: dict = {
    "model__n_estimators": [200, 300],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3],
    "model__subsample": [0.8, 1.0],
}

HGB_PARAM_GRID: dict = {
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [None, 5],
    "model__max_leaf_nodes": [15, 31],
    "model__l2_regularization": [0.0, 1.0],
}
