"""Loading and sanity-checking the raw Titanic CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from configs.config import EXPECTED_TEST_ROWS, EXPECTED_TRAIN_ROWS


def load_raw(train_path: str | Path, test_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test CSVs and run light schema validation.

    Raises:
        FileNotFoundError: if either CSV is missing.
        ValueError: if row counts or required columns look wrong.
    """
    train_path, test_path = Path(train_path), Path(test_path)
    if not train_path.exists():
        raise FileNotFoundError(f"train.csv not found at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"test.csv not found at {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if len(train_df) != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"Expected {EXPECTED_TRAIN_ROWS} train rows, got {len(train_df)}")
    if len(test_df) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Expected {EXPECTED_TEST_ROWS} test rows, got {len(test_df)}")
    if "Survived" not in train_df.columns:
        raise ValueError("train.csv must contain the 'Survived' target column")
    if "Survived" in test_df.columns:
        raise ValueError("test.csv must not contain labels (target leakage risk)")

    # Train and test must share the same feature schema (minus the target).
    train_features = set(train_df.columns) - {"Survived"}
    if set(test_df.columns) != train_features:
        raise ValueError(
            f"Train/test schema mismatch. "
            f"Only-in-train={train_features - set(test_df.columns)}, "
            f"only-in-test={set(test_df.columns) - train_features}"
        )
    return train_df, test_df


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missing counts and percentages, sorted descending."""
    report = pd.DataFrame(
        {"missing_count": df.isna().sum(), "missing_pct": df.isna().mean() * 100.0}
    )
    return report.sort_values("missing_pct", ascending=False)


def dataset_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Return a small dict of headline facts used in docs and notebooks."""
    return {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "target": "Survived",
        "target_mean": float(train_df["Survived"].mean()),
        "n_survivors": int(train_df["Survived"].sum()),
        "n_non_survivors": int((train_df["Survived"] == 0).sum()),
        "classification_type": "Binary",
        "metric": "Accuracy",
    }
