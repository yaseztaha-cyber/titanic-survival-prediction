"""Evaluation helpers: metrics, confusion matrices, error analysis, submission."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_predict

from configs.config import ID_COL, TARGET_COL
from src.models import make_cv


def classification_report_dict(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Accuracy + precision/recall/F1 (competition uses accuracy; rest aid diagnosis)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def confusion_matrix_labeled(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Confusion matrix with Titanic-meaningful row/column labels."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=["Actual: Died (0)", "Actual: Survived (1)"],
        columns=["Predicted: Died (0)", "Predicted: Survived (1)"],
    )


def cross_val_predictions(pipeline, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Out-of-fold predictions for unbiased error analysis."""
    preds = cross_val_predict(pipeline, X, y, cv=make_cv(), n_jobs=-1)
    return pd.Series(preds, index=X.index, name="oof_prediction")


def error_analysis_frame(
    X_raw: pd.DataFrame, y_true: pd.Series, oof_pred: pd.Series
) -> pd.DataFrame:
    """Join raw features with labels and OOF predictions, flagging errors."""
    frame = X_raw.copy()
    frame[TARGET_COL] = y_true.values
    frame["oof_prediction"] = oof_pred.values
    frame["is_error"] = (frame[TARGET_COL] != frame["oof_prediction"]).astype(int)
    return frame


def permutation_feature_importance(
    pipeline, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10
) -> pd.DataFrame:
    """Permutation importance over the raw input columns.

    ``permutation_importance`` permutes columns of the data passed to the
    pipeline (the raw train frame), so importances map 1:1 to input columns
    such as 'Sex' or 'Fare'. This answers 'which input column matters?' rather
    than 'which one-hot dummy matters?', which is more useful for this report.
    Impurity-based importance is reported separately in the notebooks.
    """
    from configs.config import RANDOM_STATE

    result = permutation_importance(
        pipeline, X, y, n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1
    )
    table = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def impurity_feature_importance(pipeline) -> pd.DataFrame:
    """Tree impurity importance mapped to post-preprocessing feature names."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Underlying estimator has no feature_importances_")
    preprocess = pipeline.named_steps["preprocess"]
    feature_names = list(preprocess.get_feature_names_out())
    table = pd.DataFrame(
        {"feature": feature_names, "importance": list(model.feature_importances_)}
    )
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def validate_submission(
    submission: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    """Enforce the Kaggle submission contract; raises on any violation."""
    assert list(submission.columns) == [ID_COL, TARGET_COL], (
        f"Columns must be {[ID_COL, TARGET_COL]}, got {list(submission.columns)}"
    )
    assert len(submission) == len(test_df), (
        f"Expected {len(test_df)} rows, got {len(submission)}"
    )
    assert submission[ID_COL].equals(test_df[ID_COL]), "PassengerId order mismatch"
    assert submission[TARGET_COL].isin([0, 1]).all(), "Survived must be binary 0/1"
    assert submission[TARGET_COL].notna().all(), "Missing predictions found"


def save_submission(
    predictions, test_df: pd.DataFrame, path: str | Path
) -> Path:
    """Build, validate, and write submission.csv (no index column)."""
    submission = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: predictions})
    validate_submission(submission, test_df)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out, index=False)
    return out
