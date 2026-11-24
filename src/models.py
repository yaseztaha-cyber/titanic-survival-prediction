"""Model definitions, cross-validation, and hyperparameter search."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from configs.config import CV_FOLDS, CV_SHUFFLE, RANDOM_STATE
from src.preprocessing import build_full_pipeline


def make_cv() -> StratifiedKFold:
    """Stratified K-fold with shuffling for reproducibility."""
    return StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=CV_SHUFFLE, random_state=RANDOM_STATE
    )


def get_candidate_models() -> dict[str, Pipeline]:
    """Return the model comparison set.

    HistGradientBoostingClassifier is used as the 'advanced boosting' entry:
    it is part of scikit-learn (no extra dependency), handles NaN natively
    (ours are already imputed), and is competitive with XGBoost/LightGBM on
    small tabular data.
    """
    return {
        "Majority Baseline": build_full_pipeline(
            DummyClassifier(strategy="most_frequent"), scale_numeric=False
        ),
        "Logistic Regression": build_full_pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            scale_numeric=True,
        ),
        "Decision Tree": build_full_pipeline(
            DecisionTreeClassifier(random_state=RANDOM_STATE), scale_numeric=False
        ),
        "Random Forest": build_full_pipeline(
            RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
            scale_numeric=False,
        ),
        "Gradient Boosting": build_full_pipeline(
            GradientBoostingClassifier(random_state=RANDOM_STATE), scale_numeric=False
        ),
        "Hist Gradient Boosting": build_full_pipeline(
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            scale_numeric=False,
        ),
    }


def cross_validate_model(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Run stratified CV and return mean/std accuracy plus fit time."""
    cv = make_cv()
    started = time.perf_counter()
    scores = cross_validate(
        pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1, return_train_score=False
    )
    elapsed = time.perf_counter() - started
    test_scores = scores["test_score"]
    return {
        "cv_accuracy_mean": float(np.mean(test_scores)),
        "cv_accuracy_std": float(np.std(test_scores)),
        "cv_scores": [float(s) for s in test_scores],
        "total_fit_time_s": round(elapsed, 2),
    }


def compare_models(
    models: dict, X: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    """Cross-validate every candidate and return a ranked comparison table."""
    rows: list[dict] = []
    for name, pipeline in models.items():
        result = cross_validate_model(pipeline, X, y)
        rows.append(
            {
                "Model": name,
                "CV Accuracy": result["cv_accuracy_mean"],
                "CV Std": result["cv_accuracy_std"],
                "Fit Time (s)": result["total_fit_time_s"],
            }
        )
    table = pd.DataFrame(rows).sort_values("CV Accuracy", ascending=False)
    return table.reset_index(drop=True)


def tune_model(
    pipeline, param_grid: dict, X: pd.DataFrame, y: pd.Series
) -> GridSearchCV:
    """Grid-search a full pipeline (preprocessing inside each fold)."""
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=make_cv(),
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(X, y)
    return search
