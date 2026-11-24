"""Preprocessing pipelines (imputation + encoding), leakage-free by design.

The feature engineer from :mod:`src.feature_engineering` is always the first
pipeline step, followed by a ColumnTransformer. Because the whole chain is a
single scikit-learn Pipeline, cross-validation fits imputers/encoders on each
training fold only.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from configs.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.feature_engineering import TitanicFeatureEngineer


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """Build the column-wise preprocessing step.

    Args:
        scale_numeric: whether to standardise numeric columns. Enabled for
            scale-sensitive models (logistic regression, SVM, KNN) and disabled
            for tree-based models (which do not need monotonic scaling).
    """
    numeric_steps: list = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(steps=numeric_steps)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def build_full_pipeline(model, scale_numeric: bool = False) -> Pipeline:
    """Assemble feature engineering + preprocessing + estimator."""
    return Pipeline(
        steps=[
            ("features", TitanicFeatureEngineer()),
            ("preprocess", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )
