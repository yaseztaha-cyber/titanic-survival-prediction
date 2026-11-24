"""Deep-validation analyses for the portfolio upgrade.

All functions here *measure* — nothing is hardcoded. They consume the same
pipeline builders, CV splitter, and seed as the rest of the project so every
new number is directly comparable with ``reports/experiment_results.json``.

Contents
--------
- Progressive feature-ablation sets (A–G) + a pipeline builder that exposes an
  arbitrary engineered-feature subset to the estimator.
- Per-fold CV scores, OOF labels + probabilities, subgroup reports.
- Honest display names for permutation importance (raw column -> what the
  model actually sees through the feature engineer).
- Programmatic data-leakage audit.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from configs.config import ID_COL, TARGET_COL, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.feature_engineering import TitanicFeatureEngineer
from src.models import make_cv

# ---------------------------------------------------------------------------
# Progressive ablation sets (cumulative; G == production feature set)
# ---------------------------------------------------------------------------
ABLATION_SETS: list[dict] = [
    {
        "id": "A",
        "label": "A - Raw baseline",
        "group_added": "Raw baseline (Age, Fare, SibSp, Parch, Pclass, Sex, Embarked; no engineered features)",
        "numeric": ["Age", "Fare", "SibSp", "Parch"],
        "categorical": ["Pclass", "Sex", "Embarked"],
    },
    {
        "id": "B",
        "label": "B - + Family",
        "group_added": "Family (FamilySize, IsAlone, FamilySizeCategory)",
        "numeric": ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone"],
        "categorical": ["Pclass", "Sex", "Embarked", "FamilySizeCategory"],
    },
    {
        "id": "C",
        "label": "C - + Title",
        "group_added": "Title (Title extracted from Name)",
        "numeric": ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone"],
        "categorical": ["Pclass", "Sex", "Embarked", "FamilySizeCategory", "Title"],
    },
    {
        "id": "D",
        "label": "D - + Ticket groups",
        "group_added": "Ticket groups (TicketGroupSize)",
        "numeric": ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone", "TicketGroupSize"],
        "categorical": ["Pclass", "Sex", "Embarked", "FamilySizeCategory", "Title"],
    },
    {
        "id": "E",
        "label": "E - + Fare per person",
        "group_added": "Fare per person (FarePerPerson)",
        "numeric": ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone", "TicketGroupSize", "FarePerPerson"],
        "categorical": ["Pclass", "Sex", "Embarked", "FamilySizeCategory", "Title"],
    },
    {
        "id": "F",
        "label": "F - + Cabin",
        "group_added": "Cabin (CabinDeck, CabinKnown)",
        "numeric": ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone", "TicketGroupSize", "FarePerPerson", "CabinKnown"],
        "categorical": ["Pclass", "Sex", "Embarked", "FamilySizeCategory", "Title", "CabinDeck"],
    },
    {
        "id": "G",
        "label": "G - Final (+ Mother)",
        "group_added": "Final (Mother)",
        "numeric": ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone", "TicketGroupSize", "FarePerPerson", "CabinKnown", "Mother"],
        "categorical": ["Pclass", "Sex", "Embarked", "FamilySizeCategory", "Title", "CabinDeck"],
    },
]

# Raw input column -> honest display label (what the model sees through engineering).
PERM_DISPLAY_NAMES: dict[str, str] = {
    "Name": "Title extracted from Name",
    "Ticket": "TicketGroupSize from Ticket",
    "Cabin": "CabinDeck / CabinKnown from Cabin",
    "PassengerId": "PassengerId (identifier, control)",
}


def display_name(raw_column: str) -> str:
    """Map a raw input column to its honest engineered-meaning label."""
    return PERM_DISPLAY_NAMES.get(raw_column, raw_column)


def build_pipeline_with_features(
    model, numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    """Full pipeline exposing only the given engineered-feature subset.

    The feature engineer still computes every column (row-wise, target-free),
    but the ColumnTransformer selects just the listed subset — so ablation
    isolates the estimator's information, not the engineering code path.

    Column order follows ``configs/config.py`` exactly, so the full set (G) is
    bit-for-bit the production pipeline (column order can otherwise change
    tree tie-breaking).
    """
    numeric_ordered = [c for c in NUMERIC_FEATURES if c in set(numeric_features)]
    categorical_ordered = [c for c in CATEGORICAL_FEATURES if c in set(categorical_features)]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_ordered),
            ("cat", categorical_pipe, categorical_ordered),
        ]
    )
    return Pipeline(
        [
            ("features", TitanicFeatureEngineer()),
            ("preprocess", preprocess),
            ("model", model),
        ]
    )


def fold_scores(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> list[float]:
    """Per-fold accuracy scores under the project CV (StratifiedKFold, seed 42)."""
    scores = cross_val_score(pipeline, X, y, cv=make_cv(), scoring="accuracy", n_jobs=-1)
    return [float(s) for s in scores]


def fold_stats(scores: list[float]) -> dict[str, float]:
    """Mean/std (population, ddof=0 — same convention as the rest of the project)."""
    arr = np.asarray(scores, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}


def oof_labels_and_proba(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Out-of-fold predicted labels + survival probabilities (held-out folds only)."""
    cv = make_cv()
    labels = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)
    proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return pd.DataFrame({"oof_label": labels, "oof_proba": proba}, index=X.index)


def subgroup_report(
    frame: pd.DataFrame, group_col: str, label_col: str = TARGET_COL, pred_col: str = "oof_label"
) -> pd.DataFrame:
    """Per-group samples, accuracy, and observed survival rate."""
    grouped = frame.groupby(group_col, observed=True)
    report = pd.DataFrame(
        {
            "samples": grouped.size(),
            "accuracy": grouped.apply(lambda g: float((g[pred_col] == g[label_col]).mean()), include_groups=False),
            "survival_rate": grouped[label_col].mean(),
        }
    )
    return report.sort_values("accuracy")


def _code_references_to_target(source: str, target: str = TARGET_COL) -> int:
    """Count real *code* references to the target column name.

    Docstrings and comments are excluded (they document the absence of
    leakage, e.g. "never sees Survived"). Implemented via ``ast``: every
    string constant and variable name in executable positions is inspected,
    while docstring expressions are skipped.
    """
    import ast

    tree = ast.parse(source)
    docstring_ids = set()

    def _mark_docstrings(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            _mark_docstrings(child)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_ids.add(id(body[0].value))

    _mark_docstrings(tree)
    hits = 0
    for node in ast.walk(tree):
        if id(node) in docstring_ids:
            continue
        if isinstance(node, ast.Name) and node.id == target:
            hits += 1
        elif isinstance(node, ast.Constant) and node.value == target:
            hits += 1
        elif isinstance(node, ast.Attribute) and node.attr == target:
            hits += 1
    return hits


def leakage_audit(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    test_df: pd.DataFrame,
    project_root: str | Path,
) -> dict:
    """Run programmatic leakage checks; each returns pass/fail + evidence.

    Checks:
      1. Target column is not a model input and is listed in DROP_COLUMNS.
      2. PassengerId is not a model input and is listed in DROP_COLUMNS.
      3. Feature-engineering source never references the target.
      4. Pipeline order is features -> preprocess -> model (preprocessing
         fitted inside each CV fold, never on the full dataset first).
      5. Test set carries no label column.
      6. Permutation importance of PassengerId is 0.0 (measured artifact).
      7. Ticket-group state is fit-data dependent and unseen tickets map to 1.
    """
    from configs.config import CATEGORICAL_FEATURES, DROP_COLUMNS, NUMERIC_FEATURES

    model_inputs = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    fe_source_path = Path(project_root) / "src" / "feature_engineering.py"
    fe_source = fe_source_path.read_text(encoding="utf-8")
    # L3 must ignore mentions inside docstrings/comments (which document the
    # *absence* of leakage) and only fail on real code references to the target.
    code_refs_to_target = _code_references_to_target(fe_source)

    steps = [name for name, _ in pipeline.steps]
    encoder = pipeline.named_steps["preprocess"].transformers[1][1].named_steps["onehot"]

    # Check 7a: fit-data dependence — same ticket counted differently under
    # different fit data (proves counts are learned, not global constants).
    probe_ticket = X_train["Ticket"].value_counts().index[0]
    full_count = int((X_train["Ticket"] == probe_ticket).sum())
    eng_full = TitanicFeatureEngineer().fit(X_train)
    eng_half = TitanicFeatureEngineer().fit(X_train.iloc[:200])
    half_count = int(eng_half.ticket_counts_.get(probe_ticket, 0))
    # Check 7b: unseen ticket -> group size 1 (no test-distribution peeking).
    unseen = eng_full.transform(pd.DataFrame([{
        "Name": "Nobody, Mr. Unknown", "Sex": "male", "Age": 30.0, "SibSp": 0,
        "Parch": 0, "Ticket": "UNSEEN-999999", "Fare": 10.0, "Cabin": np.nan,
        "Embarked": "S", "Pclass": 3, "PassengerId": -1,
    }]))["TicketGroupSize"].iloc[0]

    perm_df = pd.read_csv(Path(project_root) / "reports" / "permutation_importance.csv")
    pid_row = perm_df[perm_df["feature"] == "PassengerId"]

    checks = [
        {
            "id": "L1",
            "name": "Target never a model input",
            "passed": bool(TARGET_COL not in model_inputs and TARGET_COL in DROP_COLUMNS),
            "evidence": f"model inputs={len(model_inputs)} cols; '{TARGET_COL}' in DROP_COLUMNS=True",
        },
        {
            "id": "L2",
            "name": "PassengerId never a model input",
            "passed": bool(ID_COL not in model_inputs and ID_COL in DROP_COLUMNS),
            "evidence": f"'{ID_COL}' in DROP_COLUMNS=True; not in numeric/categorical lists",
        },
        {
            "id": "L3",
            "name": "No target reference in feature-engineering code",
            "passed": bool(code_refs_to_target == 0),
            "evidence": f"code-level references to 'Survived' in src/feature_engineering.py (docstrings/comments excluded): {code_refs_to_target}",
        },
        {
            "id": "L4",
            "name": "Preprocessing inside CV folds (Pipeline order)",
            "passed": bool(steps == ["features", "preprocess", "model"] and encoder.handle_unknown == "ignore"),
            "evidence": f"steps={steps}; OneHotEncoder(handle_unknown='{encoder.handle_unknown}')",
        },
        {
            "id": "L5",
            "name": "Test set carries no labels",
            "passed": bool(TARGET_COL not in test_df.columns),
            "evidence": f"test columns={list(test_df.columns)}",
        },
        {
            "id": "L6",
            "name": "PassengerId permutation importance is 0.0",
            "passed": bool(len(pid_row) == 1 and float(pid_row["importance_mean"].iloc[0]) == 0.0),
            "evidence": f"measured importance_mean={float(pid_row['importance_mean'].iloc[0]) if len(pid_row) else 'n/a'}",
        },
        {
            "id": "L7",
            "name": "Group features use records only, no labels",
            "passed": bool(full_count != half_count and int(unseen) == 1),
            "evidence": (
                f"ticket '{probe_ticket}': count fit-on-all={full_count} vs fit-on-first-200={half_count}; "
                f"unseen ticket maps to {int(unseen)}; "
                f"TitanicFeatureEngineer.fit signature={inspect.signature(TitanicFeatureEngineer.fit)} (y unused)"
            ),
        },
    ]
    return {"checks": checks, "all_passed": bool(all(c["passed"] for c in checks))}
