"""Feature engineering with strict leakage prevention.

Design principles
-----------------
1. Every transformation is a scikit-learn compatible transformer, so statistics
   (ticket counts, etc.) are learned on training folds only and applied to
   validation/test data via ``transform``.
2. No transformer touches the ``Survived`` column.
3. Each feature has a documented rationale (see docstrings and README).

Features
--------
- Title: social title parsed from Name, rare titles grouped to 'Rare'.
- Surname: extracted for EDA only (never fed to the model: high cardinality).
- FamilySize = SibSp + Parch + 1; IsAlone; FamilySizeCategory.
- Mother: heuristic flag for likely mothers (female, adult, parched, Mrs/Miss).
- TicketGroupSize: number of passengers sharing a ticket (fit on train only).
- FarePerPerson = Fare / TicketGroupSize (safe division).
- CabinDeck: first letter of Cabin; CabinKnown indicator.
- AgeGroup: binned age for EDA only (model uses raw Age).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TITLE_REGEX = re.compile(r",\s*([^\.]+)\.")

# Normalisation map: variants -> canonical title. Anything unlisted -> 'Rare'.
TITLE_MAPPING: dict[str, str] = {
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
}


def extract_title(name: str) -> str:
    """Extract and normalise the social title from a passenger name."""
    if not isinstance(name, str):
        return "Rare"
    match = TITLE_REGEX.search(name)
    if not match:
        return "Rare"
    raw = match.group(1).strip()
    return TITLE_MAPPING.get(raw, "Rare")


def extract_surname(name: str) -> str:
    """Extract surname (text before the comma) for family analysis only."""
    if not isinstance(name, str) or "," not in name:
        return "Unknown"
    return name.split(",")[0].strip()


def extract_deck(cabin: object) -> str:
    """Extract cabin deck letter; missing cabins become 'Missing'."""
    if not isinstance(cabin, str) or not cabin.strip():
        return "Missing"
    return cabin.strip()[0].upper()


def categorize_family_size(size: int) -> str:
    """Bin family size: Alone (1), Small (2-4), Large (5+)."""
    if size == 1:
        return "Alone"
    if size <= 4:
        return "Small"
    return "Large"


def categorize_age(age: float) -> str:
    """Bin age for EDA: Child / Teen / Adult / Senior / Unknown."""
    if pd.isna(age):
        return "Unknown"
    if age < 13:
        return "Child"
    if age < 19:
        return "Teen"
    if age < 60:
        return "Adult"
    return "Senior"


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless + stateful feature engineering in one transformer.

    The only *fitted* state is ``ticket_counts_`` (ticket -> group size observed
    in the training data). Everything else is a row-wise deterministic function,
    so there is no target leakage: ``fit`` never sees ``Survived``.

    Why fit ticket counts on train only?
        At prediction time for genuinely new passengers we cannot know the full
        future ticket distribution. Learning counts from the training fold and
        mapping unseen test tickets to 1 is the honest, leakage-free choice.
        (Counting on train+test combined would peek at test distribution.)
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TitanicFeatureEngineer":
        if "Ticket" not in X.columns:
            raise ValueError("TitanicFeatureEngineer expects a 'Ticket' column")
        self.ticket_counts_ = X["Ticket"].value_counts().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["Title"] = df["Name"].apply(extract_title)
        df["Surname"] = df["Name"].apply(extract_surname)
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        df["FamilySizeCategory"] = df["FamilySize"].apply(categorize_family_size)
        df["CabinDeck"] = df["Cabin"].apply(extract_deck)
        df["CabinKnown"] = df["Cabin"].notna().astype(int)
        df["AgeGroup"] = df["Age"].apply(categorize_age)

        counts: dict = getattr(self, "ticket_counts_", {})
        df["TicketGroupSize"] = df["Ticket"].map(counts).fillna(1).astype(int)
        # Fare is per-ticket on the Titanic; dividing by group size approximates
        # the per-person price. Guard against zero/NaN fares.
        fare = df["Fare"].fillna(0.0).clip(lower=0.0)
        df["FarePerPerson"] = fare / df["TicketGroupSize"].clip(lower=1)

        # Mother heuristic: adult female with children aboard and a Mrs/Miss
        # title. Documented as approximate -- it will misclassify some
        # guardians/step-mothers, which is why it is a small binary flag rather
        # than a core feature.
        df["Mother"] = (
            (df["Sex"] == "female")
            & (df["Parch"] > 0)
            & (df["Age"].fillna(0) > 18)
            & (df["Title"].isin(["Mrs", "Miss"]))
        ).astype(int)
        return df


def add_features_with_combined_ticket_counts(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience helper for EDA plots only (not for model training).

    Computes TicketGroupSize from the combined train+test ticket distribution
    so exploratory plots reflect true travelling groups. Model training must use
    :class:`TitanicFeatureEngineer` inside a Pipeline instead, to avoid peeking
    at test distribution during fitting.
    """
    combined_counts = pd.concat([train_df["Ticket"], test_df["Ticket"]]).value_counts().to_dict()

    def _enrich(df: pd.DataFrame) -> pd.DataFrame:
        engineer = TitanicFeatureEngineer()
        engineer.ticket_counts_ = combined_counts
        return engineer.transform(df)

    return _enrich(train_df), _enrich(test_df)
