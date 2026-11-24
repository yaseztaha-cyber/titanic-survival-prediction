"""Matplotlib/Seaborn plotting helpers with consistent, readable styling."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: figures save to disk without a display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def savefig(path: str | Path) -> Path:
    """Tight-layout + save + close the current figure."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def plot_target_distribution(train_df: pd.DataFrame, path: str | Path) -> Path:
    counts = train_df["Survived"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=["Died (0)", "Survived (1)"], y=[counts.get(0, 0), counts.get(1, 0)]
    )
    plt.title("Target distribution (train)")
    plt.ylabel("Passengers")
    for i, value in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        plt.text(i, value + 8, str(value), ha="center", fontsize=12)
    return savefig(path)


def plot_survival_by_category(
    train_df: pd.DataFrame, column: str, path: str | Path, title: str
) -> Path:
    rates = train_df.groupby(column, observed=True)["Survived"].mean().sort_values()
    plt.figure(figsize=(7, 4.5))
    sns.barplot(x=rates.index.astype(str), y=rates.values)
    plt.title(title)
    plt.ylabel("Survival rate")
    plt.xlabel(column)
    plt.ylim(0, 1)
    return savefig(path)


def plot_age_distribution(train_df: pd.DataFrame, path: str | Path) -> Path:
    plt.figure(figsize=(8, 4.5))
    for label, subset in [
        ("Survived", train_df[train_df["Survived"] == 1]["Age"].dropna()),
        ("Died", train_df[train_df["Survived"] == 0]["Age"].dropna()),
    ]:
        sns.kdeplot(subset, label=label, fill=True, alpha=0.3)
    plt.title("Age distribution by survival")
    plt.xlabel("Age")
    plt.legend()
    return savefig(path)


def plot_fare_distribution(train_df: pd.DataFrame, path: str | Path) -> Path:
    plt.figure(figsize=(8, 4.5))
    sns.boxplot(data=train_df, x="Survived", y="Fare", showfliers=False)
    plt.title("Fare distribution by survival (outliers hidden)")
    plt.xlabel("Survived")
    return savefig(path)


def plot_correlation_heatmap(train_df: pd.DataFrame, path: str | Path) -> Path:
    numeric = train_df.select_dtypes(include="number")
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Numeric correlation heatmap (train)")
    return savefig(path)


def plot_confusion_matrix(cm: pd.DataFrame, path: str | Path) -> Path:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion matrix (out-of-fold predictions)")
    return savefig(path)


def plot_model_comparison(comparison: pd.DataFrame, path: str | Path) -> Path:
    ordered = comparison.sort_values("CV Accuracy", ascending=True)
    plt.figure(figsize=(9, 5))
    plt.errorbar(
        ordered["CV Accuracy"],
        ordered["Model"],
        xerr=ordered["CV Std"],
        fmt="o",
        capsize=4,
    )
    plt.title("Model comparison (5-fold stratified CV accuracy ± std)")
    plt.xlabel("CV accuracy")
    plt.xlim(0.5, 1.0)
    return savefig(path)


def plot_feature_importance(importance: pd.DataFrame, path: str | Path, top_n: int = 15) -> Path:
    top = importance.head(top_n).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance_mean"])
    plt.title(f"Permutation importance (top {top_n}, best model)")
    plt.xlabel("Mean accuracy drop when permuted")
    return savefig(path)
