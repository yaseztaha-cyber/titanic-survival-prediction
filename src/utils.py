"""Shared utilities: seeding, I/O helpers, version logging."""

from __future__ import annotations

import platform
import random
from pathlib import Path

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and (when imported) scikit-learn randomness."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (including parents) if it does not exist."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def environment_report() -> dict[str, str]:
    """Return interpreter + key library versions for reproducibility notes."""
    report: dict[str, str] = {"python": platform.python_version()}
    for package in ("pandas", "numpy", "sklearn", "matplotlib", "seaborn"):
        try:
            module = __import__(package)
            report[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            report[package] = "not-installed"
    return report
