"""Input and output helpers for lab analysis data products."""

from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

PathType = str | PathLike[str]

__all__ = [
    "PathType",
    "load_counts_table",
    "load_density_matrix",
    "save_counts_table",
    "save_density_matrix",
]


def load_counts_table(
    path: PathType,
    *,
    index_col: str | None = None,
) -> pd.DataFrame:
    """Load a coincidence-count table from disk."""
    raise NotImplementedError


def save_counts_table(counts: pd.DataFrame, path: PathType) -> None:
    """Write a coincidence-count table to disk."""
    raise NotImplementedError


def load_density_matrix(path: PathType) -> np.ndarray:
    """Load a saved density matrix from disk."""
    raise NotImplementedError


def save_density_matrix(rho: np.ndarray, path: PathType) -> None:
    """Write a density matrix to disk."""
    raise NotImplementedError
