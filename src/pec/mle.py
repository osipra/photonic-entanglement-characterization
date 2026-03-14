"""Maximum-likelihood estimation utilities for quantum state reconstruction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

__all__ = [
    "fit_density_matrix_mle",
    "mle_diagnostics",
]


def fit_density_matrix_mle(
    counts: pd.DataFrame,
    measurement_operators: Sequence[np.ndarray],
    *,
    initial_state: np.ndarray | None = None,
    max_iterations: int = 1_000,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Estimate a physical density matrix with maximum-likelihood fitting."""
    raise NotImplementedError


def mle_diagnostics(
    counts: pd.DataFrame,
    measurement_operators: Sequence[np.ndarray],
    estimate: np.ndarray,
) -> dict[str, Any]:
    """Summarize goodness-of-fit information for an MLE solution."""
    raise NotImplementedError
