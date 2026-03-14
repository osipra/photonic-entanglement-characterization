"""Tomography workflows for photonic state reconstruction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

TomographyMethod = Literal["linear_inversion", "mle"]

__all__ = [
    "TomographyMethod",
    "linear_inversion_tomography",
    "measurement_projectors_from_labels",
    "reconstruct_density_matrix",
]


def measurement_projectors_from_labels(
    basis_labels: Sequence[str],
) -> list[np.ndarray]:
    """Build tomography projectors from a sequence of measurement labels."""
    raise NotImplementedError


def linear_inversion_tomography(
    counts: pd.DataFrame,
    measurement_operators: Sequence[np.ndarray],
) -> np.ndarray:
    """Reconstruct a density matrix with linear inversion."""
    raise NotImplementedError


def reconstruct_density_matrix(
    counts: pd.DataFrame,
    measurement_operators: Sequence[np.ndarray],
    *,
    method: TomographyMethod = "linear_inversion",
) -> np.ndarray:
    """Dispatch to a tomography routine for the requested reconstruction method."""
    raise NotImplementedError
