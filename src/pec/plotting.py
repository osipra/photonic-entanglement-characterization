"""Plotting helpers for photonic quantum information analyses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "plot_bell_state_fidelities",
    "plot_coincidence_counts",
    "plot_density_matrix",
]


def plot_density_matrix(
    rho: np.ndarray,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> Figure:
    """Visualize a reconstructed density matrix."""
    raise NotImplementedError


def plot_coincidence_counts(
    counts: pd.DataFrame,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> Figure:
    """Plot coincidence counts collected from a lab measurement."""
    raise NotImplementedError


def plot_bell_state_fidelities(
    fidelities: Mapping[str, float],
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> Figure:
    """Plot Bell-state fidelities for a reconstructed state."""
    raise NotImplementedError
