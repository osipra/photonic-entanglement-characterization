"""Metrics for comparing reconstructed quantum states and experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "concurrence",
    "fidelity",
    "linear_entropy",
    "purity",
    "trace_distance",
]


def purity(rho: np.ndarray) -> float:
    """Compute the purity of a density matrix."""
    raise NotImplementedError


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Compute the quantum state fidelity between two density matrices."""
    raise NotImplementedError


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Compute the trace distance between two density matrices."""
    raise NotImplementedError


def concurrence(rho: np.ndarray) -> float:
    """Compute the concurrence of a two-qubit density matrix."""
    raise NotImplementedError


def linear_entropy(rho: np.ndarray) -> float:
    """Compute the linear entropy of a density matrix."""
    raise NotImplementedError
