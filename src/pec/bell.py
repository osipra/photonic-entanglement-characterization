"""Helpers for Bell-state analysis in photonic entanglement experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "bell_state_fidelities",
    "bell_state_projectors",
    "dominant_bell_state",
]


def bell_state_projectors() -> dict[str, np.ndarray]:
    """Return projectors for the canonical Bell-state basis."""
    raise NotImplementedError


def bell_state_fidelities(rho: np.ndarray) -> dict[str, float]:
    """Compute Bell-state fidelities for a reconstructed density matrix."""
    raise NotImplementedError


def dominant_bell_state(rho: np.ndarray) -> str:
    """Identify the Bell state with the largest overlap."""
    raise NotImplementedError
