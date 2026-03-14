"""Reference quantum states used in photonic polarization experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np

BellStateLabel = Literal["phi_plus", "phi_minus", "psi_plus", "psi_minus"]

__all__ = [
    "BellStateLabel",
    "bell_state",
    "density_matrix",
    "ket_a",
    "ket_d",
    "ket_h",
    "ket_v",
]


def ket_h() -> np.ndarray:
    """Return the horizontal polarization basis ket."""
    raise NotImplementedError


def ket_v() -> np.ndarray:
    """Return the vertical polarization basis ket."""
    raise NotImplementedError


def ket_d() -> np.ndarray:
    """Return the diagonal polarization basis ket."""
    raise NotImplementedError


def ket_a() -> np.ndarray:
    """Return the anti-diagonal polarization basis ket."""
    raise NotImplementedError


def bell_state(label: BellStateLabel = "phi_plus") -> np.ndarray:
    """Return a named two-qubit Bell state ket."""
    raise NotImplementedError


def density_matrix(state_vector: np.ndarray) -> np.ndarray:
    """Construct a density matrix from a state vector."""
    raise NotImplementedError
