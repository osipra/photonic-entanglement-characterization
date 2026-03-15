"""Helpers for Bell-state analysis in photonic entanglement experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import metrics
from . import states

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]

__all__ = [
    "axis_observable",
    "bell_state_fidelities",
    "bell_state_projectors",
    "dominant_bell_state",
    "pauli_axis_correlations",
    "two_qubit_correlation",
    "unit_vector",
]


def _as_two_qubit_operator(matrix: ArrayLike) -> ComplexArray:
    """Convert an input into a 4x4 complex matrix for two-qubit analysis."""
    operator = np.asarray(matrix, dtype=np.complex128)
    if operator.shape != (4, 4):
        raise ValueError("Bell analysis helpers require a 4x4 two-qubit matrix.")
    return operator


def _as_direction(direction: ArrayLike) -> RealArray:
    """Convert an analyzer direction into a normalized 3-vector."""
    vector = np.asarray(direction, dtype=np.float64).reshape(-1)
    if vector.shape != (3,):
        raise ValueError("Analyzer directions must be length-3 vectors.")

    norm = np.linalg.norm(vector)
    if np.isclose(norm, 0.0):
        raise ValueError("Analyzer directions must be nonzero.")
    return vector / norm


def unit_vector(theta: float, phi: float) -> RealArray:
    """Return the Bloch-sphere unit vector for polar angle theta and azimuth phi."""
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        dtype=np.float64,
    )


def axis_observable(direction: ArrayLike) -> ComplexArray:
    """Return the single-qubit observable n.sigma for an analyzer direction."""
    nx, ny, nz = _as_direction(direction)
    return (
        nx * states.pauli("X")
        + ny * states.pauli("Y")
        + nz * states.pauli("Z")
    )


def two_qubit_correlation(
    rho: ArrayLike,
    alice_direction: ArrayLike,
    bob_direction: ArrayLike,
) -> float:
    """Compute Tr[(n.sigma tensor m.sigma) rho] for two analyzer directions."""
    rho_two_qubit = states.make_hermitian(_as_two_qubit_operator(rho))
    observable = np.kron(
        axis_observable(alice_direction),
        axis_observable(bob_direction),
    )
    return float(np.real(np.trace(observable @ rho_two_qubit)))


def pauli_axis_correlations(rho: ArrayLike) -> dict[str, float]:
    """Return the notebook-style Exx, Eyy, and Ezz Bell correlators."""
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    return {
        "Exx": two_qubit_correlation(rho, x_axis, x_axis),
        "Eyy": two_qubit_correlation(rho, y_axis, y_axis),
        "Ezz": two_qubit_correlation(rho, z_axis, z_axis),
    }


def bell_state_projectors() -> dict[str, ComplexArray]:
    """Return projectors for the four canonical Bell states."""
    return {
        label: states.density_matrix(psi)
        for label, psi in states.bell_states().items()
    }


def bell_state_fidelities(rho: ArrayLike) -> dict[str, float]:
    """Compute Bell-state fidelities for a two-qubit density matrix."""
    rho_two_qubit = _as_two_qubit_operator(rho)
    return metrics.bell_state_fidelities(rho_two_qubit)


def dominant_bell_state(rho: ArrayLike) -> str:
    """Identify the Bell state with the largest fidelity for a given rho."""
    fidelities = bell_state_fidelities(rho)
    return max(fidelities, key=fidelities.get)
