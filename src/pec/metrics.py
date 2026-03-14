"""State metrics for photonic polarization and entanglement analyses."""

from __future__ import annotations

import numpy as np
import numpy.linalg as npl
from numpy.typing import ArrayLike, NDArray

from . import states

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]

__all__ = [
    "bell_state_fidelities",
    "concurrence",
    "fidelity",
    "fidelity_pure",
    "linear_entropy",
    "purity",
    "state_eigenvalues",
    "trace_distance",
]


def _as_complex_matrix(matrix: ArrayLike) -> ComplexArray:
    """Convert a matrix-like input into a dense complex array."""
    return np.asarray(matrix, dtype=np.complex128)


def _as_complex_vector(vector: ArrayLike) -> ComplexArray:
    """Convert a ket-like input into a flat complex array."""
    return np.asarray(vector, dtype=np.complex128).reshape(-1)


def _sqrtm_psd(matrix: ArrayLike) -> ComplexArray:
    """Return the principal square root of a positive semidefinite matrix."""
    matrix_hermitian = states.make_hermitian(matrix)
    eigenvalues, eigenvectors = npl.eigh(matrix_hermitian)
    clipped = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * np.sqrt(clipped)) @ eigenvectors.conj().T


def purity(rho: ArrayLike) -> float:
    """Compute the purity Tr(rho^2) of a density matrix."""
    rho_hermitian = states.make_hermitian(rho)
    return float(np.real(np.trace(rho_hermitian @ rho_hermitian)))


def fidelity_pure(rho: ArrayLike, psi: ArrayLike) -> float:
    """Compute F = <psi|rho|psi> for a density matrix and pure target ket."""
    rho_hermitian = states.make_hermitian(rho)
    psi_vector = _as_complex_vector(psi)
    return float(np.real(np.conjugate(psi_vector) @ (rho_hermitian @ psi_vector)))


def fidelity(rho: ArrayLike, sigma: ArrayLike) -> float:
    """Compute the Uhlmann fidelity between two density matrices."""
    rho_hermitian = states.make_hermitian(rho)
    sigma_hermitian = states.make_hermitian(sigma)
    sigma_sqrt = _sqrtm_psd(sigma_hermitian)
    overlap_matrix = sigma_sqrt @ rho_hermitian @ sigma_sqrt
    overlap = float(np.real(np.trace(_sqrtm_psd(overlap_matrix))))
    return overlap**2


def bell_state_fidelities(rho: ArrayLike) -> dict[str, float]:
    """Compute overlaps with the four canonical Bell states."""
    return {
        label: fidelity_pure(rho, psi)
        for label, psi in states.bell_states().items()
    }


def state_eigenvalues(rho: ArrayLike) -> RealArray:
    """Return the eigenvalues of the Hermitian part of a density matrix."""
    return np.asarray(npl.eigvalsh(states.make_hermitian(rho)), dtype=np.float64)


def trace_distance(rho: ArrayLike, sigma: ArrayLike) -> float:
    """Compute the trace distance between two density matrices."""
    delta = states.make_hermitian(_as_complex_matrix(rho) - _as_complex_matrix(sigma))
    return float(0.5 * np.sum(np.abs(npl.eigvalsh(delta))))


def linear_entropy(rho: ArrayLike) -> float:
    """Compute the unscaled linear entropy 1 - Tr(rho^2)."""
    return 1.0 - purity(rho)


def concurrence(rho: ArrayLike) -> float:
    """Compute the concurrence of a two-qubit density matrix."""
    rho_hermitian = states.make_hermitian(rho)
    if rho_hermitian.shape != (4, 4):
        raise ValueError("concurrence requires a 4x4 two-qubit density matrix.")

    sigma_y = states.pauli("Y")
    spin_flip = np.kron(sigma_y, sigma_y)
    rho_tilde = spin_flip @ rho_hermitian.conj() @ spin_flip
    eigenvalues = np.linalg.eigvals(rho_hermitian @ rho_tilde)
    roots = np.sqrt(np.clip(np.real(np.real_if_close(eigenvalues)), 0.0, None))
    ordered = np.sort(roots)[::-1]
    return max(0.0, float(ordered[0] - ordered[1] - ordered[2] - ordered[3]))
