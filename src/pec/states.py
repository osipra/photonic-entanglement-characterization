"""Linear-algebra and state helpers for photonic polarization experiments."""

from __future__ import annotations

from itertools import product
from typing import Final, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complex128]

BasisLabel = Literal["0", "1", "H", "V", "D", "A", "R", "L"]
BellStateLabel = Literal[
    "phi_plus",
    "phi_minus",
    "psi_plus",
    "psi_minus",
    "Phi+",
    "Phi-",
    "Psi+",
    "Psi-",
]
PauliLabel = Literal["I", "X", "Y", "Z"]

_SQRT2: Final[float] = float(np.sqrt(2.0))


def _readonly(array: ComplexArray) -> ComplexArray:
    """Mark an internal array constant as read-only."""
    array.setflags(write=False)
    return array


def _as_complex_vector(state_vector: ArrayLike) -> ComplexArray:
    """Convert a state vector input into a flat complex array."""
    return np.asarray(state_vector, dtype=np.complex128).reshape(-1)


def _as_complex_matrix(matrix: ArrayLike) -> ComplexArray:
    """Convert a matrix-like input into a complex array."""
    return np.asarray(matrix, dtype=np.complex128)


_PAULI_BY_LABEL: Final[dict[PauliLabel, ComplexArray]] = {
    "I": _readonly(np.eye(2, dtype=np.complex128)),
    "X": _readonly(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)),
    "Y": _readonly(np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)),
    "Z": _readonly(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)),
}

_KET_BY_LABEL: Final[dict[BasisLabel, ComplexArray]] = {
    "0": _readonly(np.array([1.0, 0.0], dtype=np.complex128)),
    "1": _readonly(np.array([0.0, 1.0], dtype=np.complex128)),
    "H": _readonly(np.array([1.0, 0.0], dtype=np.complex128)),
    "V": _readonly(np.array([0.0, 1.0], dtype=np.complex128)),
    "D": _readonly(np.array([1.0, 1.0], dtype=np.complex128) / _SQRT2),
    "A": _readonly(np.array([1.0, -1.0], dtype=np.complex128) / _SQRT2),
    "R": _readonly(np.array([1.0, 1.0j], dtype=np.complex128) / _SQRT2),
    "L": _readonly(np.array([1.0, -1.0j], dtype=np.complex128) / _SQRT2),
}

_BELL_ALIASES: Final[dict[str, str]] = {
    "phi_plus": "phi_plus",
    "phi_minus": "phi_minus",
    "psi_plus": "psi_plus",
    "psi_minus": "psi_minus",
    "Phi+": "phi_plus",
    "Phi-": "phi_minus",
    "Psi+": "psi_plus",
    "Psi-": "psi_minus",
}

__all__ = [
    "BasisLabel",
    "BellStateLabel",
    "PauliLabel",
    "basis_ket",
    "bell_state",
    "bell_states",
    "computational_basis",
    "density_matrix",
    "is_hermitian",
    "ket_a",
    "ket_d",
    "ket_h",
    "ket_l",
    "ket_one",
    "ket_r",
    "ket_v",
    "ket_zero",
    "make_hermitian",
    "pauli",
    "pauli_matrices",
    "proj",
    "projector",
    "projector_from_label",
    "tensor_ket",
    "tensor_product",
    "tensor_projector",
    "trace_normalize",
]


def pauli(label: PauliLabel) -> ComplexArray:
    """Return a single-qubit Pauli or identity matrix by label."""
    return _PAULI_BY_LABEL[label].copy()


def pauli_matrices() -> dict[PauliLabel, ComplexArray]:
    """Return the identity and Pauli matrices used in polarization analyses."""
    return {label: matrix.copy() for label, matrix in _PAULI_BY_LABEL.items()}


def ket_zero() -> ComplexArray:
    """Return the computational |0> ket."""
    return _KET_BY_LABEL["0"].copy()


def ket_one() -> ComplexArray:
    """Return the computational |1> ket."""
    return _KET_BY_LABEL["1"].copy()


def ket_h() -> ComplexArray:
    """Return the horizontal polarization basis ket."""
    return _KET_BY_LABEL["H"].copy()


def ket_v() -> ComplexArray:
    """Return the vertical polarization basis ket."""
    return _KET_BY_LABEL["V"].copy()


def ket_d() -> ComplexArray:
    """Return the diagonal polarization basis ket."""
    return _KET_BY_LABEL["D"].copy()


def ket_a() -> ComplexArray:
    """Return the anti-diagonal polarization basis ket."""
    return _KET_BY_LABEL["A"].copy()


def ket_r() -> ComplexArray:
    """Return the right-circular polarization basis ket."""
    return _KET_BY_LABEL["R"].copy()


def ket_l() -> ComplexArray:
    """Return the left-circular polarization basis ket."""
    return _KET_BY_LABEL["L"].copy()


def basis_ket(label: BasisLabel) -> ComplexArray:
    """Return a named single-qubit basis ket used in the notebooks."""
    try:
        return _KET_BY_LABEL[label].copy()
    except KeyError as exc:
        raise ValueError(f"Unsupported basis label: {label!r}") from exc


def tensor_product(*factors: ArrayLike) -> ComplexArray:
    """Return the Kronecker product of one or more ket-like factors."""
    if not factors:
        raise ValueError("tensor_product requires at least one factor.")

    result = _as_complex_vector(factors[0])
    for factor in factors[1:]:
        result = np.kron(result, _as_complex_vector(factor))
    return result


def tensor_ket(labels: str | Sequence[BasisLabel]) -> ComplexArray:
    """Build a tensor-product ket from a string or sequence of basis labels."""
    symbols = list(labels) if isinstance(labels, str) else list(labels)
    if not symbols:
        raise ValueError("tensor_ket requires at least one basis label.")

    return tensor_product(*(basis_ket(label) for label in symbols))


def computational_basis(num_qubits: int) -> dict[str, ComplexArray]:
    """Return the computational tensor-product basis for the given qubit count."""
    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1.")

    return {
        "".join(bits): tensor_ket(bits)
        for bits in product(("0", "1"), repeat=num_qubits)
    }


def bell_states() -> dict[str, ComplexArray]:
    """Return the canonical two-qubit Bell states."""
    ket00 = tensor_ket("00")
    ket01 = tensor_ket("01")
    ket10 = tensor_ket("10")
    ket11 = tensor_ket("11")

    return {
        "phi_plus": (ket00 + ket11) / _SQRT2,
        "phi_minus": (ket00 - ket11) / _SQRT2,
        "psi_plus": (ket01 + ket10) / _SQRT2,
        "psi_minus": (ket01 - ket10) / _SQRT2,
    }


def bell_state(label: BellStateLabel = "phi_plus") -> ComplexArray:
    """Return a named Bell state ket."""
    try:
        canonical_label = _BELL_ALIASES[label]
    except KeyError as exc:
        raise ValueError(f"Unsupported Bell-state label: {label!r}") from exc
    return bell_states()[canonical_label].copy()


def projector(state_vector: ArrayLike) -> ComplexArray:
    """Return the rank-1 projector |psi><psi| for a ket-like input."""
    ket = _as_complex_vector(state_vector).reshape(-1, 1)
    return ket @ ket.conj().T


def proj(state_vector: ArrayLike) -> ComplexArray:
    """Return a short-hand alias for :func:`projector`."""
    return projector(state_vector)


def projector_from_label(label: BasisLabel) -> ComplexArray:
    """Return a single-qubit projector for a named basis ket."""
    return projector(basis_ket(label))


def tensor_projector(labels: str | Sequence[BasisLabel]) -> ComplexArray:
    """Return a tensor-product projector for one or more basis labels."""
    return projector(tensor_ket(labels))


def density_matrix(state_vector: ArrayLike) -> ComplexArray:
    """Construct a pure-state density matrix from a ket-like input."""
    return projector(state_vector)


def make_hermitian(matrix: ArrayLike) -> ComplexArray:
    """Return the Hermitian part of a matrix."""
    matrix_array = _as_complex_matrix(matrix)
    return 0.5 * (matrix_array + matrix_array.conj().T)


def is_hermitian(matrix: ArrayLike, *, atol: float = 1e-9) -> bool:
    """Return whether a matrix is Hermitian within a numerical tolerance."""
    matrix_array = _as_complex_matrix(matrix)
    return bool(np.allclose(matrix_array, matrix_array.conj().T, atol=atol))


def trace_normalize(matrix: ArrayLike) -> ComplexArray:
    """Scale a matrix so that its trace is one."""
    matrix_array = _as_complex_matrix(matrix)
    trace = np.trace(matrix_array)
    if np.isclose(trace, 0.0):
        raise ValueError("Cannot trace-normalize a matrix with zero trace.")
    return matrix_array / trace
