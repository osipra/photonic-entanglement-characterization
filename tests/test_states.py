"""Behavioral tests for reusable state and linear-algebra helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pec import states


def test_pauli_matrices_match_standard_definitions() -> None:
    """The Pauli helpers should reproduce the usual single-qubit matrices."""
    paulis = states.pauli_matrices()

    assert set(paulis) == {"I", "X", "Y", "Z"}
    assert np.allclose(paulis["I"], np.eye(2, dtype=np.complex128))
    assert np.allclose(paulis["X"], np.array([[0, 1], [1, 0]], dtype=np.complex128))
    assert np.allclose(paulis["Y"], np.array([[0, -1j], [1j, 0]], dtype=np.complex128))
    assert np.allclose(paulis["Z"], np.array([[1, 0], [0, -1]], dtype=np.complex128))


def test_single_qubit_basis_kets_follow_lab_conventions() -> None:
    """Common polarization basis states should match the notebook definitions."""
    assert np.allclose(states.ket_zero(), np.array([1, 0], dtype=np.complex128))
    assert np.allclose(states.ket_one(), np.array([0, 1], dtype=np.complex128))
    assert np.allclose(states.ket_h(), states.ket_zero())
    assert np.allclose(states.ket_v(), states.ket_one())
    assert np.allclose(states.ket_d(), np.array([1, 1], dtype=np.complex128) / np.sqrt(2.0))
    assert np.allclose(states.ket_a(), np.array([1, -1], dtype=np.complex128) / np.sqrt(2.0))
    assert np.allclose(states.ket_r(), np.array([1, 1j], dtype=np.complex128) / np.sqrt(2.0))
    assert np.allclose(states.ket_l(), np.array([1, -1j], dtype=np.complex128) / np.sqrt(2.0))


def test_tensor_kets_and_computational_basis_are_consistent() -> None:
    """Tensor helpers should construct multi-qubit basis states cleanly."""
    assert np.allclose(
        states.tensor_product(states.ket_h(), states.ket_v()),
        np.array([0, 1, 0, 0], dtype=np.complex128),
    )
    assert np.allclose(states.tensor_ket("HV"), np.array([0, 1, 0, 0], dtype=np.complex128))

    basis = states.computational_basis(2)

    assert set(basis) == {"00", "01", "10", "11"}
    assert np.allclose(basis["00"], np.array([1, 0, 0, 0], dtype=np.complex128))
    assert np.allclose(basis["11"], np.array([0, 0, 0, 1], dtype=np.complex128))


def test_bell_states_match_standard_two_qubit_definitions() -> None:
    """Bell state helpers should reproduce the canonical entangled kets."""
    phi_plus = states.bell_state("phi_plus")
    psi_minus = states.bell_state("Psi-")

    assert np.allclose(
        phi_plus,
        np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2.0),
    )
    assert np.allclose(
        psi_minus,
        np.array([0, 1, -1, 0], dtype=np.complex128) / np.sqrt(2.0),
    )
    assert np.isclose(np.vdot(phi_plus, phi_plus), 1.0)
    assert np.isclose(np.vdot(phi_plus, psi_minus), 0.0)


def test_projectors_and_density_matrices_are_rank_one_for_kets() -> None:
    """Projector helpers should build pure-state density matrices."""
    ket_d = states.ket_d()
    projector = states.projector(ket_d)

    assert np.allclose(projector, states.proj(ket_d))
    assert np.allclose(projector, states.density_matrix(ket_d))
    assert np.allclose(projector, states.projector_from_label("D"))
    assert np.allclose(np.trace(projector), 1.0)
    assert states.is_hermitian(projector)


def test_tensor_projector_uses_tensor_basis_labels() -> None:
    """Tensor projectors should align with tensor-product basis kets."""
    expected = states.projector(states.tensor_ket("HV"))

    assert np.allclose(states.tensor_projector("HV"), expected)


def test_make_hermitian_and_trace_normalize_clean_up_density_matrices() -> None:
    """Hermitian and trace normalization helpers should be notebook-compatible."""
    raw = np.array([[1.0, 2.0 + 1.0j], [3.0 - 1.0j, 4.0]], dtype=np.complex128)

    hermitian = states.make_hermitian(raw)
    normalized = states.trace_normalize(hermitian)

    assert np.allclose(hermitian, np.array([[1.0, 2.5 + 1.0j], [2.5 - 1.0j, 4.0]], dtype=np.complex128))
    assert states.is_hermitian(hermitian)
    assert np.isclose(np.trace(normalized), 1.0)


def test_trace_normalize_rejects_zero_trace() -> None:
    """Trace normalization should fail loudly for zero-trace inputs."""
    zero_trace = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="zero trace"):
        states.trace_normalize(zero_trace)


def test_state_helper_validation_errors_are_informative() -> None:
    """Basic state helpers should reject unsupported labels and empty inputs."""
    with pytest.raises(ValueError, match="basis label"):
        states.basis_ket("Q")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Bell-state label"):
        states.bell_state("Omega")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="at least one factor"):
        states.tensor_product()

    with pytest.raises(ValueError, match="at least 1"):
        states.computational_basis(0)
