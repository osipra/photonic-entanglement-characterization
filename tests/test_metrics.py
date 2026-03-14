"""Behavioral tests for reusable state metrics."""

from __future__ import annotations

import numpy as np

from pec import metrics
from pec import states


def test_purity_matches_pure_and_mixed_state_limits() -> None:
    """Purity should distinguish pure states from maximally mixed states."""
    rho_pure = states.density_matrix(states.ket_h())
    rho_mixed = 0.5 * np.eye(2, dtype=np.complex128)

    assert np.isclose(metrics.purity(rho_pure), 1.0)
    assert np.isclose(metrics.purity(rho_mixed), 0.5)


def test_pure_state_fidelity_matches_notebook_overlap_formula() -> None:
    """Pure-state fidelity should reproduce <psi|rho|psi>."""
    rho_d = states.density_matrix(states.ket_d())

    assert np.isclose(metrics.fidelity_pure(rho_d, states.ket_d()), 1.0)
    assert np.isclose(metrics.fidelity_pure(rho_d, states.ket_h()), 0.5)


def test_density_matrix_fidelity_handles_pure_and_mixed_cases() -> None:
    """General fidelity should agree with standard qubit benchmarks."""
    rho_h = states.density_matrix(states.ket_h())
    rho_v = states.density_matrix(states.ket_v())
    rho_mixed = 0.5 * np.eye(2, dtype=np.complex128)

    assert np.isclose(metrics.fidelity(rho_h, rho_h), 1.0)
    assert np.isclose(metrics.fidelity(rho_h, rho_v), 0.0)
    assert np.isclose(metrics.fidelity(rho_h, rho_mixed), 0.5)


def test_bell_state_fidelities_identify_matching_bell_state() -> None:
    """Bell-state fidelities should peak on the matching Bell projector."""
    rho_phi_plus = states.density_matrix(states.bell_state("phi_plus"))
    fidelities = metrics.bell_state_fidelities(rho_phi_plus)

    assert set(fidelities) == {"phi_plus", "phi_minus", "psi_plus", "psi_minus"}
    assert np.isclose(fidelities["phi_plus"], 1.0)
    assert np.isclose(fidelities["phi_minus"], 0.0)
    assert np.isclose(fidelities["psi_plus"], 0.0)
    assert np.isclose(fidelities["psi_minus"], 0.0)


def test_state_eigenvalues_return_real_spectrum_of_hermitian_part() -> None:
    """State eigenvalues should be computed from the Hermitianized matrix."""
    rho = np.array([[0.75, 0.1 + 0.2j], [0.1 - 0.2j, 0.25]], dtype=np.complex128)
    eigenvalues = metrics.state_eigenvalues(rho)

    assert np.allclose(eigenvalues, np.linalg.eigvalsh(states.make_hermitian(rho)))
    assert np.isclose(np.sum(eigenvalues), 1.0)


def test_trace_distance_and_linear_entropy_cover_basic_state_mixedness() -> None:
    """Simple derived state metrics should behave as expected."""
    rho_h = states.density_matrix(states.ket_h())
    rho_v = states.density_matrix(states.ket_v())
    rho_mixed = 0.5 * np.eye(2, dtype=np.complex128)

    assert np.isclose(metrics.trace_distance(rho_h, rho_h), 0.0)
    assert np.isclose(metrics.trace_distance(rho_h, rho_v), 1.0)
    assert np.isclose(metrics.linear_entropy(rho_h), 0.0)
    assert np.isclose(metrics.linear_entropy(rho_mixed), 0.5)
