"""Behavioral tests for reusable Bell-analysis helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pec import bell
from pec import states


def test_unit_vector_matches_standard_cartesian_directions() -> None:
    """Bloch-sphere helper should reproduce the x and z axes."""
    assert np.allclose(bell.unit_vector(np.pi / 2.0, 0.0), np.array([1.0, 0.0, 0.0]))
    assert np.allclose(bell.unit_vector(0.0, 0.0), np.array([0.0, 0.0, 1.0]))


def test_axis_observable_recovers_pauli_operators() -> None:
    """Axis observables along x, y, and z should match the Pauli matrices."""
    assert np.allclose(bell.axis_observable([1.0, 0.0, 0.0]), states.pauli("X"))
    assert np.allclose(bell.axis_observable([0.0, 1.0, 0.0]), states.pauli("Y"))
    assert np.allclose(bell.axis_observable([0.0, 0.0, 1.0]), states.pauli("Z"))


def test_two_qubit_correlation_reproduces_phi_plus_pauli_correlators() -> None:
    """Bell correlator helper should match the standard Phi+ expectations."""
    rho_phi_plus = states.density_matrix(states.bell_state("phi_plus"))

    exx = bell.two_qubit_correlation(rho_phi_plus, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    eyy = bell.two_qubit_correlation(rho_phi_plus, [0.0, 1.0, 0.0], [0.0, 1.0, 0.0])
    ezz = bell.two_qubit_correlation(rho_phi_plus, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])

    assert np.isclose(exx, 1.0)
    assert np.isclose(eyy, -1.0)
    assert np.isclose(ezz, 1.0)


def test_pauli_axis_correlations_return_notebook_style_keys() -> None:
    """Convenience correlator helper should expose Exx, Eyy, and Ezz."""
    rho_phi_plus = states.density_matrix(states.bell_state("phi_plus"))
    correlators = bell.pauli_axis_correlations(rho_phi_plus)

    assert set(correlators) == {"Exx", "Eyy", "Ezz"}
    assert np.isclose(correlators["Exx"], 1.0)
    assert np.isclose(correlators["Eyy"], -1.0)
    assert np.isclose(correlators["Ezz"], 1.0)


def test_bell_state_projectors_and_fidelities_match_bell_basis_targets() -> None:
    """Bell projectors and Bell fidelities should align on canonical Bell states."""
    rho_phi_plus = states.density_matrix(states.bell_state("phi_plus"))
    projectors = bell.bell_state_projectors()
    fidelities = bell.bell_state_fidelities(rho_phi_plus)

    assert np.allclose(projectors["phi_plus"], rho_phi_plus)
    assert np.isclose(fidelities["phi_plus"], 1.0)
    assert bell.dominant_bell_state(rho_phi_plus) == "phi_plus"


def test_bell_helpers_validate_directions_and_two_qubit_shape() -> None:
    """Bell helpers should reject malformed directions and non-two-qubit states."""
    with pytest.raises(ValueError, match="length-3"):
        bell.axis_observable([1.0, 0.0])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="4x4"):
        bell.two_qubit_correlation(np.eye(2, dtype=np.complex128), [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
