"""Behavioral tests for optional QuTiP interoperability helpers."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pytest

pytest.importorskip("qutip")

from pec import bell
from pec import metrics
from pec import states

qutip_tools = import_module("pec.qutip_tools")


def test_to_qobj_and_from_qobj_round_trip_kets_and_density_matrices() -> None:
    """NumPy state objects should round-trip through QuTiP cleanly."""
    ket_phi_plus = states.bell_state("phi_plus")
    rho_phi_plus = states.density_matrix(ket_phi_plus)

    ket_qobj = qutip_tools.to_qobj(ket_phi_plus, dims=[2, 2])
    rho_qobj = qutip_tools.to_qobj(rho_phi_plus, dims=[2, 2])

    assert ket_qobj.isket
    assert rho_qobj.isoper
    assert ket_qobj.dims[0] == [2, 2]
    assert int(np.prod(ket_qobj.dims[1])) == 1
    assert rho_qobj.dims == [[2, 2], [2, 2]]
    assert np.allclose(qutip_tools.from_qobj(ket_qobj), ket_phi_plus)
    assert np.allclose(qutip_tools.from_qobj(rho_qobj), rho_phi_plus)


def test_qutip_metric_wrappers_match_pec_metric_conventions() -> None:
    """QuTiP validation helpers should agree with the PEC metric implementations."""
    rho_phi_plus = states.density_matrix(states.bell_state("phi_plus"))
    rho_mixed = states.trace_normalize(0.85 * rho_phi_plus + 0.15 * states.tensor_projector("HV"))

    assert np.isclose(qutip_tools.qutip_purity(rho_mixed), metrics.purity(rho_mixed))
    assert np.isclose(qutip_tools.qutip_fidelity(rho_mixed, rho_phi_plus), metrics.fidelity(rho_mixed, rho_phi_plus))
    assert np.isclose(
        qutip_tools.qutip_trace_distance(rho_mixed, rho_phi_plus),
        metrics.trace_distance(rho_mixed, rho_phi_plus),
    )


def test_qutip_expect_matches_known_bell_correlator() -> None:
    """Expectation helpers should reproduce simple Bell-state observables."""
    rho_phi_plus = states.density_matrix(states.bell_state("phi_plus"))
    zz_operator = np.kron(states.pauli("Z"), states.pauli("Z"))

    expectation = qutip_tools.qutip_expect(zz_operator, rho_phi_plus)

    assert np.isclose(expectation, 1.0)
    assert np.isclose(expectation, bell.two_qubit_correlation(rho_phi_plus, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]))


def test_compare_metric_values_reports_readable_agreement() -> None:
    """Comparison helper should expose PEC and QuTiP values side by side."""
    rho_phi_plus = states.density_matrix(states.bell_state("phi_plus"))
    rho_mixed = states.trace_normalize(0.9 * rho_phi_plus + 0.1 * states.tensor_projector("HV"))

    comparison = qutip_tools.compare_metric_values(rho_mixed, rho_phi_plus)

    assert set(comparison) == {"purity", "fidelity", "trace_distance"}
    assert comparison["purity"]["within_tolerance"]
    assert comparison["fidelity"]["within_tolerance"]
    assert comparison["trace_distance"]["within_tolerance"]
    assert comparison["fidelity"]["abs_diff"] < 1e-8
