"""Behavioral tests for reusable MLE helpers."""

from __future__ import annotations

import numpy as np

from pec import mle
from pec import states


def test_lower_triangular_parameterization_returns_valid_density_matrix() -> None:
    """Notebook-style lower-triangular parameters should produce a physical rho."""
    rho = mle.density_matrix_from_lower_triangular_params([1.0, 0.0, 0.0, 0.0], dimension=2)

    assert rho.shape == (2, 2)
    assert np.isclose(np.trace(rho), 1.0)
    assert np.all(np.linalg.eigvalsh(states.make_hermitian(rho)) >= -1e-12)
    assert np.allclose(rho, states.density_matrix(states.ket_h()))


def test_dense_parameterization_returns_valid_density_matrix() -> None:
    """Dense PSD-factor parameters should also normalize to a physical rho."""
    params = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    rho = mle.density_matrix_from_dense_params(params, dimension=2)

    assert rho.shape == (2, 2)
    assert np.isclose(np.trace(rho), 1.0)
    assert np.all(np.linalg.eigvalsh(states.make_hermitian(rho)) >= -1e-12)
    assert np.allclose(rho, states.density_matrix(states.ket_h()))


def test_measurement_probabilities_match_simple_projective_measurements() -> None:
    """Born probabilities should agree with obvious single-qubit projectors."""
    rho_h = states.density_matrix(states.ket_h())
    measurement_operators = [
        states.projector_from_label("H"),
        states.projector_from_label("V"),
        states.projector_from_label("D"),
    ]

    probabilities = mle.measurement_probabilities(rho_h, measurement_operators)

    assert np.allclose(probabilities, np.array([1.0, 0.0, 0.5]))


def test_poisson_objectives_prefer_the_matching_state() -> None:
    """Likelihood helpers should favor the state that matches the observed counts."""
    counts = np.array([100.0, 0.0])
    measurement_operators = [
        states.projector_from_label("H"),
        states.projector_from_label("V"),
    ]
    rho_h = states.density_matrix(states.ket_h())
    rho_v = states.density_matrix(states.ket_v())

    nll_h = mle.poisson_negative_log_likelihood(counts, measurement_operators, rho_h, total_counts=100.0)
    nll_v = mle.poisson_negative_log_likelihood(counts, measurement_operators, rho_v, total_counts=100.0)
    chi2_h = mle.poisson_chi2_loss(counts, measurement_operators, rho_h, total_counts=100.0)
    chi2_v = mle.poisson_chi2_loss(counts, measurement_operators, rho_v, total_counts=100.0)

    assert nll_h < nll_v
    assert chi2_h < chi2_v


def test_fit_density_matrix_mle_returns_a_valid_density_matrix() -> None:
    """The generic fitter should return a normalized PSD estimate on simple data."""
    counts = np.array([90.0, 10.0])
    measurement_operators = [
        states.projector_from_label("H"),
        states.projector_from_label("V"),
    ]

    rho_fit = mle.fit_density_matrix_mle(
        counts,
        measurement_operators,
        parameterization="lower_triangular",
        objective="poisson_nll",
        max_iterations=200,
        tolerance=1e-6,
    )

    assert rho_fit.shape == (2, 2)
    assert np.isclose(np.trace(rho_fit), 1.0)
    assert np.all(np.linalg.eigvalsh(states.make_hermitian(rho_fit)) >= -1e-8)


def test_mle_diagnostics_report_probabilities_and_objectives() -> None:
    """Diagnostics should expose the fitted probabilities and basic loss values."""
    counts = np.array([80.0, 20.0])
    measurement_operators = [
        states.projector_from_label("H"),
        states.projector_from_label("V"),
    ]
    rho_h = states.density_matrix(states.ket_h())

    diagnostics = mle.mle_diagnostics(counts, measurement_operators, rho_h)

    assert np.allclose(diagnostics["probabilities"], np.array([1.0, 1e-12]), atol=1e-12)
    assert np.isclose(diagnostics["trace"], 1.0)
    assert np.isfinite(diagnostics["poisson_nll"])
    assert diagnostics["poisson_chi2"] >= 0.0
