"""Behavioral tests for reusable tomography helpers."""

from __future__ import annotations

import math

import numpy as np

from pec import states
from pec import tomography


def test_bloch_vector_from_axis_probabilities_matches_h_state() -> None:
    """Notebook Bloch reconstruction should map pH=1, pD=pR=0.5 to +z."""
    bloch = tomography.bloch_vector_from_axis_probabilities(1.0, 0.5, 0.5)

    assert np.allclose(bloch, np.array([0.0, 0.0, 1.0]))


def test_single_qubit_axis_probabilities_from_counts_supports_pairs_and_minmax() -> None:
    """Single-qubit count helper should support both direct pairs and min/max calibration."""
    paired = tomography.single_qubit_axis_probabilities_from_counts(
        {"H": 80.0, "V": 20.0, "D": 50.0, "A": 50.0, "R": 50.0, "L": 50.0}
    )
    minmax = tomography.single_qubit_axis_probabilities_from_counts(
        {"P_H": 80.0, "P_V": 20.0, "P_D": 50.0, "P_R": 50.0, "P_max": 100.0, "P_min": 0.0}
    )

    assert np.allclose([paired["pH"], paired["pD"], paired["pR"]], [0.8, 0.5, 0.5])
    assert np.allclose([minmax["pH"], minmax["pD"], minmax["pR"]], [0.8, 0.5, 0.5])


def test_single_qubit_density_matrix_from_probabilities_reconstructs_r_state() -> None:
    """H/D/R probabilities should reconstruct the expected single-qubit state."""
    rho_r = tomography.single_qubit_density_matrix_from_probabilities(0.5, 0.5, 1.0)

    assert np.allclose(rho_r, states.density_matrix(states.ket_r()))


def test_measurement_projectors_from_labels_support_single_and_two_qubit_labels() -> None:
    """Label assembly should reuse the state helpers for both 1Q and 2Q projectors."""
    projectors = tomography.measurement_projectors_from_labels(["H", "R", "HV"])

    assert np.allclose(projectors[0], states.projector_from_label("H"))
    assert np.allclose(projectors[1], states.projector_from_label("R"))
    assert np.allclose(projectors[2], states.tensor_projector("HV"))


def test_linear_inversion_tomography_recovers_h_state_from_probabilities() -> None:
    """Least-squares inversion should recover a simple one-qubit basis state."""
    rho_h = states.density_matrix(states.ket_h())
    measurement_operators = tomography.measurement_projectors_from_labels(["H", "V", "D", "A", "R", "L"])
    observations = np.array([1.0, 0.0, 0.5, 0.5, 0.5, 0.5])

    rho_fit = tomography.linear_inversion_tomography(observations, measurement_operators)

    assert np.allclose(rho_fit, rho_h)


def test_reconstruct_density_matrix_recovers_two_qubit_bell_state_from_labels() -> None:
    """High-level reconstruction should recover a two-qubit Bell state from labeled measurements."""
    rho_phi_plus = states.density_matrix(states.bell_state("phi_plus"))
    labels = [
        "HH",
        "HV",
        "HD",
        "HR",
        "VH",
        "VV",
        "VD",
        "VR",
        "DH",
        "DV",
        "DD",
        "DR",
        "RH",
        "RV",
        "RD",
        "RR",
    ]
    observations = {
        label: float(
            np.real(
                np.trace(
                    states.tensor_projector(label) @ rho_phi_plus,
                )
            )
        )
        for label in labels
    }

    rho_fit = tomography.reconstruct_density_matrix(observations, method="linear_inversion")

    assert np.allclose(rho_fit, rho_phi_plus)


def test_reconstruct_density_matrix_dispatches_single_qubit_and_mle_paths() -> None:
    """High-level reconstruction should support both count dictionaries and MLE dispatch."""
    rho_from_counts = tomography.reconstruct_density_matrix(
        {"H": 80.0, "V": 20.0, "D": 50.0, "A": 50.0, "R": 50.0, "L": 50.0},
        method="linear_inversion",
    )
    rho_from_mle = tomography.reconstruct_density_matrix(
        np.array([90.0, 10.0]),
        measurement_labels=["H", "V"],
        method="mle",
    )

    expected = 0.5 * (states.pauli("I") + 0.6 * states.pauli("Z"))
    assert np.allclose(rho_from_counts, expected)
    assert np.isclose(np.trace(rho_from_mle), 1.0)


def test_reconstruct_density_matrix_allows_mle_with_mapping_inputs() -> None:
    """MLE dispatch should not be intercepted by the notebook single-qubit shortcut."""
    rho_from_mle = tomography.reconstruct_density_matrix(
        {"H": 90.0, "V": 10.0},
        method="mle",
    )

    assert np.isclose(np.trace(rho_from_mle), 1.0)


def test_reconstruction_summary_collects_state_bell_and_chsh_quantities() -> None:
    """Summary helper should combine metrics, Bell overlaps, and optional CHSH data."""
    rho_phi_plus = states.density_matrix(states.bell_state("phi_plus"))
    z_axis = np.array([0.0, 0.0, 1.0])
    x_axis = np.array([1.0, 0.0, 0.0])
    b_axis = (z_axis + x_axis) / math.sqrt(2.0)
    bp_axis = (z_axis - x_axis) / math.sqrt(2.0)

    summary = tomography.reconstruction_summary(
        rho_phi_plus,
        target_state=states.bell_state("phi_plus"),
        chsh_settings=(z_axis, x_axis, b_axis, bp_axis),
    )

    assert np.isclose(summary["trace"], 1.0)
    assert np.isclose(summary["purity"], 1.0)
    assert np.isclose(summary["fidelity"], 1.0)
    assert summary["dominant_bell_state"] == "phi_plus"
    assert np.isclose(summary["chsh_s"], 2.0 * math.sqrt(2.0))
