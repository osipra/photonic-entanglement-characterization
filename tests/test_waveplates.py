"""Behavioral tests for waveplate and analyzer-setting helpers."""

from __future__ import annotations

import numpy as np

from pec import states
from pec import waveplates


def _same_projector(left: np.ndarray, right: np.ndarray) -> bool:
    """Return whether two kets represent the same pure state up to global phase."""
    return bool(np.allclose(states.projector(left), states.projector(right)))


def test_quarter_wave_plate_matrix_matches_notebook_zero_angle() -> None:
    """A zero-angle QWP should match the notebook Jones matrix convention."""
    matrix = waveplates.quarter_wave_plate_matrix(0.0)

    assert np.allclose(matrix, np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128))


def test_half_wave_plate_matrix_matches_notebook_zero_angle() -> None:
    """A zero-angle HWP should match the notebook Jones matrix convention."""
    matrix = waveplates.half_wave_plate_matrix(0.0)

    assert np.allclose(matrix, np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128))


def test_measurement_basis_from_zero_angles_is_hv_basis() -> None:
    """A zero-angle analyzer should measure the computational H/V basis."""
    positive_state, negative_state = waveplates.measurement_basis_from_waveplates(0.0, 0.0)

    assert _same_projector(positive_state, states.ket_h())
    assert _same_projector(negative_state, states.ket_v())


def test_measurement_basis_from_known_angles_matches_standard_lab_states() -> None:
    """Simple QWP-HWP settings should reproduce standard D/A and R/L analyzers."""
    diagonal_positive, diagonal_negative = waveplates.measurement_basis_from_waveplates(45.0, 67.5)
    circular_positive, circular_negative = waveplates.measurement_basis_from_waveplates(0.0, 22.5)

    assert _same_projector(diagonal_positive, states.ket_d())
    assert _same_projector(diagonal_negative, states.ket_a())
    assert _same_projector(circular_positive, states.ket_r())
    assert _same_projector(circular_negative, states.ket_l())


def test_bloch_direction_from_state_matches_standard_polarization_states() -> None:
    """Pure-state Bloch vectors should match the expected Pauli-axis directions."""
    z_direction = waveplates.bloch_direction_from_state(states.ket_h())
    x_direction = waveplates.bloch_direction_from_state(states.ket_d())
    y_direction = waveplates.bloch_direction_from_state(states.ket_r())

    assert np.allclose(z_direction, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(x_direction, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(y_direction, np.array([0.0, 1.0, 0.0]))


def test_measurement_basis_for_bloch_direction_matches_pauli_axes() -> None:
    """Bloch-direction eigenkets should match the expected lab basis states."""
    x_positive, x_negative = waveplates.measurement_basis_for_bloch_direction(np.array([1.0, 0.0, 0.0]))
    z_positive, z_negative = waveplates.measurement_basis_for_bloch_direction(np.array([0.0, 0.0, 1.0]))

    assert _same_projector(x_positive, states.ket_d())
    assert _same_projector(x_negative, states.ket_a())
    assert _same_projector(z_positive, states.ket_h())
    assert _same_projector(z_negative, states.ket_v())


def test_waveplate_settings_helpers_recover_target_measurement_states() -> None:
    """Fitted waveplate settings should reproduce the intended positive-outcome state."""
    qwp_d, hwp_d, error_d = waveplates.waveplate_settings_for_state(states.ket_d(), seed=7)
    qwp_r, hwp_r, error_r = waveplates.waveplate_settings_for_bloch_direction(
        np.array([0.0, 1.0, 0.0]),
        seed=11,
    )

    measured_d, _ = waveplates.measurement_basis_from_waveplates(qwp_d, hwp_d)
    measured_r, _ = waveplates.measurement_basis_from_waveplates(qwp_r, hwp_r)

    assert _same_projector(measured_d, states.ket_d())
    assert _same_projector(measured_r, states.ket_r())
    assert error_d <= 1e-8
    assert error_r <= 1e-8


def test_waveplate_settings_for_label_matches_direct_state_fit() -> None:
    """The label convenience wrapper should agree with the direct state-based fitter."""
    label_qwp, label_hwp, label_error = waveplates.waveplate_settings_for_label("H", seed=3)
    state_qwp, state_hwp, state_error = waveplates.waveplate_settings_for_state(states.ket_h(), seed=3)

    label_positive, _ = waveplates.measurement_basis_from_waveplates(label_qwp, label_hwp)
    state_positive, _ = waveplates.measurement_basis_from_waveplates(state_qwp, state_hwp)

    assert _same_projector(label_positive, states.ket_h())
    assert _same_projector(state_positive, states.ket_h())
    assert label_error <= 1e-8
    assert state_error <= 1e-8
