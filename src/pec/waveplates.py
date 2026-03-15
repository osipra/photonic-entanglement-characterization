"""Waveplate and analyzer-setting helpers for polarization measurements."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from . import bell
from . import states

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
WaveplateSettings = tuple[float, float, float]

__all__ = [
    "analyzer_unitary_from_waveplates",
    "bloch_direction_from_state",
    "half_wave_plate_matrix",
    "measurement_basis_for_bloch_direction",
    "measurement_basis_from_waveplates",
    "quarter_wave_plate_matrix",
    "waveplate_settings_for_bloch_direction",
    "waveplate_settings_for_label",
    "waveplate_settings_for_state",
]


def _rotation_matrix(angle_radians: float) -> ComplexArray:
    """Return the real polarization-basis rotation matrix for an optic angle."""
    cosine = np.cos(angle_radians)
    sine = np.sin(angle_radians)
    return np.array(
        [
            [cosine, -sine],
            [sine, cosine],
        ],
        dtype=np.complex128,
    )


def _retarder_matrix(angle_degrees: float, phase_delay: complex) -> ComplexArray:
    """Return the Jones matrix for a rotated single-qubit phase retarder."""
    angle_radians = np.deg2rad(float(angle_degrees))
    rotation = _rotation_matrix(angle_radians)
    retarder = np.array(
        [
            [1.0, 0.0],
            [0.0, phase_delay],
        ],
        dtype=np.complex128,
    )
    return rotation.conj().T @ retarder @ rotation


def _normalize_state(state_vector: ArrayLike) -> ComplexArray:
    """Convert a ket-like input into a normalized single-qubit state vector."""
    ket = np.asarray(state_vector, dtype=np.complex128).reshape(-1)
    if ket.shape != (2,):
        raise ValueError("Waveplate helpers require a single-qubit state vector.")

    norm = np.linalg.norm(ket)
    if np.isclose(norm, 0.0):
        raise ValueError("State vectors must be nonzero.")
    return ket / norm


def _canonicalize_global_phase(state_vector: ArrayLike) -> ComplexArray:
    """Fix the global phase so the first nonzero component is real and positive."""
    ket = _normalize_state(state_vector)
    for amplitude in ket:
        if not np.isclose(abs(amplitude), 0.0):
            return ket * np.exp(-1j * np.angle(amplitude))
    return ket


def _objective_for_target_state(target_state: ComplexArray):
    """Return a scalar overlap loss for fitting waveplate analyzer settings."""

    def objective(angles: Sequence[float]) -> float:
        qwp_angle, hwp_angle = float(angles[0]), float(angles[1])
        positive_state, _ = measurement_basis_from_waveplates(qwp_angle, hwp_angle)
        overlap = np.vdot(positive_state, target_state)
        return float(1.0 - abs(overlap) ** 2)

    return objective


def _fit_waveplate_settings(
    target_state: ComplexArray,
    *,
    n_starts: int,
    seed: int,
    max_iterations: int,
) -> WaveplateSettings:
    """Fit QWP-HWP analyzer angles for a target positive-outcome state."""
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    objective = _objective_for_target_state(target_state)
    bounds = [(0.0, 180.0), (0.0, 180.0)]
    rng = np.random.default_rng(seed)

    start_points = [
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([0.0, 22.5], dtype=np.float64),
        np.array([0.0, 67.5], dtype=np.float64),
        np.array([45.0, 0.0], dtype=np.float64),
        np.array([135.0, 0.0], dtype=np.float64),
        np.array([45.0, 45.0], dtype=np.float64),
        np.array([135.0, 45.0], dtype=np.float64),
    ]
    random_starts = max(0, n_starts - len(start_points))
    for _ in range(random_starts):
        start_points.append(rng.uniform(0.0, 180.0, size=2))

    best_result = None
    for start in start_points:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iterations},
        )
        if best_result is None or float(result.fun) < float(best_result.fun):
            best_result = result

    if best_result is None:
        raise RuntimeError("Waveplate-angle optimization did not produce a result.")
    qwp_angle, hwp_angle = best_result.x
    return (
        float(qwp_angle % 180.0),
        float(hwp_angle % 180.0),
        float(best_result.fun),
    )


def quarter_wave_plate_matrix(angle_degrees: float) -> ComplexArray:
    """Return the Jones matrix for a quarter-wave plate at the given angle."""
    return _retarder_matrix(angle_degrees, 1j)


def half_wave_plate_matrix(angle_degrees: float) -> ComplexArray:
    """Return the Jones matrix for a half-wave plate at the given angle."""
    return _retarder_matrix(angle_degrees, -1.0)


def analyzer_unitary_from_waveplates(
    qwp_angle_degrees: float,
    hwp_angle_degrees: float,
) -> ComplexArray:
    """Return the notebook QWP-then-HWP analyzer unitary in the H/V basis."""
    return (
        half_wave_plate_matrix(hwp_angle_degrees)
        @ quarter_wave_plate_matrix(qwp_angle_degrees)
    )


def measurement_basis_from_waveplates(
    qwp_angle_degrees: float,
    hwp_angle_degrees: float,
) -> tuple[ComplexArray, ComplexArray]:
    """Return the input-state basis measured as H/V by a QWP-HWP-PBS analyzer."""
    analyzer = analyzer_unitary_from_waveplates(
        qwp_angle_degrees,
        hwp_angle_degrees,
    )
    positive_state = analyzer.conj().T @ states.ket_h()
    negative_state = analyzer.conj().T @ states.ket_v()
    return (
        _canonicalize_global_phase(positive_state),
        _canonicalize_global_phase(negative_state),
    )


def bloch_direction_from_state(state_vector: ArrayLike) -> RealArray:
    """Return the Bloch-sphere direction for a pure single-qubit polarization state."""
    ket = _normalize_state(state_vector)
    rho = states.projector(ket)
    direction = np.array(
        [
            np.real(np.trace(rho @ states.pauli("X"))),
            np.real(np.trace(rho @ states.pauli("Y"))),
            np.real(np.trace(rho @ states.pauli("Z"))),
        ],
        dtype=np.float64,
    )
    norm = np.linalg.norm(direction)
    if np.isclose(norm, 0.0):
        raise ValueError("Pure-state Bloch directions must be nonzero.")
    return direction / norm


def measurement_basis_for_bloch_direction(
    direction: ArrayLike,
) -> tuple[ComplexArray, ComplexArray]:
    """Return the +/- eigenkets of n.sigma for a Bloch-sphere analyzer direction."""
    observable = bell.axis_observable(direction)
    eigenvalues, eigenvectors = np.linalg.eigh(observable)
    positive_state = eigenvectors[:, int(np.argmax(eigenvalues))]
    negative_state = eigenvectors[:, int(np.argmin(eigenvalues))]
    return (
        _canonicalize_global_phase(positive_state),
        _canonicalize_global_phase(negative_state),
    )


def waveplate_settings_for_state(
    state_vector: ArrayLike,
    *,
    n_starts: int = 64,
    seed: int = 0,
    max_iterations: int = 2_500,
) -> WaveplateSettings:
    """Fit QWP-HWP analyzer settings that make the positive outcome measure a target state."""
    target_state = _canonicalize_global_phase(state_vector)
    return _fit_waveplate_settings(
        target_state,
        n_starts=n_starts,
        seed=seed,
        max_iterations=max_iterations,
    )


def waveplate_settings_for_bloch_direction(
    direction: ArrayLike,
    *,
    n_starts: int = 64,
    seed: int = 0,
    max_iterations: int = 2_500,
) -> WaveplateSettings:
    """Fit QWP-HWP analyzer settings for the +1 eigenstate of a Bloch direction."""
    positive_state, _ = measurement_basis_for_bloch_direction(direction)
    return waveplate_settings_for_state(
        positive_state,
        n_starts=n_starts,
        seed=seed,
        max_iterations=max_iterations,
    )


def waveplate_settings_for_label(
    label: states.BasisLabel,
    *,
    n_starts: int = 64,
    seed: int = 0,
    max_iterations: int = 2_500,
) -> WaveplateSettings:
    """Fit QWP-HWP analyzer settings for a standard lab basis label like H, D, or R."""
    return waveplate_settings_for_state(
        states.basis_ket(label),
        n_starts=n_starts,
        seed=seed,
        max_iterations=max_iterations,
    )
