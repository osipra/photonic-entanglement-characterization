"""Waveplate modeling helpers for polarization-state preparation and analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "half_wave_plate_matrix",
    "measurement_basis_from_waveplates",
    "quarter_wave_plate_matrix",
]


def quarter_wave_plate_matrix(angle_degrees: float) -> np.ndarray:
    """Return the Jones matrix for a quarter-wave plate at the given angle."""
    raise NotImplementedError


def half_wave_plate_matrix(angle_degrees: float) -> np.ndarray:
    """Return the Jones matrix for a half-wave plate at the given angle."""
    raise NotImplementedError


def measurement_basis_from_waveplates(
    qwp_angle_degrees: float,
    hwp_angle_degrees: float,
) -> np.ndarray:
    """Return the measurement basis implied by a QWP-HWP analyzer setting."""
    raise NotImplementedError
