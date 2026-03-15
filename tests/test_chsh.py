"""Behavioral tests for reusable CHSH-analysis helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pec import chsh
from pec import states


def test_correlator_from_counts_matches_standard_chsh_formula() -> None:
    """A single correlator should use the usual same-minus-different convention."""
    counts = {"++": 40.0, "+-": 10.0, "-+": 5.0, "--": 45.0}

    assert np.isclose(chsh.correlator_from_counts(counts), 0.7)
    assert np.isclose(chsh.correlator_from_counts({"pp": 40.0, "pm": 10.0, "mp": 5.0, "mm": 45.0}), 0.7)


def test_correlators_from_counts_builds_the_four_canonical_chsh_terms() -> None:
    """Setting-indexed counts should become the ab, abp, apb, and apbp correlators."""
    counts = {
        "ab": {"++": 35.0, "+-": 15.0, "-+": 15.0, "--": 35.0},
        "abp": {"++": 30.0, "+-": 20.0, "-+": 10.0, "--": 40.0},
        "apb": {"++": 25.0, "+-": 20.0, "-+": 15.0, "--": 40.0},
        "apbp": {"++": 15.0, "+-": 35.0, "-+": 25.0, "--": 25.0},
    }

    correlators = chsh.correlators_from_counts(counts)

    assert np.isclose(correlators["ab"], 0.4)
    assert np.isclose(correlators["abp"], 0.4)
    assert np.isclose(correlators["apb"], 0.3)
    assert np.isclose(correlators["apbp"], -0.2)


def test_chsh_s_value_uses_notebook_sign_convention() -> None:
    """S should follow E(a,b) + E(a,b') + E(a',b) - E(a',b')."""
    correlators = {"ab": 0.7, "abp": 0.6, "apb": 0.5, "apbp": -0.4}

    assert np.isclose(chsh.chsh_s_value(correlators), 2.2)


def test_chsh_s_from_rho_reaches_tsirelson_bound_for_phi_plus() -> None:
    """Standard x-z plane CHSH settings should give 2*sqrt(2) for Phi+."""
    rho_phi_plus = states.density_matrix(states.bell_state("phi_plus"))
    z_axis = np.array([0.0, 0.0, 1.0])
    x_axis = np.array([1.0, 0.0, 0.0])
    b_axis = (z_axis + x_axis) / math.sqrt(2.0)
    bp_axis = (z_axis - x_axis) / math.sqrt(2.0)

    correlators = chsh.correlators_from_rho(rho_phi_plus, z_axis, x_axis, b_axis, bp_axis)
    s_value = chsh.chsh_s_from_rho(rho_phi_plus, z_axis, x_axis, b_axis, bp_axis)

    assert np.isclose(correlators["ab"], 1.0 / math.sqrt(2.0))
    assert np.isclose(correlators["abp"], 1.0 / math.sqrt(2.0))
    assert np.isclose(correlators["apb"], 1.0 / math.sqrt(2.0))
    assert np.isclose(correlators["apbp"], -1.0 / math.sqrt(2.0))
    assert np.isclose(s_value, 2.0 * math.sqrt(2.0))


def test_violation_margin_uses_absolute_s_value() -> None:
    """Violation margin should measure excess over the classical CHSH bound."""
    correlators = {"ab": -0.7, "abp": -0.6, "apb": -0.5, "apbp": 0.4}

    assert np.isclose(chsh.violation_margin(correlators), 0.2)


def test_chsh_helpers_validate_missing_settings_and_zero_counts() -> None:
    """CHSH utilities should fail clearly on incomplete or degenerate inputs."""
    with pytest.raises(ValueError, match="Missing CHSH settings"):
        chsh.correlators_from_counts({"ab": {"++": 1.0}})

    with pytest.raises(ValueError, match="nonzero total count"):
        chsh.correlator_from_counts({"++": 0.0, "+-": 0.0, "-+": 0.0, "--": 0.0})
