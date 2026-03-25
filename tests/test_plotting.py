"""Behavioral tests for reusable matplotlib-based plotting helpers."""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib.figure import Figure

from pec import plotting
from pec import states


def test_plot_density_matrix_creates_real_and_imag_heatmaps() -> None:
    """Density-matrix plotting should create one heatmap per requested component."""
    rho = states.density_matrix(states.bell_state("phi_plus"))

    figure = plotting.plot_density_matrix(
        rho,
        title="Bell State",
        annotate=False,
        colorbar=False,
    )

    heatmap_axes = [axis for axis in figure.axes if axis.images]
    assert isinstance(figure, Figure)
    assert len(heatmap_axes) == 2
    assert heatmap_axes[0].images[0].get_array().shape == (4, 4)
    assert heatmap_axes[0].get_title() == "Real"
    assert heatmap_axes[1].get_title() == "Imag"
    assert figure._suptitle is not None
    assert figure._suptitle.get_text() == "Bell State"


def test_plot_coincidence_counts_uses_mapping_order_and_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coincidence-count plots should not depend on matplotlib.figure being pre-imported."""
    counts = {"HH": 34749.0, "HV": 324.0, "VV": 35805.0}
    monkeypatch.delattr(matplotlib, "figure", raising=False)

    figure = plotting.plot_coincidence_counts(counts, title="Counts")
    axis = figure.axes[0]

    assert isinstance(figure, Figure)
    assert [patch.get_height() for patch in axis.patches] == [34749.0, 324.0, 35805.0]
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["HH", "HV", "VV"]
    assert axis.get_title() == "Counts"


def test_plot_coincidence_counts_uses_provided_axes() -> None:
    """Coincidence-count plots should draw into a supplied axes and return its figure."""
    counts = [10.0, 20.0]
    figure = Figure(figsize=(5.0, 4.0))
    axis = figure.subplots()

    returned_figure = plotting.plot_coincidence_counts(counts, labels=["HH", "VV"], ax=axis, title="Counts")

    assert returned_figure is figure
    assert [patch.get_height() for patch in axis.patches] == [10.0, 20.0]
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["HH", "VV"]
    assert axis.get_title() == "Counts"


def test_plot_bell_state_fidelities_uses_canonical_phi_psi_order() -> None:
    """Bell fidelities should display in the standard Phi/Psi ordering."""
    fidelities = {
        "psi_minus": 0.01,
        "phi_plus": 0.96,
        "psi_plus": 0.02,
        "phi_minus": 0.03,
    }

    figure = plotting.plot_bell_state_fidelities(fidelities, title="Bell Fidelities")
    axis = figure.axes[0]

    assert isinstance(figure, Figure)
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["Phi+", "Phi-", "Psi+", "Psi-"]
    assert np.allclose([patch.get_height() for patch in axis.patches], [0.96, 0.03, 0.02, 0.01])
    assert axis.get_title() == "Bell Fidelities"


def test_plot_chsh_correlators_adds_s_value_annotation() -> None:
    """CHSH correlator plots should annotate the derived S value when all four terms exist."""
    correlators = {
        "ab": 1.0,
        "abp": 1.0,
        "apb": 1.0,
        "apbp": -1.0,
    }

    figure = plotting.plot_chsh_correlators(correlators, title="CHSH")
    axis = figure.axes[0]

    assert isinstance(figure, Figure)
    assert len(axis.patches) == 4
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["E(a,b)", "E(a,b')", "E(a',b)", "E(a',b')"]
    assert any("S = 4.000" in text.get_text() for text in axis.texts)
    assert axis.get_title() == "CHSH"
