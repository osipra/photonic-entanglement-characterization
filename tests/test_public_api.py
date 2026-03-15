"""Checks for the intended public API surface of the scaffold."""

from __future__ import annotations

import inspect
from importlib import import_module

import pec

EXPECTED_MODULES = [
    "bell",
    "chsh",
    "io",
    "metrics",
    "mle",
    "plotting",
    "states",
    "tomography",
    "waveplates",
]

EXPECTED_FUNCTIONS = {
    "states": [
        "pauli",
        "pauli_matrices",
        "ket_zero",
        "ket_one",
        "ket_h",
        "ket_v",
        "ket_d",
        "ket_a",
        "ket_r",
        "ket_l",
        "basis_ket",
        "tensor_product",
        "tensor_ket",
        "computational_basis",
        "bell_state",
        "bell_states",
        "projector",
        "proj",
        "projector_from_label",
        "tensor_projector",
        "density_matrix",
        "make_hermitian",
        "is_hermitian",
        "trace_normalize",
    ],
    "metrics": [
        "purity",
        "fidelity_pure",
        "fidelity",
        "bell_state_fidelities",
        "state_eigenvalues",
        "trace_distance",
        "concurrence",
        "linear_entropy",
    ],
    "mle": [
        "density_matrix_from_lower_triangular_params",
        "density_matrix_from_dense_params",
        "measurement_probabilities",
        "poisson_negative_log_likelihood",
        "poisson_chi2_loss",
        "fit_density_matrix_mle",
        "mle_diagnostics",
    ],
    "tomography": [
        "bloch_vector_from_axis_probabilities",
        "single_qubit_axis_probabilities_from_counts",
        "single_qubit_density_matrix_from_probabilities",
        "single_qubit_density_matrix_from_counts",
        "project_to_physical_density_matrix",
        "measurement_projectors_from_labels",
        "linear_inversion_tomography",
        "reconstruct_density_matrix",
        "reconstruction_summary",
    ],
    "bell": [
        "unit_vector",
        "axis_observable",
        "two_qubit_correlation",
        "pauli_axis_correlations",
        "bell_state_projectors",
        "bell_state_fidelities",
        "dominant_bell_state",
    ],
    "chsh": [
        "correlator_from_counts",
        "correlators_from_counts",
        "correlators_from_rho",
        "chsh_s_value",
        "chsh_s_from_rho",
        "violation_margin",
    ],
    "waveplates": [
        "analyzer_unitary_from_waveplates",
        "bloch_direction_from_state",
        "quarter_wave_plate_matrix",
        "half_wave_plate_matrix",
        "measurement_basis_for_bloch_direction",
        "measurement_basis_from_waveplates",
        "waveplate_settings_for_bloch_direction",
        "waveplate_settings_for_label",
        "waveplate_settings_for_state",
    ],
    "io": [
        "standardize_polarization_label",
        "standardize_chsh_setting_label",
        "standardize_chsh_outcome_label",
        "standardize_counts_table",
        "counts_dict_from_table",
        "load_counts_table",
        "save_counts_table",
        "load_density_matrix",
        "save_density_matrix",
    ],
    "plotting": [
        "plot_density_matrix",
        "plot_coincidence_counts",
        "plot_bell_state_fidelities",
    ],
}


def test_package_exports_expected_modules() -> None:
    """The top-level package should expose the expected public modules."""
    assert pec.__version__ == "0.1.0"
    assert set(EXPECTED_MODULES).issubset(pec.__all__)

    for module_name in EXPECTED_MODULES:
        assert hasattr(pec, module_name)


def test_module_public_functions_are_documented() -> None:
    """Each scaffolded function should be part of the module public surface."""
    for module_name, function_names in EXPECTED_FUNCTIONS.items():
        module = import_module(f"pec.{module_name}")

        for function_name in function_names:
            function = getattr(module, function_name)

            assert function_name in module.__all__
            assert inspect.isfunction(function)
            assert function.__doc__
