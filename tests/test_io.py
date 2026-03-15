"""Behavioral tests for lab-facing data-ingestion helpers."""

from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from pec import io


def test_standardize_polarization_label_accepts_common_lab_aliases() -> None:
    """Polarization labels should normalize across case and separator variations."""
    assert io.standardize_polarization_label(" hv ") == "HV"
    assert io.standardize_polarization_label("p-h") == "P_H"
    assert io.standardize_polarization_label("P min") == "P_min"


def test_standardize_chsh_labels_accept_common_aliases() -> None:
    """CHSH settings and outcomes should normalize to the package conventions."""
    assert io.standardize_chsh_setting_label("a,b'") == "abp"
    assert io.standardize_chsh_setting_label("a' , b") == "apb"
    assert io.standardize_chsh_outcome_label("pm") == "+-"
    assert io.standardize_chsh_outcome_label("- +") == "-+"


def test_standardize_counts_table_normalizes_tall_measurement_exports() -> None:
    """Spreadsheet-style measurement/count tables should become canonical tall tables."""
    raw = pd.DataFrame(
        {
            "Measurement State": [" hh ", "rl", "P h"],
            "Coincidence Counts": ["34749", "33586", "80"],
        }
    )

    standardized = io.standardize_counts_table(raw)

    assert list(standardized.index) == ["HH", "RL", "P_H"]
    assert list(standardized.columns) == ["counts"]
    assert np.allclose(standardized["counts"].to_numpy(), np.array([34749.0, 33586.0, 80.0]))


def test_standardize_counts_table_normalizes_chsh_wide_exports() -> None:
    """Wide CHSH exports should normalize both setting and outcome labels."""
    raw = pd.DataFrame(
        {
            "Setting": ["a,b", "a,b'", "a',b", "a',b'"],
            "pp": [10, 11, 12, 13],
            "pm": [1, 2, 3, 4],
            "mp": [5, 6, 7, 8],
            "mm": [9, 10, 11, 12],
        }
    )

    standardized = io.standardize_counts_table(raw, index_col="Setting")

    assert list(standardized.index) == ["ab", "abp", "apb", "apbp"]
    assert list(standardized.columns) == ["++", "+-", "-+", "--"]
    assert standardized.loc["abp", "+-"] == 2


def test_counts_table_round_trip_csv_and_mapping_conversion(tmp_path) -> None:
    """Count tables should round-trip through CSV and remain easy to convert to mappings."""
    counts = pd.DataFrame({"counts": [34749.0, 324.0]}, index=pd.Index(["HH", "HV"], name="label"))
    path = tmp_path / "counts.csv"

    io.save_counts_table(counts, path)
    loaded = io.load_counts_table(path)
    mapping = io.counts_dict_from_table(loaded)

    assert list(loaded.index) == ["HH", "HV"]
    assert list(loaded.columns) == ["counts"]
    assert mapping == {"HH": 34749.0, "HV": 324.0}


def test_density_matrix_round_trips_in_text_and_binary_formats(tmp_path) -> None:
    """Density matrices should round-trip through both CSV and NPY storage."""
    rho = np.array(
        [
            [0.5 + 0.0j, 0.0 - 0.5j],
            [0.0 + 0.5j, 0.5 + 0.0j],
        ],
        dtype=np.complex128,
    )
    csv_path = tmp_path / "rho.csv"
    npy_path = tmp_path / "rho.npy"

    io.save_density_matrix(rho, csv_path)
    io.save_density_matrix(rho, npy_path)

    loaded_csv = io.load_density_matrix(csv_path)
    loaded_npy = io.load_density_matrix(npy_path)

    assert np.allclose(loaded_csv, rho)
    assert np.allclose(loaded_npy, rho)
