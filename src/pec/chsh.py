"""CHSH-specific analysis utilities for Bell inequality experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy.typing import ArrayLike

from . import bell

if TYPE_CHECKING:
    import pandas as pd

OutcomeCounts = Mapping[str, float]
CorrelatorMap = Mapping[str, float]

_SETTING_KEYS = ("ab", "abp", "apb", "apbp")
_OUTCOME_ALIASES = {
    "++": "++",
    "+-": "+-",
    "-+": "-+",
    "--": "--",
    "pp": "++",
    "pm": "+-",
    "mp": "-+",
    "mm": "--",
}

__all__ = [
    "chsh_s_from_rho",
    "chsh_s_value",
    "correlator_from_counts",
    "correlators_from_counts",
    "correlators_from_rho",
    "violation_margin",
]


def _normalized_outcome_counts(counts: OutcomeCounts) -> dict[str, float]:
    """Map supported CHSH outcome labels onto the canonical ++,+-,-+,-- names."""
    normalized = {"++": 0.0, "+-": 0.0, "-+": 0.0, "--": 0.0}

    for raw_label, value in counts.items():
        label = raw_label.replace(" ", "").lower()
        try:
            canonical_label = _OUTCOME_ALIASES[label]
        except KeyError as exc:
            raise ValueError(f"Unsupported CHSH outcome label: {raw_label!r}") from exc
        normalized[canonical_label] += float(value)

    return normalized


def correlator_from_counts(counts: OutcomeCounts) -> float:
    """Compute a single CHSH correlator from ++,+-,-+,-- coincidence counts."""
    outcomes = _normalized_outcome_counts(counts)
    total = sum(outcomes.values())
    if np.isclose(total, 0.0):
        raise ValueError("CHSH correlators require a nonzero total count.")

    same = outcomes["++"] + outcomes["--"]
    different = outcomes["+-"] + outcomes["-+"]
    return float((same - different) / total)


def correlators_from_counts(counts: Mapping[str, OutcomeCounts] | pd.DataFrame) -> dict[str, float]:
    """Compute the four canonical CHSH correlators from setting-indexed counts."""
    if hasattr(counts, "iterrows"):
        row_mapping = {
            str(index): {str(column): float(value) for column, value in row.items()}
            for index, row in counts.iterrows()
        }
    else:
        row_mapping = counts

    missing = [key for key in _SETTING_KEYS if key not in row_mapping]
    if missing:
        raise ValueError(f"Missing CHSH settings: {missing!r}")

    return {
        key: correlator_from_counts(row_mapping[key])
        for key in _SETTING_KEYS
    }


def correlators_from_rho(
    rho: ArrayLike,
    a: ArrayLike,
    a_prime: ArrayLike,
    b: ArrayLike,
    b_prime: ArrayLike,
) -> dict[str, float]:
    """Compute the four CHSH correlators directly from rho and analyzer directions."""
    return {
        "ab": bell.two_qubit_correlation(rho, a, b),
        "abp": bell.two_qubit_correlation(rho, a, b_prime),
        "apb": bell.two_qubit_correlation(rho, a_prime, b),
        "apbp": bell.two_qubit_correlation(rho, a_prime, b_prime),
    }


def chsh_s_value(correlators: CorrelatorMap) -> float:
    """Compute S = E(a,b) + E(a,b') + E(a',b) - E(a',b')."""
    missing = [key for key in _SETTING_KEYS if key not in correlators]
    if missing:
        raise ValueError(f"Missing CHSH correlators: {missing!r}")

    return float(
        correlators["ab"]
        + correlators["abp"]
        + correlators["apb"]
        - correlators["apbp"]
    )


def chsh_s_from_rho(
    rho: ArrayLike,
    a: ArrayLike,
    a_prime: ArrayLike,
    b: ArrayLike,
    b_prime: ArrayLike,
) -> float:
    """Compute the CHSH S value directly from rho and four analyzer directions."""
    return chsh_s_value(correlators_from_rho(rho, a, a_prime, b, b_prime))


def violation_margin(
    correlators: CorrelatorMap,
    *,
    classical_bound: float = 2.0,
) -> float:
    """Compute by how much |S| exceeds a chosen CHSH bound."""
    return abs(chsh_s_value(correlators)) - classical_bound
