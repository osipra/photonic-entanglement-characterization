"""CHSH-specific analysis utilities for Bell inequality experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "chsh_s_value",
    "correlators_from_counts",
    "violation_margin",
]


def correlators_from_counts(counts: pd.DataFrame) -> dict[str, float]:
    """Convert coincidence counts into CHSH correlators."""
    raise NotImplementedError


def chsh_s_value(correlators: Mapping[str, float]) -> float:
    """Compute the CHSH S parameter from a correlator mapping."""
    raise NotImplementedError


def violation_margin(
    correlators: Mapping[str, float],
    *,
    classical_bound: float = 2.0,
) -> float:
    """Compute the amount by which a CHSH experiment exceeds a classical bound."""
    raise NotImplementedError
