"""Tomography workflows for photonic state reconstruction."""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

import numpy as np
import numpy.linalg as npl
from numpy.typing import ArrayLike, NDArray

from . import bell
from . import chsh
from . import metrics
from . import mle
from . import states

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
TomographyMethod = Literal["linear_inversion", "mle"]

__all__ = [
    "TomographyMethod",
    "bloch_vector_from_axis_probabilities",
    "default_stats_fn",
    "linear_inversion_tomography",
    "measurement_projectors_from_labels",
    "project_to_physical_density_matrix",
    "reconstruct_density_matrix",
    "reconstruction_summary",
    "single_qubit_axis_probabilities_from_counts",
    "single_qubit_density_matrix_from_counts",
    "single_qubit_density_matrix_from_probabilities",
]


def _as_observation_array(values: ArrayLike) -> RealArray:
    """Convert tomography observations into a flat float array."""
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _lookup_optional_count(counts: Mapping[str, float], *names: str) -> float | None:
    """Return the first available count-like value for a set of possible keys."""
    for name in names:
        if name in counts:
            return float(counts[name])
    return None


def _pair_probability(
    counts: Mapping[str, float],
    *,
    positive_names: tuple[str, ...],
    negative_names: tuple[str, ...],
    use_min_max_fallback: bool = False,
) -> float:
    """Convert a measurement pair into the probability of the positive outcome."""
    positive = _lookup_optional_count(counts, *positive_names)
    negative = _lookup_optional_count(counts, *negative_names)

    if positive is not None and negative is not None:
        total = positive + negative
        return float(positive / total) if total > 1e-15 else 0.5

    if use_min_max_fallback and positive is not None:
        p_max = _lookup_optional_count(counts, "P_max")
        p_min = _lookup_optional_count(counts, "P_min")
        if p_max is None or p_min is None:
            raise ValueError("Min/max calibration counts are required for this probability.")

        denominator = p_max - p_min
        if abs(denominator) < 1e-15:
            return 0.5
        return float(np.clip((positive - p_min) / denominator, 0.0, 1.0))

    raise ValueError(f"Missing counts for {positive_names[0]} and {negative_names[0]}.")


def _has_single_qubit_axis_data(counts: Mapping[str, float]) -> bool:
    """Return whether a count mapping matches the notebook single-qubit workflow."""
    has_h = "H" in counts or "P_H" in counts
    has_v = "V" in counts or "P_V" in counts
    has_d = "D" in counts or "P_D" in counts
    has_a = "A" in counts or "P_A" in counts
    has_r = "R" in counts or "P_R" in counts
    has_l = "L" in counts or "P_L" in counts
    has_minmax = "P_max" in counts and "P_min" in counts
    return (
        has_h
        and has_v
        and has_d
        and (has_a or has_minmax)
        and has_r
        and (has_l or has_minmax)
    )


def bloch_vector_from_axis_probabilities(p_h: float, p_d: float, p_r: float) -> RealArray:
    """Convert H/D/R probabilities into a single-qubit Bloch vector."""
    return np.array(
        [
            2.0 * p_d - 1.0,
            2.0 * p_r - 1.0,
            2.0 * p_h - 1.0,
        ],
        dtype=np.float64,
    )


def project_to_physical_density_matrix(rho: ArrayLike, *, eps: float = 0.0) -> ComplexArray:
    """Project a Hermitian estimate onto the PSD, trace-one density-matrix set."""
    rho_hermitian = states.make_hermitian(rho)
    eigenvalues, eigenvectors = npl.eigh(rho_hermitian)
    clipped = np.clip(eigenvalues, eps, None)
    if np.sum(clipped) <= 1e-15:
        clipped = np.ones_like(clipped) / clipped.size
    else:
        clipped = clipped / np.sum(clipped)
    return eigenvectors @ np.diag(clipped) @ eigenvectors.conj().T


def single_qubit_axis_probabilities_from_counts(counts: Mapping[str, float]) -> dict[str, float]:
    """Compute pH, pD, and pR from notebook-style single-qubit measurement counts."""
    p_h = _pair_probability(
        counts,
        positive_names=("H", "P_H"),
        negative_names=("V", "P_V"),
    )
    p_d = _pair_probability(
        counts,
        positive_names=("D", "P_D"),
        negative_names=("A", "P_A"),
        use_min_max_fallback=True,
    )
    p_r = _pair_probability(
        counts,
        positive_names=("R", "P_R"),
        negative_names=("L", "P_L"),
        use_min_max_fallback=True,
    )
    return {"pH": p_h, "pD": p_d, "pR": p_r}


def single_qubit_density_matrix_from_probabilities(
    p_h: float,
    p_d: float,
    p_r: float,
    *,
    physical: bool = True,
) -> ComplexArray:
    """Reconstruct a single-qubit density matrix from H/D/R basis probabilities."""
    x, y, z = bloch_vector_from_axis_probabilities(p_h, p_d, p_r)
    rho_linear = 0.5 * (
        states.pauli("I")
        + x * states.pauli("X")
        + y * states.pauli("Y")
        + z * states.pauli("Z")
    )
    return project_to_physical_density_matrix(rho_linear) if physical else states.make_hermitian(rho_linear)


def single_qubit_density_matrix_from_counts(
    counts: Mapping[str, float],
    *,
    physical: bool = True,
) -> ComplexArray:
    """Reconstruct a single-qubit state from notebook-style axis counts or powers."""
    probabilities = single_qubit_axis_probabilities_from_counts(counts)
    return single_qubit_density_matrix_from_probabilities(
        probabilities["pH"],
        probabilities["pD"],
        probabilities["pR"],
        physical=physical,
    )


def measurement_projectors_from_labels(basis_labels: Sequence[str]) -> list[ComplexArray]:
    """Build single- or two-qubit projectors from basis labels like H, R, or HV."""
    projectors: list[ComplexArray] = []
    for raw_label in basis_labels:
        label = raw_label.strip()
        if not label:
            raise ValueError("Measurement labels must be non-empty.")
        if len(label) == 1:
            projectors.append(states.projector_from_label(label))
        else:
            projectors.append(states.tensor_projector(label))
    return projectors


def linear_inversion_tomography(
    counts: ArrayLike,
    measurement_operators: Sequence[ArrayLike],
    *,
    physical: bool = True,
) -> ComplexArray:
    """Reconstruct a density matrix from probability-like measurement observations."""
    observations = _as_observation_array(counts)
    operator_stack = np.asarray(measurement_operators, dtype=np.complex128)
    if operator_stack.ndim != 3 or operator_stack.shape[1] != operator_stack.shape[2]:
        raise ValueError("measurement_operators must be a sequence of square matrices.")
    if observations.size != operator_stack.shape[0]:
        raise ValueError("counts and measurement_operators must have the same length.")

    design_matrix = operator_stack.transpose(0, 2, 1).reshape(operator_stack.shape[0], -1)
    rho_vector, *_ = npl.lstsq(design_matrix, observations, rcond=None)
    rho_linear = rho_vector.reshape(operator_stack.shape[1], operator_stack.shape[2])
    rho_linear = states.make_hermitian(rho_linear)
    return project_to_physical_density_matrix(rho_linear) if physical else rho_linear


def reconstruct_density_matrix(
    counts: ArrayLike | Mapping[str, float],
    measurement_operators: Sequence[ArrayLike] | None = None,
    *,
    measurement_labels: Sequence[str] | None = None,
    method: TomographyMethod = "linear_inversion",
    physical: bool = True,
    mle_parameterization: mle.Parameterization = "lower_triangular",
    mle_objective: mle.ObjectiveName = "poisson_nll",
) -> ComplexArray:
    """Reconstruct a density matrix from notebook-style counts or normalized observations.

    For `method="linear_inversion"`, generic array or mapping inputs are interpreted as
    probability-like measurement observations unless they match the notebook single-qubit
    H/D/R workflow handled by :func:`single_qubit_density_matrix_from_counts`.
    For `method="mle"`, inputs are passed through to the reusable MLE layer, which is the
    intended path for raw count data with supplied measurement operators or labels.
    """
    if (
        method == "linear_inversion"
        and measurement_operators is None
        and measurement_labels is None
        and isinstance(counts, Mapping)
        and _has_single_qubit_axis_data(counts)
    ):
        return single_qubit_density_matrix_from_counts(counts, physical=physical)

    if isinstance(counts, Mapping):
        labels = list(counts) if measurement_labels is None else list(measurement_labels)
        observations = np.array([float(counts[label]) for label in labels], dtype=np.float64)
        if measurement_operators is None:
            measurement_operators = measurement_projectors_from_labels(labels)
    else:
        observations = _as_observation_array(counts)
        if measurement_operators is None:
            if measurement_labels is None:
                raise ValueError("measurement_operators or measurement_labels are required.")
            measurement_operators = measurement_projectors_from_labels(measurement_labels)

    if method == "linear_inversion":
        return linear_inversion_tomography(observations, measurement_operators, physical=physical)
    if method == "mle":
        return mle.fit_density_matrix_mle(
            observations,
            measurement_operators,
            parameterization=mle_parameterization,
            objective=mle_objective,
        )
    raise ValueError(f"Unsupported tomography method: {method!r}")


def default_stats_fn(rho: ComplexArray) -> dict[str, float]:
    """Standard metrics for bootstrap reporting: purity, concurrence, tangle, and Bell fidelities."""
    from . import metrics, bell
    result = {
        "purity": metrics.purity(rho),
        "concurrence": metrics.concurrence(rho),
        "tangle": metrics.concurrence(rho) ** 2,
    }
    for label, f in bell.bell_state_fidelities(rho).items():
        result[f"fidelity_{label}"] = f
    return result


def reconstruction_summary(
    rho: ArrayLike,
    *,
    target_state: ArrayLike | None = None,
    chsh_settings: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike] | None = None,
) -> dict[str, Any]:
    """Summarize physically relevant state-reconstruction quantities."""
    rho_physical = project_to_physical_density_matrix(rho)
    summary: dict[str, Any] = {
        "trace": float(np.real(np.trace(rho_physical))),
        "purity": metrics.purity(rho_physical),
        "eigenvalues": metrics.state_eigenvalues(rho_physical),
    }

    if target_state is not None:
        summary["fidelity"] = metrics.fidelity_pure(rho_physical, target_state)

    if rho_physical.shape == (4, 4):
        summary["bell_state_fidelities"] = bell.bell_state_fidelities(rho_physical)
        summary["dominant_bell_state"] = bell.dominant_bell_state(rho_physical)
        summary["pauli_axis_correlations"] = bell.pauli_axis_correlations(rho_physical)

        if chsh_settings is not None:
            a, a_prime, b, b_prime = chsh_settings
            summary["chsh_s"] = chsh.chsh_s_from_rho(rho_physical, a, a_prime, b, b_prime)

    return summary
