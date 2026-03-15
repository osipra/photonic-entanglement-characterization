"""Maximum-likelihood estimation utilities for quantum state reconstruction."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import numpy.linalg as npl
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from . import states

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
Parameterization = Literal["lower_triangular", "dense"]
ObjectiveName = Literal["poisson_nll", "poisson_chi2"]

__all__ = [
    "density_matrix_from_dense_params",
    "density_matrix_from_lower_triangular_params",
    "fit_density_matrix_mle",
    "measurement_probabilities",
    "mle_diagnostics",
    "poisson_chi2_loss",
    "poisson_negative_log_likelihood",
]


def _as_counts_array(counts: ArrayLike) -> RealArray:
    """Convert count-like input into a flat float array."""
    return np.asarray(counts, dtype=np.float64).reshape(-1)


def _as_operator_stack(measurement_operators: Sequence[ArrayLike]) -> ComplexArray:
    """Convert measurement operators into a dense stack of square matrices."""
    operator_stack = np.asarray(measurement_operators, dtype=np.complex128)
    if operator_stack.ndim != 3:
        raise ValueError("measurement_operators must be a sequence of square matrices.")
    if operator_stack.shape[1] != operator_stack.shape[2]:
        raise ValueError("measurement_operators must be square.")
    return operator_stack


def _parameter_count_for_lower_triangular(dimension: int) -> int:
    """Return the number of real parameters for a lower-triangular PSD factor."""
    return dimension * dimension


def _parameter_count_for_dense(dimension: int) -> int:
    """Return the number of real parameters for a full complex PSD factor."""
    return 2 * dimension * dimension


def _pack_lower_triangular_factor(factor: ArrayLike) -> RealArray:
    """Pack a lower-triangular complex factor into real optimization parameters."""
    lower = np.asarray(factor, dtype=np.complex128)
    dimension = lower.shape[0]
    params = np.zeros(_parameter_count_for_lower_triangular(dimension), dtype=np.float64)
    idx = 0

    for i in range(dimension):
        params[idx] = float(np.real(lower[i, i]))
        idx += 1

    for i in range(1, dimension):
        for j in range(i):
            params[idx] = float(np.real(lower[i, j]))
            params[idx + 1] = float(np.imag(lower[i, j]))
            idx += 2

    return params


def _pack_dense_factor(factor: ArrayLike) -> RealArray:
    """Pack a dense complex factor into alternating real and imaginary parameters."""
    dense = np.asarray(factor, dtype=np.complex128)
    params = np.zeros(_parameter_count_for_dense(dense.shape[0]), dtype=np.float64)
    params[0::2] = np.real(dense).reshape(-1)
    params[1::2] = np.imag(dense).reshape(-1)
    return params


def _psd_factor_from_density_matrix(rho: ArrayLike) -> ComplexArray:
    """Return a dense square-root factor whose product reconstructs rho."""
    rho_matrix = states.trace_normalize(states.make_hermitian(rho))
    eigenvalues, eigenvectors = npl.eigh(rho_matrix)
    clipped = np.clip(eigenvalues, 0.0, None)
    return eigenvectors @ np.diag(np.sqrt(clipped))


def density_matrix_from_lower_triangular_params(
    params: ArrayLike,
    dimension: int,
) -> ComplexArray:
    """Map a lower-triangular PSD factor parameterization to a valid density matrix."""
    flat_params = np.asarray(params, dtype=np.float64).reshape(-1)
    expected = _parameter_count_for_lower_triangular(dimension)
    if flat_params.size != expected:
        raise ValueError(f"Expected {expected} parameters for dimension {dimension}.")

    factor = np.zeros((dimension, dimension), dtype=np.complex128)
    idx = 0

    for i in range(dimension):
        factor[i, i] = flat_params[idx]
        idx += 1

    for i in range(1, dimension):
        for j in range(i):
            factor[i, j] = flat_params[idx] + 1j * flat_params[idx + 1]
            idx += 2

    return states.trace_normalize(factor @ factor.conj().T)


def density_matrix_from_dense_params(params: ArrayLike, dimension: int) -> ComplexArray:
    """Map a dense complex PSD factor parameterization to a valid density matrix."""
    flat_params = np.asarray(params, dtype=np.float64).reshape(-1)
    expected = _parameter_count_for_dense(dimension)
    if flat_params.size != expected:
        raise ValueError(f"Expected {expected} parameters for dimension {dimension}.")

    factor = (flat_params[0::2] + 1j * flat_params[1::2]).reshape(dimension, dimension)
    return states.trace_normalize(factor @ factor.conj().T)


def measurement_probabilities(
    rho: ArrayLike,
    measurement_operators: Sequence[ArrayLike],
) -> RealArray:
    """Compute Born-rule probabilities for a density matrix and operator list."""
    rho_matrix = np.asarray(rho, dtype=np.complex128)
    operator_stack = _as_operator_stack(measurement_operators)
    dimension = operator_stack.shape[1]
    if rho_matrix.shape != (dimension, dimension):
        raise ValueError("rho shape must match the measurement-operator dimension.")

    probabilities = np.real(np.einsum("kij,ji->k", operator_stack, rho_matrix))
    return np.asarray(probabilities, dtype=np.float64)


def _estimate_total_counts(counts: RealArray, probabilities: RealArray, *, floor: float) -> float:
    """Estimate the total count scale using the notebook normalization rule."""
    denominator = float(np.sum(np.clip(probabilities, floor, None)))
    if denominator <= floor:
        return float(np.sum(counts))
    return float(np.sum(counts) / denominator)


def poisson_negative_log_likelihood(
    counts: ArrayLike,
    measurement_operators: Sequence[ArrayLike],
    rho: ArrayLike,
    *,
    total_counts: float | None = None,
    floor: float = 1e-12,
) -> float:
    """Compute the Poisson negative log-likelihood up to a data-only constant."""
    count_array = _as_counts_array(counts)
    probabilities = np.clip(measurement_probabilities(rho, measurement_operators), floor, None)
    if count_array.size != probabilities.size:
        raise ValueError("counts and measurement_operators must have the same length.")
    scale = total_counts if total_counts is not None else _estimate_total_counts(count_array, probabilities, floor=floor)
    mu = np.clip(scale * probabilities, floor, None)
    return float(np.sum(mu - count_array * np.log(mu)))


def poisson_chi2_loss(
    counts: ArrayLike,
    measurement_operators: Sequence[ArrayLike],
    rho: ArrayLike,
    *,
    total_counts: float | None = None,
    floor: float = 1e-12,
) -> float:
    """Compute the notebook-style Poisson chi-squared approximation."""
    count_array = _as_counts_array(counts)
    probabilities = np.clip(measurement_probabilities(rho, measurement_operators), floor, None)
    if count_array.size != probabilities.size:
        raise ValueError("counts and measurement_operators must have the same length.")
    scale = total_counts if total_counts is not None else _estimate_total_counts(count_array, probabilities, floor=floor)
    mu = np.clip(scale * probabilities, floor, None)
    return float(np.sum(((mu - count_array) ** 2) / (2.0 * mu)))


def _parameterization_function(parameterization: Parameterization):
    """Return the density-matrix constructor for a named parameterization."""
    if parameterization == "lower_triangular":
        return density_matrix_from_lower_triangular_params
    if parameterization == "dense":
        return density_matrix_from_dense_params
    raise ValueError(f"Unsupported MLE parameterization: {parameterization!r}")


def _objective_function(objective: ObjectiveName):
    """Return the scalar objective function for a named MLE loss."""
    if objective == "poisson_nll":
        return poisson_negative_log_likelihood
    if objective == "poisson_chi2":
        return poisson_chi2_loss
    raise ValueError(f"Unsupported MLE objective: {objective!r}")


def _initial_params(
    dimension: int,
    parameterization: Parameterization,
    initial_state: ArrayLike | None,
) -> RealArray:
    """Build an optimizer seed for the requested parameterization."""
    if initial_state is None:
        if parameterization == "lower_triangular":
            seed = np.zeros(_parameter_count_for_lower_triangular(dimension), dtype=np.float64)
            seed[:dimension] = 1.0
            return seed

        seed = np.zeros(_parameter_count_for_dense(dimension), dtype=np.float64)
        factor = np.eye(dimension, dtype=np.complex128)
        return _pack_dense_factor(factor)

    rho_seed = states.trace_normalize(
        states.make_hermitian(initial_state) + 1e-12 * np.eye(dimension, dtype=np.complex128)
    )
    if parameterization == "lower_triangular":
        factor = npl.cholesky(rho_seed)
        return _pack_lower_triangular_factor(factor)

    return _pack_dense_factor(_psd_factor_from_density_matrix(rho_seed))


def fit_density_matrix_mle(
    counts: ArrayLike,
    measurement_operators: Sequence[ArrayLike],
    *,
    initial_state: ArrayLike | None = None,
    parameterization: Parameterization = "lower_triangular",
    objective: ObjectiveName = "poisson_nll",
    fit_total_counts: bool = True,
    method: str = "Powell",
    max_iterations: int = 1_000,
    tolerance: float = 1e-9,
) -> ComplexArray:
    """Estimate a physical density matrix with a reusable MLE optimization loop."""
    operator_stack = _as_operator_stack(measurement_operators)
    count_array = _as_counts_array(counts)
    if count_array.size != operator_stack.shape[0]:
        raise ValueError("counts and measurement_operators must have the same length.")

    dimension = operator_stack.shape[1]
    parameterization_fn = _parameterization_function(parameterization)
    objective_fn = _objective_function(objective)
    params0 = _initial_params(dimension, parameterization, initial_state)

    if fit_total_counts:
        rho0 = parameterization_fn(params0, dimension)
        probs0 = np.clip(measurement_probabilities(rho0, operator_stack), 1e-12, None)
        total0 = _estimate_total_counts(count_array, probs0, floor=1e-12)
        x0 = np.concatenate([params0, [np.log(max(total0, 1e-12))]])

        def loss(theta: ArrayLike) -> float:
            theta_array = np.asarray(theta, dtype=np.float64)
            rho_trial = parameterization_fn(theta_array[:-1], dimension)
            total_counts = float(np.exp(theta_array[-1]))
            return objective_fn(
                count_array,
                operator_stack,
                rho_trial,
                total_counts=total_counts,
            )

    else:
        total_fixed = float(np.sum(count_array))
        x0 = params0

        def loss(theta: ArrayLike) -> float:
            theta_array = np.asarray(theta, dtype=np.float64)
            rho_trial = parameterization_fn(theta_array, dimension)
            return objective_fn(
                count_array,
                operator_stack,
                rho_trial,
                total_counts=total_fixed,
            )

    options = {"maxiter": max_iterations}
    if method == "Powell":
        options.update({"xtol": tolerance, "ftol": tolerance})
    else:
        options.update({"xatol": tolerance, "fatol": tolerance})

    result = minimize(loss, x0, method=method, options=options)
    best_params = np.asarray(result.x[:-1] if fit_total_counts else result.x, dtype=np.float64)
    return parameterization_fn(best_params, dimension)


def mle_diagnostics(
    counts: ArrayLike,
    measurement_operators: Sequence[ArrayLike],
    estimate: ArrayLike,
) -> dict[str, Any]:
    """Summarize probabilities, count scale, and objective values for an estimate."""
    count_array = _as_counts_array(counts)
    operator_stack = _as_operator_stack(measurement_operators)
    if count_array.size != operator_stack.shape[0]:
        raise ValueError("counts and measurement_operators must have the same length.")
    rho_estimate = states.trace_normalize(states.make_hermitian(estimate))
    probabilities = np.clip(measurement_probabilities(rho_estimate, operator_stack), 1e-12, None)
    total_counts = _estimate_total_counts(count_array, probabilities, floor=1e-12)
    expected_counts = total_counts * probabilities
    eigenvalues = npl.eigvalsh(rho_estimate)

    return {
        "probabilities": probabilities,
        "expected_counts": expected_counts,
        "total_counts": total_counts,
        "trace": float(np.real(np.trace(rho_estimate))),
        "min_eigenvalue": float(np.min(np.real(eigenvalues))),
        "poisson_nll": poisson_negative_log_likelihood(
            count_array,
            operator_stack,
            rho_estimate,
            total_counts=total_counts,
        ),
        "poisson_chi2": poisson_chi2_loss(
            count_array,
            operator_stack,
            rho_estimate,
            total_counts=total_counts,
        ),
    }
