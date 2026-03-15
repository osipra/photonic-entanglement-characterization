"""Optional QuTiP interoperability and validation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import metrics
from . import states

if TYPE_CHECKING:
    import qutip as qt

ComplexArray = NDArray[np.complex128]
ComparisonEntry = dict[str, float | bool]
ComparisonMap = dict[str, ComparisonEntry]

__all__ = [
    "compare_metric_values",
    "from_qobj",
    "qutip_expect",
    "qutip_fidelity",
    "qutip_purity",
    "qutip_trace_distance",
    "to_qobj",
]


def _require_qutip() -> Any:
    """Import QuTiP on demand for optional interoperability helpers."""
    try:
        import qutip as qt
    except ModuleNotFoundError as exc:
        raise ImportError(
            "QuTiP is required for pec.qutip_tools. Install the optional dependency with `pip install .[qutip]`."
        ) from exc
    return qt


def _qutip_metric_function(name: str):
    """Resolve a QuTiP metric helper across supported namespace layouts."""
    qt = _require_qutip()
    metrics_module = getattr(qt, "metrics", None)
    if metrics_module is not None and hasattr(metrics_module, name):
        return getattr(metrics_module, name)
    if hasattr(qt, name):
        return getattr(qt, name)
    raise AttributeError(f"QuTiP does not expose a metric function named {name!r}.")


def _as_complex_array(x: ArrayLike) -> ComplexArray:
    """Convert an array-like input into a dense complex array."""
    return np.asarray(x, dtype=np.complex128)


def _coerce_quantum_array(x: ArrayLike) -> tuple[ComplexArray, bool]:
    """Normalize ket-like and operator-like inputs into dense arrays."""
    array = _as_complex_array(x)

    if array.ndim == 1:
        return array.reshape(-1, 1), True

    if array.ndim == 2:
        if 1 in array.shape and array.shape[0] != array.shape[1]:
            return array.reshape(-1, 1), True
        if array.shape[0] == array.shape[1]:
            return array, False

    raise ValueError("Expected a ket-like vector or a square operator matrix.")


def _normalize_dims(
    dims: Sequence[int] | Sequence[Sequence[int]] | None,
    *,
    dimension: int,
    is_ket: bool,
) -> list[list[int]]:
    """Normalize a flat or nested dimension specification for QuTiP."""
    if dims is None:
        if is_ket:
            return [[dimension], [1]]
        return [[dimension], [dimension]]

    is_nested = (
        len(dims) == 2
        and all(isinstance(part, Sequence) and not isinstance(part, (str, bytes)) for part in dims)
    )
    if is_nested:
        left = [int(value) for value in dims[0]]  # type: ignore[index]
        right = [int(value) for value in dims[1]]  # type: ignore[index]
    else:
        left = [int(value) for value in dims]  # type: ignore[arg-type]
        right = [1] * len(left) if is_ket else left.copy()

    left_size = int(np.prod(left, dtype=np.int64))
    right_size = int(np.prod(right, dtype=np.int64))
    if left_size != dimension:
        raise ValueError("QuTiP dims do not match the Hilbert-space dimension of the input.")
    if is_ket and right_size != 1:
        raise ValueError("Ket dims must have output dimension 1.")
    if not is_ket and right_size != dimension:
        raise ValueError("Operator dims must match the matrix dimension on both sides.")

    return [left, right]


def _is_qobj_instance(x: object) -> bool:
    """Return whether an object is a QuTiP Qobj instance."""
    try:
        qt = _require_qutip()
    except ImportError:
        return False
    return isinstance(x, qt.Qobj)


def _as_density_matrix(x: ArrayLike | object) -> ComplexArray:
    """Convert ket-like or operator-like input into a NumPy density matrix."""
    array = from_qobj(x) if _is_qobj_instance(x) else _as_complex_array(x)  # type: ignore[arg-type]
    if array.ndim == 1:
        return states.density_matrix(array)
    if array.ndim == 2 and 1 in array.shape and array.shape[0] != array.shape[1]:
        return states.density_matrix(array.reshape(-1))
    if array.ndim == 2 and array.shape[0] == array.shape[1]:
        return array
    raise ValueError("Expected a ket-like vector or a square operator matrix.")


def _as_density_qobj(x: ArrayLike | object):
    """Convert ket-like or operator-like input into a QuTiP density operator."""
    qobj = to_qobj(x)
    if qobj.isket:
        return qobj.proj()
    if qobj.isbra:
        return qobj.dag().proj()
    if qobj.isoper and qobj.shape[0] == qobj.shape[1]:
        return qobj
    raise ValueError("Expected a ket-like vector or a square operator matrix.")


def _pure_ket_from_array(state: ArrayLike | object, *, atol: float = 1e-10) -> ComplexArray | None:
    """Return a normalized ket when a NumPy-like state is pure."""
    if _is_qobj_instance(state):
        pure_qobj = _pure_ket_from_qobj(state, atol=atol)
        return None if pure_qobj is None else from_qobj(pure_qobj)

    array = _as_complex_array(state)  # type: ignore[arg-type]
    if array.ndim == 1:
        norm = np.linalg.norm(array)
        if np.isclose(norm, 0.0):
            return None
        return array / norm

    if array.ndim == 2 and 1 in array.shape and array.shape[0] != array.shape[1]:
        vector = array.reshape(-1)
        norm = np.linalg.norm(vector)
        if np.isclose(norm, 0.0):
            return None
        return vector / norm

    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("Expected a ket-like vector or a square operator matrix.")

    rho = states.trace_normalize(states.make_hermitian(array))
    purity = metrics.purity(rho)
    if not np.isclose(purity, 1.0, atol=atol, rtol=0.0):
        return None

    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    index = int(np.argmax(np.real(eigenvalues)))
    dominant = float(np.real(eigenvalues[index]))
    if not np.isclose(dominant, 1.0, atol=atol, rtol=0.0):
        return None
    ket = np.asarray(eigenvectors[:, index], dtype=np.complex128)
    norm = np.linalg.norm(ket)
    if np.isclose(norm, 0.0):
        return None
    return ket / norm


def _pure_ket_from_qobj(state: object, *, atol: float = 1e-10):
    """Return a ket representation for a pure state when available."""
    qt = _require_qutip()
    qobj = to_qobj(state)

    if qobj.isbra:
        return qobj.dag()
    if qobj.isket:
        return qobj.unit()
    if not qobj.isoper or qobj.shape[0] != qobj.shape[1]:
        raise ValueError("Expected a ket-like vector or a square operator matrix.")

    purity = float(np.real((qobj * qobj).tr()))
    if not np.isclose(purity, 1.0, atol=atol, rtol=0.0):
        return None

    eigenvalues, eigenstates = qobj.eigenstates()
    index = int(np.argmax(np.real(eigenvalues)))
    dominant = float(np.real(eigenvalues[index]))
    if not np.isclose(dominant, 1.0, atol=atol, rtol=0.0):
        return None
    ket = eigenstates[index]
    if isinstance(ket, qt.Qobj) and ket.isbra:
        return ket.dag().unit()
    return ket.unit()


def _comparison_entry(
    pec_value: float,
    qutip_value: float,
    *,
    atol: float,
    rtol: float,
) -> ComparisonEntry:
    """Build a readable metric-comparison record."""
    abs_diff = abs(float(pec_value) - float(qutip_value))
    return {
        "pec": float(pec_value),
        "qutip": float(qutip_value),
        "abs_diff": float(abs_diff),
        "within_tolerance": bool(np.isclose(pec_value, qutip_value, atol=atol, rtol=rtol)),
    }


def to_qobj(
    x: ArrayLike | object,
    dims: Sequence[int] | Sequence[Sequence[int]] | None = None,
):
    """Convert a NumPy ket or operator into a QuTiP ``Qobj``."""
    qt = _require_qutip()
    if isinstance(x, qt.Qobj):
        if dims is None:
            return x
        dense, is_ket = _coerce_quantum_array(x.full())
        qobj_dims = _normalize_dims(dims, dimension=dense.shape[0], is_ket=is_ket)
        return qt.Qobj(dense, dims=qobj_dims)

    dense, is_ket = _coerce_quantum_array(x)  # type: ignore[arg-type]
    qobj_dims = _normalize_dims(dims, dimension=dense.shape[0], is_ket=is_ket)
    return qt.Qobj(dense, dims=qobj_dims)


def from_qobj(q: object) -> ComplexArray:
    """Convert a QuTiP ``Qobj`` back into a dense NumPy array."""
    qt = _require_qutip()
    if not isinstance(q, qt.Qobj):
        raise TypeError("from_qobj expects a QuTiP Qobj instance.")

    dense = np.asarray(q.full(), dtype=np.complex128)
    if q.isket or q.isbra:
        return dense.reshape(-1)
    return dense


def qutip_purity(rho: ArrayLike | object) -> float:
    """Compute the purity of a state using QuTiP operators."""
    density = _as_density_qobj(rho)
    return float(np.real((density * density).tr()))


def qutip_fidelity(a: ArrayLike | object, b: ArrayLike | object) -> float:
    """Compute fidelity using the same squared convention as ``pec.metrics``."""
    qt = _require_qutip()
    fidelity_fn = _qutip_metric_function("fidelity")

    pure_a = _pure_ket_from_qobj(a)
    pure_b = _pure_ket_from_qobj(b)

    if pure_a is not None and pure_b is not None:
        overlap = pure_a.overlap(pure_b)
        return float(np.clip(abs(overlap) ** 2, 0.0, 1.0))

    if pure_a is not None:
        expectation = qt.expect(_as_density_qobj(b), pure_a)
        return float(np.clip(np.real(expectation), 0.0, 1.0))

    if pure_b is not None:
        expectation = qt.expect(_as_density_qobj(a), pure_b)
        return float(np.clip(np.real(expectation), 0.0, 1.0))

    raw_fidelity = float(np.real(fidelity_fn(_as_density_qobj(a), _as_density_qobj(b))))
    return float(np.clip(raw_fidelity, 0.0, None) ** 2)


def qutip_trace_distance(a: ArrayLike | object, b: ArrayLike | object) -> float:
    """Compute the trace distance between two states with QuTiP."""
    tracedist_fn = _qutip_metric_function("tracedist")
    return float(np.real(tracedist_fn(_as_density_qobj(a), _as_density_qobj(b))))


def qutip_expect(op: ArrayLike | object, state: ArrayLike | object) -> float | complex:
    """Compute an expectation value with QuTiP."""
    qt = _require_qutip()
    value = np.real_if_close(qt.expect(to_qobj(op), to_qobj(state)))
    if np.iscomplexobj(value):
        return complex(value)
    return float(value)


def compare_metric_values(
    a: ArrayLike | object,
    b: ArrayLike | object | None = None,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-8,
) -> ComparisonMap:
    """Compare PEC metric values against QuTiP for one or two state inputs."""
    rho = _as_density_matrix(a)
    comparisons: ComparisonMap = {
        "purity": _comparison_entry(
            metrics.purity(rho),
            qutip_purity(a),
            atol=atol,
            rtol=rtol,
        )
    }

    if b is not None:
        sigma = _as_density_matrix(b)
        pure_a = _pure_ket_from_array(a)
        pure_b = _pure_ket_from_array(b)

        if pure_b is not None:
            pec_fidelity = metrics.fidelity_pure(rho, pure_b)
        elif pure_a is not None:
            pec_fidelity = metrics.fidelity_pure(sigma, pure_a)
        else:
            pec_fidelity = metrics.fidelity(rho, sigma)

        comparisons["fidelity"] = _comparison_entry(
            pec_fidelity,
            qutip_fidelity(a, b),
            atol=atol,
            rtol=rtol,
        )
        comparisons["trace_distance"] = _comparison_entry(
            metrics.trace_distance(rho, sigma),
            qutip_trace_distance(a, b),
            atol=atol,
            rtol=rtol,
        )

    return comparisons
