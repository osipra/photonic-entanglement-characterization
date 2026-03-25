"""Plotting helpers for photonic quantum information analyses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
DensityMatrixComponent = Literal["real", "imag"]

_BELL_CANONICAL_ORDER = [
    ("phi_plus", "Phi+"),
    ("phi_minus", "Phi-"),
    ("psi_plus", "Psi+"),
    ("psi_minus", "Psi-"),
]
_CHSH_CANONICAL_ORDER = [
    ("ab", "E(a,b)"),
    ("abp", "E(a,b')"),
    ("apb", "E(a',b)"),
    ("apbp", "E(a',b')"),
]

__all__ = [
    "plot_bell_state_fidelities",
    "plot_chsh_correlators",
    "plot_coincidence_counts",
    "plot_density_matrix",
]


def _require_matplotlib():
    """Import matplotlib on demand so the package can import without it."""
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for pec.plotting helpers. Install the project dependencies to use pec.plotting."
        ) from exc
    return matplotlib


def _default_basis_labels(dimension: int) -> list[str]:
    """Return sensible basis labels for common single- and two-qubit matrices."""
    if dimension == 2:
        return ["H", "V"]
    if dimension == 4:
        return ["HH", "HV", "VH", "VV"]
    return [str(index) for index in range(dimension)]


def _component_matrix(rho: ComplexArray, component: DensityMatrixComponent) -> RealArray:
    """Extract the requested real or imaginary density-matrix component."""
    if component == "real":
        return np.asarray(np.real(rho), dtype=np.float64)
    if component == "imag":
        return np.asarray(np.imag(rho), dtype=np.float64)
    raise ValueError(f"Unsupported density-matrix component: {component!r}")


def _annotation_color(value: float, scale: float) -> str:
    """Choose a readable text color for a heatmap annotation."""
    if np.isclose(scale, 0.0):
        return "black"
    return "white" if abs(value) > 0.5 * scale else "black"


def _label_value_pairs(
    values: Mapping[str, float] | ArrayLike | object,
    *,
    labels: Sequence[str] | None = None,
) -> tuple[list[str], RealArray]:
    """Normalize mapping-, series-, and array-like inputs into labels and values."""
    if isinstance(values, Mapping):
        keys = [str(label) for label in values.keys()]
        numeric_values = np.asarray(list(values.values()), dtype=np.float64)
        return keys, numeric_values

    if hasattr(values, "columns") and hasattr(values, "index") and hasattr(values, "to_numpy"):
        shape = getattr(values, "shape", None)
        if shape is None:
            raise ValueError("Could not determine the shape of the count table.")
        if len(shape) != 2:
            raise ValueError("Tabular count inputs must be two-dimensional.")
        if shape[1] == 1:
            return (
                [str(label) for label in values.index],
                np.asarray(values.iloc[:, 0].to_numpy(), dtype=np.float64),
            )
        if shape[0] == 1:
            return (
                [str(label) for label in values.columns],
                np.asarray(values.iloc[0, :].to_numpy(), dtype=np.float64),
            )
        raise ValueError("Count tables must have exactly one row or one column for plotting.")

    if hasattr(values, "index") and hasattr(values, "to_numpy"):
        return (
            [str(label) for label in values.index],
            np.asarray(values.to_numpy(), dtype=np.float64).reshape(-1),
        )

    numeric_values = np.asarray(values, dtype=np.float64).reshape(-1)
    if labels is None:
        plot_labels = [str(index) for index in range(numeric_values.size)]
    else:
        plot_labels = [str(label) for label in labels]
        if len(plot_labels) != numeric_values.size:
            raise ValueError("labels must match the number of plotted values.")
    return plot_labels, numeric_values


def _ordered_mapping_values(
    values: Mapping[str, float],
    canonical_order: Sequence[tuple[str, str]],
) -> tuple[list[str], RealArray]:
    """Order a mapping according to a canonical internal/public label sequence."""
    normalized = {
        str(key).lower().replace("'", "").replace("+", "_plus").replace("-", "_minus"): float(value)
        for key, value in values.items()
    }
    pretty_labels: list[str] = []
    ordered_values: list[float] = []
    for internal_label, pretty_label in canonical_order:
        if internal_label in normalized:
            pretty_labels.append(pretty_label)
            ordered_values.append(normalized[internal_label])

    if pretty_labels:
        return pretty_labels, np.asarray(ordered_values, dtype=np.float64)

    return _label_value_pairs(values)


def plot_density_matrix(
    rho: ArrayLike,
    *,
    ax: Axes | Sequence[Axes] | None = None,
    title: str | None = None,
    basis_labels: Sequence[str] | None = None,
    components: Sequence[DensityMatrixComponent] = ("real", "imag"),
    annotate: bool = True,
    colorbar: bool = True,
    cmap: str = "RdBu_r",
) -> Figure:
    """Visualize the real and/or imaginary parts of a density matrix as heatmaps."""
    matplotlib = _require_matplotlib()
    matrix = np.asarray(rho, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("rho must be a square 2D matrix.")

    component_list = [str(component) for component in components]
    if not component_list:
        raise ValueError("components must contain at least one matrix component.")

    dimension = matrix.shape[0]
    tick_labels = list(basis_labels) if basis_labels is not None else _default_basis_labels(dimension)
    if len(tick_labels) != dimension:
        raise ValueError("basis_labels must match the density-matrix dimension.")

    if ax is None:
        figure = matplotlib.figure.Figure(
            figsize=(4.5 * len(component_list), 4.0),
            constrained_layout=True,
        )
        axes_list = list(np.atleast_1d(figure.subplots(1, len(component_list))))
    elif hasattr(ax, "imshow"):
        if len(component_list) != 1:
            raise ValueError("A single Axes can only be used when plotting one component.")
        axes_list = [ax]
        figure = ax.figure
    else:
        axes_list = list(ax)
        if len(axes_list) != len(component_list):
            raise ValueError("The number of axes must match the requested matrix components.")
        figure = axes_list[0].figure

    scale = max(np.max(np.abs(_component_matrix(matrix, component))) for component in component_list)
    scale = float(scale if scale > 0.0 else 1.0)

    for axis, component in zip(axes_list, component_list, strict=True):
        data = _component_matrix(matrix, component)
        image = axis.imshow(data, cmap=cmap, vmin=-scale, vmax=scale, interpolation="nearest")
        axis.set_xticks(range(dimension), tick_labels, rotation=45, ha="right")
        axis.set_yticks(range(dimension), tick_labels)
        axis.set_xlabel("Ket")
        axis.set_ylabel("Bra")
        axis.set_title(component.capitalize())

        if annotate:
            for row in range(dimension):
                for column in range(dimension):
                    value = float(data[row, column])
                    axis.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=_annotation_color(value, scale),
                        fontsize=9,
                    )

        if colorbar:
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    if title:
        figure.suptitle(title)
    return figure


def plot_coincidence_counts(
    counts: Mapping[str, float] | ArrayLike | object,
    *,
    labels: Sequence[str] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    ylabel: str = "Counts",
    color: str = "C0",
) -> Figure:
    """Plot lab coincidence counts as a simple bar chart."""
    _require_matplotlib()
    plot_labels, plot_values = _label_value_pairs(counts, labels=labels)

    if ax is None:
        from matplotlib import pyplot as plt

        figure, axis = plt.subplots(
            figsize=(max(5.0, 0.6 * len(plot_labels)), 4.0),
            constrained_layout=True,
        )
    else:
        axis = ax
        figure = ax.figure

    positions = np.arange(len(plot_labels), dtype=np.float64)
    axis.bar(positions, plot_values, color=color)
    axis.set_xticks(positions, plot_labels, rotation=45, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Measurement")
    if title:
        axis.set_title(title)
    return figure


def plot_bell_state_fidelities(
    fidelities: Mapping[str, float],
    *,
    ax: Axes | None = None,
    title: str | None = None,
    color: str = "C1",
    show_unity_line: bool = True,
) -> Figure:
    """Plot Bell-state fidelities in a canonical Phi/Psi ordering."""
    matplotlib = _require_matplotlib()
    plot_labels, plot_values = _ordered_mapping_values(fidelities, _BELL_CANONICAL_ORDER)

    if ax is None:
        figure = matplotlib.figure.Figure(figsize=(5.0, 4.0), constrained_layout=True)
        axis = figure.subplots()
    else:
        axis = ax
        figure = ax.figure

    positions = np.arange(len(plot_labels), dtype=np.float64)
    axis.bar(positions, plot_values, color=color)
    axis.set_xticks(positions, plot_labels)
    axis.set_ylim(0.0, max(1.0, 1.1 * float(np.max(plot_values, initial=1.0))))
    axis.set_ylabel("Fidelity")
    if show_unity_line:
        axis.axhline(1.0, color="0.3", linestyle="--", linewidth=1.0)
    if title:
        axis.set_title(title)
    return figure


def plot_chsh_correlators(
    correlators: Mapping[str, float],
    *,
    ax: Axes | None = None,
    title: str | None = None,
    color: str = "C2",
    show_s_value: bool = True,
) -> Figure:
    """Plot the four CHSH correlators with optional annotation of the resulting S value."""
    matplotlib = _require_matplotlib()
    plot_labels, plot_values = _ordered_mapping_values(correlators, _CHSH_CANONICAL_ORDER)

    if ax is None:
        figure = matplotlib.figure.Figure(figsize=(6.0, 4.0), constrained_layout=True)
        axis = figure.subplots()
    else:
        axis = ax
        figure = ax.figure

    positions = np.arange(len(plot_labels), dtype=np.float64)
    axis.bar(positions, plot_values, color=color)
    axis.set_xticks(positions, plot_labels)
    axis.set_ylim(-1.1, 1.1)
    axis.axhline(0.0, color="0.2", linewidth=1.0)
    axis.axhline(1.0, color="0.7", linestyle="--", linewidth=1.0)
    axis.axhline(-1.0, color="0.7", linestyle="--", linewidth=1.0)
    axis.set_ylabel("Correlator")

    if show_s_value and {"ab", "abp", "apb", "apbp"} <= set(correlators):
        s_value = (
            float(correlators["ab"])
            + float(correlators["abp"])
            + float(correlators["apb"])
            - float(correlators["apbp"])
        )
        axis.text(
            0.98,
            0.95,
            f"S = {s_value:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "0.8", "boxstyle": "round,pad=0.25"},
        )

    if title:
        axis.set_title(title)
    return figure
