"""Input and output helpers for lab analysis data products."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
import re
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    import pandas as pd

PathType = str | PathLike[str]
ComplexArray = NDArray[np.complex128]

_MEASUREMENT_LABEL_CANDIDATES = {
    "label",
    "labels",
    "state",
    "states",
    "setting",
    "settings",
    "measurement",
    "measurements",
    "measurementlabel",
    "measurementlabels",
    "basis",
    "projector",
    "outcome",
}
_COUNT_COLUMN_CANDIDATES = {
    "count",
    "counts",
    "coincidence",
    "coincidences",
    "coincidencecount",
    "coincidencecounts",
    "value",
    "values",
    "n",
    "ns",
}
_CHSH_OUTCOME_ALIASES = {
    "++": "++",
    "+-": "+-",
    "-+": "-+",
    "--": "--",
    "pp": "++",
    "pm": "+-",
    "mp": "-+",
    "mm": "--",
}
_CHSH_SETTING_ALIASES = {
    "ab": "ab",
    "abp": "abp",
    "apb": "apb",
    "apbp": "apbp",
    "a,b": "ab",
    "a,bp": "abp",
    "ap,b": "apb",
    "ap,bp": "apbp",
}

__all__ = [
    "PathType",
    "counts_dict_from_table",
    "load_counts_table",
    "load_density_matrix",
    "save_counts_table",
    "save_density_matrix",
    "standardize_chsh_outcome_label",
    "standardize_chsh_setting_label",
    "standardize_counts_table",
    "standardize_polarization_label",
]


def _require_pandas():
    """Import pandas on demand so the rest of the package can import without it."""
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required for pec.io helpers. Install the project dependencies to use pec.io."
        ) from exc
    return pd


def _clean_text(value: object) -> str:
    """Normalize whitespace in a text-like cell or label."""
    return re.sub(r"\s+", " ", str(value).strip())


def _clean_column_key(value: object) -> str:
    """Convert a column label to a lowercase key for heuristic matching."""
    text = _clean_text(value).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def _looks_numeric(series: pd.Series) -> bool:
    """Return whether a series is numeric or can be parsed as numeric."""
    pd = _require_pandas()
    if pd.api.types.is_numeric_dtype(series):
        return True
    converted = pd.to_numeric(series, errors="coerce")
    return bool(converted.notna().any())


def _standardize_index_or_columns(labels: pd.Index) -> pd.Index:
    """Apply label standardization to a DataFrame index or column index."""
    pd = _require_pandas()
    normalized: list[object] = []
    for label in labels:
        if isinstance(label, str):
            normalized.append(_standardize_table_label(label))
        else:
            normalized.append(label)
    return pd.Index(normalized)


def _drop_empty_rows_and_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows and columns that are empty after spreadsheet import."""
    _require_pandas()
    cleaned = frame.copy()
    cleaned.columns = [_clean_text(column) for column in cleaned.columns]
    cleaned = cleaned.dropna(axis=0, how="all").dropna(axis=1, how="all")
    non_unnamed_columns = [
        column
        for column in cleaned.columns
        if not _clean_column_key(column).startswith("unnamed")
    ]
    return cleaned.loc[:, non_unnamed_columns]


def _detect_label_column(frame: pd.DataFrame) -> str | None:
    """Heuristically identify a measurement-label column in a tall count table."""
    for column in frame.columns:
        if _clean_column_key(column) in _MEASUREMENT_LABEL_CANDIDATES:
            return str(column)

    for column in frame.columns:
        if not _looks_numeric(frame[column]):
            return str(column)
    return None


def _detect_count_column(frame: pd.DataFrame, *, label_column: str | None) -> str | None:
    """Heuristically identify the count-value column in a tall count table."""
    for column in frame.columns:
        if column == label_column:
            continue
        if _clean_column_key(column) in _COUNT_COLUMN_CANDIDATES:
            return str(column)

    for column in frame.columns:
        if column == label_column:
            continue
        if _looks_numeric(frame[column]):
            return str(column)
    return None


def _looks_like_tall_count_table(frame: pd.DataFrame) -> bool:
    """Return whether a DataFrame looks like a label/count export."""
    if frame.shape[1] < 2:
        return False
    label_column = _detect_label_column(frame)
    count_column = _detect_count_column(frame, label_column=label_column)
    return label_column is not None and count_column is not None


def _parse_complex_cell(value: object) -> complex:
    """Parse a text cell into a complex number."""
    pd = _require_pandas()
    if isinstance(value, complex):
        return value
    if pd.isna(value):
        return 0.0 + 0.0j

    text = _clean_text(value).replace(" ", "")
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    return complex(text)


def _format_complex_cell(value: complex) -> str:
    """Format a complex number as a parseable text cell."""
    real = float(np.real(value))
    imag = float(np.imag(value))
    return f"{real:.18g}{imag:+.18g}j"


def _standardize_table_label(label: str) -> str:
    """Normalize a table label across polarization and CHSH-style aliases."""
    for standardizer in (
        standardize_chsh_setting_label,
        standardize_chsh_outcome_label,
        standardize_polarization_label,
    ):
        try:
            return standardizer(label)
        except ValueError:
            continue
    return _clean_text(label)


def standardize_polarization_label(label: str) -> str:
    """Standardize a polarization measurement label like H, HV, or P_H."""
    text = _clean_text(label).upper()
    text = text.replace("|", "").replace("<", "").replace(">", "")
    text = text.replace("(", "").replace(")", "")
    condensed = re.sub(r"[\s,;:/\\\-]+", "", text)

    probability_aliases = {
        "PH": "P_H",
        "PV": "P_V",
        "PD": "P_D",
        "PA": "P_A",
        "PR": "P_R",
        "PL": "P_L",
        "PMAX": "P_max",
        "PMIN": "P_min",
    }
    underscore_condensed = re.sub(r"[\s,;:/\\\-]+", "", text.replace("_", ""))
    if underscore_condensed in probability_aliases:
        return probability_aliases[underscore_condensed]

    if condensed and set(condensed) <= set("01HVDARL"):
        return condensed

    raise ValueError(f"Unsupported polarization label: {label!r}")


def standardize_chsh_outcome_label(label: str) -> str:
    """Standardize a CHSH coincidence-outcome label like ++ or pm."""
    text = _clean_text(label).lower()
    normalized = re.sub(r"[\s,;:_/\\]+", "", text)
    try:
        return _CHSH_OUTCOME_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported CHSH outcome label: {label!r}") from exc


def standardize_chsh_setting_label(label: str) -> str:
    """Standardize a CHSH analyzer-setting label like a,b' or apbp."""
    text = _clean_text(label).lower()
    normalized = text.replace("prime", "p")
    normalized = normalized.replace("'", "p")
    normalized = re.sub(r"[\s;:_/\\()]+", "", normalized)
    try:
        return _CHSH_SETTING_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported CHSH setting label: {label!r}") from exc


def standardize_counts_table(
    counts: pd.DataFrame,
    *,
    index_col: str | None = None,
    label_col: str | None = None,
    count_col: str | None = None,
) -> pd.DataFrame:
    """Normalize spreadsheet-like count tables into lab-friendly internal labels."""
    pd = _require_pandas()
    frame = _drop_empty_rows_and_columns(counts)

    if index_col is not None:
        if index_col not in frame.columns:
            raise ValueError(f"Index column {index_col!r} was not found in the table.")
        frame = frame.set_index(index_col)

    if label_col is not None:
        if label_col not in frame.columns:
            raise ValueError(f"Label column {label_col!r} was not found in the table.")
    elif _looks_like_tall_count_table(frame):
        label_col = _detect_label_column(frame)

    if label_col is not None:
        count_column = count_col or _detect_count_column(frame, label_column=label_col)
        if count_column is None:
            raise ValueError("Could not determine the count column for a tall count table.")

        labels = frame[label_col].map(_standardize_table_label)
        counts_series = pd.to_numeric(frame[count_column], errors="coerce")
        if counts_series.isna().any():
            raise ValueError("Count values must be numeric after table standardization.")

        standardized = counts_series.to_frame(name="counts")
        standardized.index = pd.Index(labels, name="label")
        return standardized

    standardized = frame.copy()
    if not isinstance(standardized.index, pd.RangeIndex):
        standardized.index = _standardize_index_or_columns(standardized.index)
    standardized.columns = _standardize_index_or_columns(pd.Index(standardized.columns))

    for column in standardized.columns:
        standardized[column] = pd.to_numeric(standardized[column], errors="ignore")
    return standardized


def counts_dict_from_table(
    counts: pd.DataFrame,
    *,
    index_col: str | None = None,
    label_col: str | None = None,
    count_col: str | None = None,
) -> dict[str, float]:
    """Convert a count table into a standardized label-to-count mapping."""
    standardized = standardize_counts_table(
        counts,
        index_col=index_col,
        label_col=label_col,
        count_col=count_col,
    )

    if list(standardized.columns) == ["counts"]:
        return {str(label): float(value) for label, value in standardized["counts"].items()}

    if standardized.shape[1] == 1:
        column = standardized.columns[0]
        return {str(label): float(value) for label, value in standardized[column].items()}

    raise ValueError("counts_dict_from_table requires a tall count table with one count column.")


def load_counts_table(
    path: PathType,
    *,
    index_col: str | None = None,
    label_col: str | None = None,
    count_col: str | None = None,
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    """Load and standardize a lab count table from CSV, TSV, or spreadsheet export."""
    pd = _require_pandas()
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        frame = pd.read_csv(file_path)
    elif suffix in {".tsv", ".txt"}:
        frame = pd.read_csv(file_path, sep="\t")
    elif suffix in {".xls", ".xlsx"}:
        frame = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        raise ValueError(f"Unsupported count-table extension: {suffix!r}")

    return standardize_counts_table(
        frame,
        index_col=index_col,
        label_col=label_col,
        count_col=count_col,
    )


def save_counts_table(counts: pd.DataFrame, path: PathType) -> None:
    """Write a standardized count table to CSV, TSV, or spreadsheet format."""
    pd = _require_pandas()
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    include_index = bool(
        counts.index.name is not None
        or not isinstance(counts.index, pd.RangeIndex)
    )

    if suffix == ".csv":
        counts.to_csv(file_path, index=include_index)
        return
    if suffix in {".tsv", ".txt"}:
        counts.to_csv(file_path, sep="\t", index=include_index)
        return
    if suffix in {".xls", ".xlsx"}:
        counts.to_excel(file_path, index=include_index)
        return
    raise ValueError(f"Unsupported count-table extension: {suffix!r}")


def load_density_matrix(path: PathType) -> ComplexArray:
    """Load a density matrix from `.npy`, `.csv`, `.tsv`, or `.txt` storage."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".npy":
        return np.asarray(np.load(file_path), dtype=np.complex128)
    pd = _require_pandas()
    if suffix == ".csv":
        frame = pd.read_csv(file_path, header=None)
    elif suffix in {".tsv", ".txt"}:
        frame = pd.read_csv(file_path, header=None, sep="\t")
    else:
        raise ValueError(f"Unsupported density-matrix extension: {suffix!r}")

    matrix = frame.applymap(_parse_complex_cell).to_numpy(dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Density matrices must load as square 2D arrays.")
    return matrix


def save_density_matrix(rho: ArrayLike, path: PathType) -> None:
    """Write a density matrix to `.npy`, `.csv`, `.tsv`, or `.txt` storage."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    matrix = np.asarray(rho, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Density matrices must be square 2D arrays.")

    if suffix == ".npy":
        np.save(file_path, matrix)
        return

    pd = _require_pandas()
    text_frame = pd.DataFrame(matrix).applymap(_format_complex_cell)
    if suffix == ".csv":
        text_frame.to_csv(file_path, index=False, header=False)
        return
    if suffix in {".tsv", ".txt"}:
        text_frame.to_csv(file_path, sep="\t", index=False, header=False)
        return
    raise ValueError(f"Unsupported density-matrix extension: {suffix!r}")
