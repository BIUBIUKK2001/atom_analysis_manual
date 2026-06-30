from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .intensity import compute_disk_intensity_table


_FIXED_PREFERRED_COLUMNS = [
    "point_id",
    "atom_id",
    "x_px",
    "y_px",
    "x_nm",
    "y_nm",
    "class_id",
    "class_name",
    "class_color",
    "roi_id",
    "roi_name",
    "roi_color",
    "scope_id",
    "coordinate_source",
    "source_table",
    "slice_index",
    "slice_label",
    "channel_name",
    "image_key",
    "stack_path",
    "disk_radius_px",
    "n_pixels",
    "disk_intensity_sum",
    "disk_intensity_mean",
    "is_edge",
    "status",
    "coordinate_mode",
]

_REFINED_DIAGNOSTIC_COLUMNS = [
    "x_seed_px",
    "y_seed_px",
    "x_input_px",
    "y_input_px",
    "x_fit_px",
    "y_fit_px",
    "quality_score",
    "center_shift_px",
    "attempted_center_shift_px",
    "center_shift_rejected",
    "position_source",
    "fit_success",
    "refinement_method",
    "refinement_path",
    "fit_residual",
    "amplitude",
    "local_background",
    "sigma_x",
    "sigma_y",
    "theta",
    "refinement_config_source",
    "nn_context_mode",
]


def _validate_stack(stack: np.ndarray) -> np.ndarray:
    arr = np.asarray(stack)
    if arr.ndim != 3:
        raise ValueError("stack must be a 3D array with shape (n_slices, height, width).")
    return arr


def _resolve_slice_indices_and_labels(
    n_slices: int,
    slice_indices: Sequence[int] | None,
    slice_labels: Sequence[object] | None,
    *,
    validate: bool = True,
) -> tuple[list[int], list[object]]:
    if slice_indices is None:
        indices = list(range(int(n_slices)))
    else:
        indices = [int(value) for value in slice_indices]
    if validate:
        invalid = [idx for idx in indices if idx < 0 or idx >= int(n_slices)]
        if invalid:
            raise ValueError(f"slice_indices contains invalid slice index/indices: {invalid}")
    if slice_labels is None:
        labels: list[object] = list(indices)
    else:
        labels = list(slice_labels)
        if len(labels) != len(indices):
            raise ValueError("slice_labels must have the same length as slice_indices.")
    return indices, labels


def _ordered_columns(data: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    for column in preferred:
        if column not in data.columns:
            data[column] = pd.NA
    return data[preferred + [column for column in data.columns if column not in preferred]]


def compute_stack_disk_intensity_table(
    points: pd.DataFrame,
    stack: np.ndarray,
    *,
    disk_radius_px: float,
    slice_indices: Sequence[int] | None = None,
    slice_labels: Sequence[object] | None = None,
    channel_name: str | None = None,
    image_key: str = "stack",
    coordinate_source: str | None = None,
    stack_path: str | None = None,
    x_offset_px: float = 0.0,
    y_offset_px: float = 0.0,
) -> pd.DataFrame:
    """Integrate fixed seed coordinates on every selected stack slice."""

    arr = _validate_stack(stack)
    indices, labels = _resolve_slice_indices_and_labels(arr.shape[0], slice_indices, slice_labels)
    source_points = points.copy() if points is not None else pd.DataFrame()
    tables: list[pd.DataFrame] = []

    for slice_index, slice_label in zip(indices, labels, strict=True):
        slice_points = source_points.copy()
        if not slice_points.empty:
            slice_points["x_px"] = pd.to_numeric(slice_points.get("x_px"), errors="coerce") + float(x_offset_px)
            slice_points["y_px"] = pd.to_numeric(slice_points.get("y_px"), errors="coerce") + float(y_offset_px)
        table = compute_disk_intensity_table(
            slice_points,
            arr[int(slice_index)],
            disk_radius_px=disk_radius_px,
            channel_name=channel_name,
            image_key=image_key,
            coordinate_source=coordinate_source,
        )
        table["slice_index"] = int(slice_index)
        table["slice_label"] = slice_label
        table["stack_path"] = stack_path
        table["coordinate_mode"] = "fixed"
        tables.append(table)

    result = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return _ordered_columns(result, _FIXED_PREFERRED_COLUMNS)


def _invalid_slice_record(row: pd.Series, *, disk_radius_px: float, channel_name: str | None, image_key: str, stack_path: str | None) -> dict[str, Any]:
    out = row.to_dict()
    out.update(
        {
            "channel_name": channel_name,
            "image_key": image_key,
            "stack_path": stack_path,
            "disk_radius_px": float(disk_radius_px),
            "n_pixels": 0,
            "disk_intensity_sum": np.nan,
            "disk_intensity_mean": np.nan,
            "is_edge": True,
            "status": "invalid_slice",
            "coordinate_mode": "slice_refined",
        }
    )
    return out


def compute_per_slice_disk_intensity_table(
    slice_points: pd.DataFrame,
    stack: np.ndarray,
    *,
    disk_radius_px: float,
    coordinate_columns: tuple[str, str] = ("x_px", "y_px"),
    slice_index_column: str = "slice_index",
    channel_name: str | None = None,
    image_key: str = "stack_refined",
    coordinate_source: str | None = None,
    stack_path: str | None = None,
) -> pd.DataFrame:
    """Integrate one row per atom/slice on its own stack slice."""

    arr = _validate_stack(stack)
    points = slice_points.copy() if slice_points is not None else pd.DataFrame()
    if points.empty:
        return _ordered_columns(pd.DataFrame(), _FIXED_PREFERRED_COLUMNS + _REFINED_DIAGNOSTIC_COLUMNS)
    if slice_index_column not in points.columns:
        raise ValueError(f"slice_points must contain {slice_index_column!r}.")
    x_column, y_column = coordinate_columns
    if x_column not in points.columns or y_column not in points.columns:
        raise ValueError(f"slice_points must contain coordinate columns {coordinate_columns!r}.")

    rows: list[pd.DataFrame | dict[str, Any]] = []
    slice_values = pd.to_numeric(points[slice_index_column], errors="coerce")
    valid_mask = slice_values.notna() & (slice_values >= 0) & (slice_values < arr.shape[0])

    for row_index, row in points.loc[~valid_mask].iterrows():
        rows.append(_invalid_slice_record(row, disk_radius_px=disk_radius_px, channel_name=channel_name, image_key=image_key, stack_path=stack_path))

    valid_points = points.loc[valid_mask].copy()
    valid_points["__slice_index_int"] = slice_values.loc[valid_mask].astype(int).to_numpy()
    for slice_index, group in valid_points.groupby("__slice_index_int", sort=True, dropna=False):
        table_points = group.drop(columns=["__slice_index_int"]).copy()
        if x_column != "x_px":
            table_points["x_px"] = pd.to_numeric(table_points[x_column], errors="coerce")
        if y_column != "y_px":
            table_points["y_px"] = pd.to_numeric(table_points[y_column], errors="coerce")
        table = compute_disk_intensity_table(
            table_points,
            arr[int(slice_index)],
            disk_radius_px=disk_radius_px,
            channel_name=channel_name,
            image_key=image_key,
            coordinate_source=coordinate_source,
        )
        table["coordinate_mode"] = "slice_refined"
        table["stack_path"] = stack_path
        rows.append(table)

    frames = [item for item in rows if isinstance(item, pd.DataFrame)]
    records = [item for item in rows if isinstance(item, dict)]
    result_parts: list[pd.DataFrame] = []
    if frames:
        result_parts.append(pd.concat(frames, ignore_index=True))
    if records:
        result_parts.append(pd.DataFrame(records))
    result = pd.concat(result_parts, ignore_index=True, sort=False) if result_parts else pd.DataFrame()
    if slice_index_column != "slice_index" and slice_index_column in result.columns:
        result["slice_index"] = result[slice_index_column]
    return _ordered_columns(result, _FIXED_PREFERRED_COLUMNS + _REFINED_DIAGNOSTIC_COLUMNS)


def summarize_stack_disk_intensity(
    stack_intensity_table: pd.DataFrame,
    *,
    group_by: tuple[str, ...] = ("slice_index", "slice_label", "class_id", "class_name", "channel_name"),
    metric: str = "disk_intensity_mean",
) -> pd.DataFrame:
    """Summarize stack disk intensity by slice and atom class/species."""

    columns = list(group_by) + ["metric", "count", "mean", "std", "sem", "median", "q25", "q75", "min", "max"]
    if stack_intensity_table is None or stack_intensity_table.empty:
        return pd.DataFrame(columns=columns)
    if metric not in stack_intensity_table.columns:
        raise ValueError(f"Metric column {metric!r} is not present in stack_intensity_table.")

    data = stack_intensity_table.copy()
    for column in group_by:
        if column not in data.columns:
            data[column] = pd.NA
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    summary = (
        data.groupby(list(group_by), dropna=False, sort=True)[metric]
        .agg(
            count="count",
            mean="mean",
            std="std",
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            min="min",
            max="max",
        )
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"].replace(0, np.nan))
    summary["metric"] = str(metric)
    return summary[columns]
