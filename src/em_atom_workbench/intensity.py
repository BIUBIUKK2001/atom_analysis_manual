from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .simple_quant import AnalysisROI, assign_points_to_rois


@dataclass(frozen=True)
class DiskIntensityConfig:
    image_channel: str | None = None
    image_key: str = "raw"
    disk_radius_px: float = 2.0
    coordinate_source: str = "refined"
    use_keep_only: bool = True


_COORDINATE_TABLES = {
    "candidate": "candidate_points",
    "refined": "refined_points",
    "curated": "curated_points",
}

_UNIT_TO_NM = {
    "nm": 1.0,
    "nanometer": 1.0,
    "nanometers": 1.0,
    "a": 0.1,
    "angstrom": 0.1,
    "angstroms": 0.1,
    "å": 0.1,
    "ångström": 0.1,
    "ångströms": 0.1,
    "pm": 0.001,
    "picometer": 0.001,
    "picometers": 0.001,
    "um": 1000.0,
    "µm": 1000.0,
    "μm": 1000.0,
    "micrometer": 1000.0,
    "micrometers": 1000.0,
}

_POINT_COLUMNS = [
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
    "keep",
    "coordinate_source",
    "source_table",
]


def _pixel_to_nm_from_session(session: Any) -> float | None:
    calibration = getattr(session, "pixel_calibration", None)
    size = getattr(calibration, "size", None)
    if size is None:
        return None
    size = float(size)
    if not np.isfinite(size) or size <= 0.0:
        return None
    unit_key = str(getattr(calibration, "unit", "px") or "px").strip().lower()
    if unit_key == "px":
        return None
    factor = _UNIT_TO_NM.get(unit_key)
    if factor is None:
        return None
    return size * factor


def _normalize_coordinate_source(coordinate_source: str) -> str:
    source = str(coordinate_source).strip().lower()
    if source not in _COORDINATE_TABLES:
        raise ValueError("coordinate_source must be one of 'candidate', 'refined', or 'curated'.")
    return source


def _selected_coordinate_table(session: Any, coordinate_source: str) -> tuple[pd.DataFrame, str]:
    source = _normalize_coordinate_source(coordinate_source)
    table_name = _COORDINATE_TABLES[source]
    table = getattr(session, table_name, pd.DataFrame())
    if table is None or table.empty:
        if source == "candidate":
            message = "candidate_points is empty; run Notebook01 detection/candidate review first."
        elif source == "refined":
            message = "refined_points is empty; run Notebook01 refinement first or choose COORDINATE_SOURCE='candidate'."
        else:
            message = "curated_points is empty; run Notebook01 final curation first or choose COORDINATE_SOURCE='refined'."
        raise ValueError(message)
    return table.copy().reset_index(drop=True), table_name


def _as_filter_set(values: Any, *, cast=str) -> set | None:
    if values is None:
        return None
    if isinstance(values, str):
        return {cast(values)}
    return {cast(value) for value in values}


def _default_class_name(class_id: Any) -> str:
    return f"class_{int(class_id)}" if pd.notna(class_id) else "class_unknown"


def prepare_disk_intensity_points(
    session: Any,
    *,
    coordinate_source: str = "refined",
    use_keep_only: bool = True,
    class_filter=None,
    class_id_filter=None,
    rois: list[AnalysisROI] | tuple[AnalysisROI, ...] | None = None,
) -> pd.DataFrame:
    """Prepare atom coordinates for fixed-radius disk intensity integration."""

    source = _normalize_coordinate_source(coordinate_source)
    table, table_name = _selected_coordinate_table(session, source)
    if "x_px" not in table.columns or "y_px" not in table.columns:
        raise ValueError(f"{table_name} must contain x_px and y_px columns.")

    has_class_id = "class_id" in table.columns
    has_class_name = "class_name" in table.columns
    if class_id_filter is not None and not has_class_id:
        raise ValueError(f"{table_name} does not contain class_id; class_id_filter cannot be applied.")
    if class_filter is not None and not has_class_name:
        raise ValueError(f"{table_name} does not contain class_name; class_filter cannot be applied.")

    warnings_out: list[str] = []
    if source == "candidate" and not has_class_id and not has_class_name:
        message = "candidate_points do not contain class labels; class-based filtering/plotting may be unavailable."
        warnings_out.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    points = pd.DataFrame(index=table.index)
    if "atom_id" in table.columns:
        atom_ids = pd.to_numeric(table["atom_id"], errors="coerce")
        fallback_ids = pd.Series(np.arange(len(table), dtype=int), index=table.index)
        atom_ids = atom_ids.where(atom_ids.notna(), fallback_ids)
        points["atom_id"] = atom_ids.astype(int)
    else:
        points["atom_id"] = np.arange(len(table), dtype=int)
    if "point_id" in table.columns:
        points["point_id"] = table["point_id"].astype(str)
    else:
        points["point_id"] = [f"atom:{int(value)}" for value in points["atom_id"]]
    points["x_px"] = pd.to_numeric(table["x_px"], errors="coerce").astype(float)
    points["y_px"] = pd.to_numeric(table["y_px"], errors="coerce").astype(float)

    if has_class_id:
        points["class_id"] = pd.to_numeric(table["class_id"], errors="coerce")
    else:
        points["class_id"] = pd.Series([pd.NA] * len(points), index=points.index, dtype="object")
    if has_class_name:
        points["class_name"] = table["class_name"].astype("object")
    else:
        points["class_name"] = [_default_class_name(value) for value in points["class_id"]]
    missing_name = points["class_name"].isna()
    if missing_name.any():
        points.loc[missing_name, "class_name"] = [
            _default_class_name(value) for value in points.loc[missing_name, "class_id"]
        ]

    points["class_color"] = table["class_color"].astype("object") if "class_color" in table.columns else pd.NA
    has_keep = "keep" in table.columns
    points["keep"] = table["keep"].fillna(True).astype(bool) if has_keep else True
    points["source_type"] = "atom"
    points["point_set"] = "atoms"
    points["coordinate_source"] = source
    points["source_table"] = table_name

    pixel_to_nm = _pixel_to_nm_from_session(session)
    if "x_nm" in table.columns:
        points["x_nm"] = pd.to_numeric(table["x_nm"], errors="coerce")
    elif pixel_to_nm is not None:
        points["x_nm"] = points["x_px"] * pixel_to_nm
    else:
        points["x_nm"] = np.nan
    if "y_nm" in table.columns:
        points["y_nm"] = pd.to_numeric(table["y_nm"], errors="coerce")
    elif pixel_to_nm is not None:
        points["y_nm"] = points["y_px"] * pixel_to_nm
    else:
        points["y_nm"] = np.nan

    if use_keep_only and has_keep:
        points = points.loc[points["keep"] == True].copy()  # noqa: E712
    allowed_names = _as_filter_set(class_filter, cast=str)
    if allowed_names is not None:
        points = points.loc[points["class_name"].astype(str).isin(allowed_names)].copy()
    allowed_ids = _as_filter_set(class_id_filter, cast=int)
    if allowed_ids is not None:
        points = points.loc[points["class_id"].isin(allowed_ids)].copy()

    points = assign_points_to_rois(points.reset_index(drop=True), rois)
    if "scope_id" not in points.columns:
        points["scope_id"] = (
            points["roi_id"].astype(str) + ":" + points["point_set"].astype(str) + ":" + points["source_table"].astype(str)
        )
    for column in _POINT_COLUMNS:
        if column not in points.columns:
            points[column] = pd.NA
    points.attrs["pixel_to_nm"] = pixel_to_nm
    points.attrs["coordinate_source"] = source
    points.attrs["source_table"] = table_name
    points.attrs["warnings"] = warnings_out
    return points[_POINT_COLUMNS + [column for column in points.columns if column not in _POINT_COLUMNS]].reset_index(drop=True)


def build_disk_offsets(radius_px: float) -> np.ndarray:
    radius = float(radius_px)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_px must be a positive finite number.")
    limit = int(np.ceil(radius))
    offsets: list[tuple[int, int]] = []
    radius_sq = radius * radius
    for dy in range(-limit, limit + 1):
        for dx in range(-limit, limit + 1):
            if float(dx * dx + dy * dy) <= radius_sq:
                offsets.append((dx, dy))
    return np.asarray(offsets, dtype=int).reshape(-1, 2)


def compute_disk_intensity_table(
    points: pd.DataFrame,
    image: np.ndarray,
    *,
    disk_radius_px: float,
    channel_name: str | None = None,
    image_key: str = "raw",
    coordinate_source: str | None = None,
) -> pd.DataFrame:
    """Integrate image intensity in a fixed-radius disk around each point."""

    radius = float(disk_radius_px)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("disk_radius_px must be a positive finite number.")
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("image must be a 2D array. Select a single IMAGE_CHANNEL before integration.")
    if points is None:
        points = pd.DataFrame()

    height, width = arr.shape
    rows: list[dict[str, Any]] = []
    for _, row in points.iterrows():
        out = row.to_dict()
        source = coordinate_source or row.get("coordinate_source", None)
        out["channel_name"] = channel_name
        out["image_key"] = image_key
        out["coordinate_source"] = None if source is None else str(source)
        out["disk_radius_px"] = radius

        x = pd.to_numeric(pd.Series([row.get("x_px", np.nan)]), errors="coerce").iloc[0]
        y = pd.to_numeric(pd.Series([row.get("y_px", np.nan)]), errors="coerce").iloc[0]
        if not np.isfinite(float(x)) or not np.isfinite(float(y)):
            out.update(
                {
                    "n_pixels": 0,
                    "disk_intensity_sum": np.nan,
                    "disk_intensity_mean": np.nan,
                    "is_edge": True,
                    "status": "no_pixels",
                }
            )
            rows.append(out)
            continue

        x = float(x)
        y = float(y)
        x_min = int(np.floor(x - radius))
        x_max = int(np.ceil(x + radius))
        y_min = int(np.floor(y - radius))
        y_max = int(np.ceil(y + radius))
        is_edge = x_min < 0 or y_min < 0 or x_max >= width or y_max >= height
        clipped_x_min = max(0, x_min)
        clipped_x_max = min(width - 1, x_max)
        clipped_y_min = max(0, y_min)
        clipped_y_max = min(height - 1, y_max)
        if clipped_x_max < clipped_x_min or clipped_y_max < clipped_y_min:
            values = np.asarray([], dtype=arr.dtype)
        else:
            cols = np.arange(clipped_x_min, clipped_x_max + 1)
            rows_px = np.arange(clipped_y_min, clipped_y_max + 1)
            cc, rr = np.meshgrid(cols, rows_px)
            mask = (cc.astype(float) - x) ** 2 + (rr.astype(float) - y) ** 2 <= radius * radius
            values = arr[rr[mask], cc[mask]]

        n_pixels = int(values.size)
        if n_pixels == 0:
            total = np.nan
            mean = np.nan
            status = "no_pixels"
        else:
            total = float(np.sum(values))
            mean = float(total / n_pixels)
            status = "edge_clipped" if is_edge else "ok"
        out.update(
            {
                "n_pixels": n_pixels,
                "disk_intensity_sum": total,
                "disk_intensity_mean": mean,
                "is_edge": bool(is_edge),
                "status": status,
            }
        )
        rows.append(out)

    result = pd.DataFrame(rows)
    preferred = [
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
        "channel_name",
        "image_key",
        "coordinate_source",
        "source_table",
        "disk_radius_px",
        "n_pixels",
        "disk_intensity_sum",
        "disk_intensity_mean",
        "is_edge",
        "status",
    ]
    for column in preferred:
        if column not in result.columns:
            result[column] = pd.NA
    return result[preferred + [column for column in result.columns if column not in preferred]]


def summarize_disk_intensity(
    intensity_table: pd.DataFrame,
    *,
    group_by: tuple[str, ...] = ("coordinate_source", "class_id", "class_name", "channel_name"),
    metric: str = "disk_intensity_sum",
) -> pd.DataFrame:
    columns = list(group_by) + ["count", "mean", "std", "median", "q25", "q75", "min", "max"]
    if intensity_table is None or intensity_table.empty:
        return pd.DataFrame(columns=columns)
    if metric not in intensity_table.columns:
        raise ValueError(f"Metric column {metric!r} is not present in intensity_table.")

    data = intensity_table.copy()
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
    return summary[columns]
