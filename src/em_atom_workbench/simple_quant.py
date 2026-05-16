from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class DirectionSpec:
    name: str
    vector_px: tuple[float, float] | None = None
    from_atom_ids: tuple[int, int] | None = None
    from_xy_px: tuple[tuple[float, float], tuple[float, float]] | None = None
    snap_to_nearest_atoms: bool = False


@dataclass(frozen=True)
class DirectionalSpacingTask:
    name: str
    direction: str
    source_classes: tuple[str, ...] | None = None
    target_classes: tuple[str, ...] | None = None
    source_class_ids: tuple[int, ...] | None = None
    target_class_ids: tuple[int, ...] | None = None
    min_parallel_distance_px: float = 1.0
    max_parallel_distance_px: float | None = None
    perpendicular_tolerance_px: float = 3.0
    angle_tolerance_deg: float = 15.0
    unique_pairs: bool = False


@dataclass(frozen=True)
class PairDistanceTask:
    name: str
    source_classes: tuple[str, ...] | None = None
    target_classes: tuple[str, ...] | None = None
    source_class_ids: tuple[int, ...] | None = None
    target_class_ids: tuple[int, ...] | None = None
    explicit_atom_pairs: tuple[tuple[int, int], ...] | None = None
    mode: str = "nearest"
    max_distance_px: float | None = None
    direction: str | None = None
    angle_tolerance_deg: float | None = None
    unique_pairs: bool = True


@dataclass(frozen=True)
class LineGroupingTask:
    name: str
    direction: str
    classes: tuple[str, ...] | None = None
    class_ids: tuple[int, ...] | None = None
    group_axis: str = "t"
    sort_axis: str | None = None
    line_tolerance_px: float = 3.0
    min_atoms_per_line: int = 4
    max_in_line_gap_px: float | None = None
    line_width_method: str = "p95_p5"


@dataclass(frozen=True)
class AnalysisROI:
    roi_id: str
    roi_name: str | None = None
    polygon_xy_px: tuple[tuple[float, float], ...] | None = None
    color: str = "#ff9f1c"
    class_ids: tuple[int, ...] | None = None
    class_names: tuple[str, ...] | None = None
    enabled: bool = True


@dataclass(frozen=True)
class BasisVectorSpec:
    name: str
    vector_px: tuple[float, float] | None = None
    from_point_ids: tuple[str, str] | None = None
    from_atom_ids: tuple[int, int] | None = None
    from_xy_px: tuple[tuple[float, float], tuple[float, float]] | None = None
    snap_to_nearest_points: bool = True
    use_length_as_period: bool = True
    period_px: float | None = None
    roi_id: str | None = None
    basis_role: str | None = None


@dataclass(frozen=True)
class NearestForwardTask:
    name: str
    basis: str
    point_set: str = "atoms"
    roi_ids: tuple[str, ...] | None = None
    source_class_ids: tuple[int, ...] | None = None
    target_class_ids: tuple[int, ...] | None = None
    source_class_names: tuple[str, ...] | None = None
    target_class_names: tuple[str, ...] | None = None
    min_parallel_distance_px: float = 1.0
    max_parallel_distance_px: float | None = None
    perpendicular_tolerance_px: float = 3.0
    angle_tolerance_deg: float = 15.0
    unique_pairs: bool = False


@dataclass(frozen=True)
class PairSegmentTask:
    name: str
    point_set: str = "atoms"
    roi_ids: tuple[str, ...] | None = None
    source_class_ids: tuple[int, ...] | None = None
    target_class_ids: tuple[int, ...] | None = None
    source_class_names: tuple[str, ...] | None = None
    target_class_names: tuple[str, ...] | None = None
    explicit_point_pairs: tuple[tuple[str, str], ...] | None = None
    explicit_atom_pairs: tuple[tuple[int, int], ...] | None = None
    mode: str = "nearest"
    max_distance_px: float | None = None
    basis: str | None = None
    angle_tolerance_deg: float | None = None
    unique_pairs: bool = True
    create_pair_centers: bool = False
    pair_center_class_name: str | None = None


@dataclass(frozen=True)
class PeriodicVectorTask:
    name: str
    basis: str
    point_set: str = "atoms"
    roi_ids: tuple[str, ...] | None = None
    class_ids: tuple[int, ...] | None = None
    class_names: tuple[str, ...] | None = None
    match_radius_fraction: float = 0.30
    match_radius_px: float | None = None
    start_policy: str = "edge_min_projection"
    one_to_one: bool = True
    max_steps_per_chain: int | None = None


@dataclass(frozen=True)
class LineGuideTask:
    name: str
    basis: str
    point_set: str = "atoms"
    roi_ids: tuple[str, ...] | None = None
    class_ids: tuple[int, ...] | None = None
    class_names: tuple[str, ...] | None = None
    group_axis: str = "t"
    line_tolerance_px: float = 3.0
    min_points_per_line: int = 4
    generate_consecutive_segments: bool = False
    max_in_line_gap_px: float | None = None


_SOURCE_TABLES = {
    "curated": "curated_points",
    "refined": "refined_points",
    "candidate": "candidate_points",
}

_UNIT_TO_NM = {
    "nm": 1.0,
    "nanometer": 1.0,
    "nanometers": 1.0,
    "a": 0.1,
    "å": 0.1,
    "angstrom": 0.1,
    "angstroms": 0.1,
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


def _pixel_to_nm_from_calibration(calibration: Any) -> float | None:
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


def _pixel_to_nm_from_points(points: pd.DataFrame) -> float | None:
    attr_value = getattr(points, "attrs", {}).get("pixel_to_nm")
    if attr_value is not None and np.isfinite(float(attr_value)) and float(attr_value) > 0.0:
        return float(attr_value)
    ratios: list[np.ndarray] = []
    for px_col, nm_col in (("x_px", "x_nm"), ("y_px", "y_nm")):
        if px_col not in points.columns or nm_col not in points.columns:
            continue
        px = pd.to_numeric(points[px_col], errors="coerce").to_numpy(dtype=float)
        nm = pd.to_numeric(points[nm_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(px) & np.isfinite(nm) & (np.abs(px) > 1e-12)
        if np.any(mask):
            ratios.append(nm[mask] / px[mask])
    if not ratios:
        return None
    value = float(np.nanmedian(np.concatenate(ratios)))
    if not np.isfinite(value) or value <= 0.0:
        return None
    return value


def _selected_source_table(session: Any, source_table: str) -> tuple[pd.DataFrame, str]:
    source_key = str(source_table).lower()
    if source_key not in _SOURCE_TABLES:
        raise ValueError("source_table must be one of 'curated', 'refined', or 'candidate'.")
    if source_key == "curated":
        table = session.get_atom_table(preferred="curated")
        actual = "curated"
        if getattr(session, "curated_points", pd.DataFrame()).empty:
            actual = "refined" if not getattr(session, "refined_points", pd.DataFrame()).empty else "candidate"
    else:
        actual = source_key
        table = getattr(session, _SOURCE_TABLES[source_key], pd.DataFrame())
    if table is None or table.empty:
        raise ValueError(
            f"No atom coordinates are available in source_table={source_table!r}. "
            "Run the previous notebook and save an active session first."
        )
    return table.copy().reset_index(drop=True), actual


def prepare_quant_points(
    session: Any,
    source_table: str = "curated",
    use_keep_only: bool = True,
    class_filter: tuple[str, ...] | list[str] | set[str] | None = None,
    class_id_filter: tuple[int, ...] | list[int] | set[int] | None = None,
    roi: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    table, actual_source = _selected_source_table(session, source_table)
    if "x_px" not in table.columns or "y_px" not in table.columns:
        raise ValueError("Atom coordinate table must contain x_px and y_px columns.")

    points = pd.DataFrame()
    if "atom_id" in table.columns:
        atom_ids = pd.to_numeric(table["atom_id"], errors="coerce")
        atom_ids = atom_ids.where(atom_ids.notna(), pd.Series(np.arange(len(table), dtype=int)))
        points["atom_id"] = atom_ids.astype(int)
    else:
        points["atom_id"] = np.arange(len(table), dtype=int)
    points["x_px"] = pd.to_numeric(table["x_px"], errors="coerce").astype(float)
    points["y_px"] = pd.to_numeric(table["y_px"], errors="coerce").astype(float)
    if "class_id" in table.columns:
        points["class_id"] = pd.to_numeric(table["class_id"], errors="coerce")
    else:
        points["class_id"] = np.nan
    if "class_name" in table.columns:
        points["class_name"] = table["class_name"].astype("object")
    else:
        points["class_name"] = [
            f"class_{int(value)}" if pd.notna(value) else "class_unknown"
            for value in points["class_id"].to_numpy()
        ]
    missing_name = points["class_name"].isna()
    if missing_name.any():
        points.loc[missing_name, "class_name"] = [
            f"class_{int(value)}" if pd.notna(value) else "class_unknown"
            for value in points.loc[missing_name, "class_id"].to_numpy()
        ]
    points["class_color"] = table["class_color"].astype("object") if "class_color" in table.columns else pd.NA
    points["column_role"] = table["column_role"].astype("object") if "column_role" in table.columns else pd.NA
    points["keep"] = table["keep"].astype(bool) if "keep" in table.columns else True
    points["quality_score"] = (
        pd.to_numeric(table["quality_score"], errors="coerce") if "quality_score" in table.columns else np.nan
    )
    points["source_table"] = actual_source

    pixel_to_nm = _pixel_to_nm_from_calibration(getattr(session, "pixel_calibration", None))
    if pixel_to_nm is None:
        points["x_nm"] = np.nan
        points["y_nm"] = np.nan
    else:
        points["x_nm"] = points["x_px"] * pixel_to_nm
        points["y_nm"] = points["y_px"] * pixel_to_nm
    points.attrs["pixel_to_nm"] = pixel_to_nm

    if use_keep_only and "keep" in points.columns:
        points = points.loc[points["keep"] == True].copy()  # noqa: E712
    if class_filter is not None:
        allowed = {str(value) for value in class_filter}
        points = points.loc[points["class_name"].astype(str).isin(allowed)].copy()
    if class_id_filter is not None:
        allowed_ids = {int(value) for value in class_id_filter}
        points = points.loc[points["class_id"].isin(allowed_ids)].copy()
    if roi is not None:
        x_min, x_max, y_min, y_max = [float(value) for value in roi]
        points = points.loc[
            (points["x_px"] >= x_min)
            & (points["x_px"] <= x_max)
            & (points["y_px"] >= y_min)
            & (points["y_px"] <= y_max)
        ].copy()
    points.attrs["pixel_to_nm"] = pixel_to_nm
    return points.reset_index(drop=True)


def _default_rois(rois: list[AnalysisROI] | tuple[AnalysisROI, ...] | None) -> list[AnalysisROI]:
    enabled = [roi for roi in (rois or []) if bool(roi.enabled)]
    if enabled:
        return enabled
    return [AnalysisROI(roi_id="global", roi_name="global", polygon_xy_px=None, color="#ff9f1c")]


def _default_class_name(class_id: Any) -> str:
    return f"class_{int(class_id)}" if pd.notna(class_id) else "class_unknown"


def _point_id(row: pd.Series) -> str:
    value = row.get("point_id", pd.NA)
    if pd.notna(value):
        return str(value)
    atom_id = row.get("atom_id", pd.NA)
    return f"atom:{int(atom_id)}" if pd.notna(atom_id) else ""


def _atom_id_or_na(row: pd.Series) -> Any:
    value = row.get("atom_id", pd.NA)
    return int(value) if pd.notna(value) else pd.NA


def _pixel_to_nm_from_row_pair(source: pd.Series, target: pd.Series) -> float:
    keys = ("x_nm", "y_nm")
    if not all(key in source.index and key in target.index for key in keys):
        return np.nan
    values = [source["x_nm"], source["y_nm"], target["x_nm"], target["y_nm"]]
    if not np.isfinite(pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)).all():
        return np.nan
    return float(np.hypot(float(target["x_nm"]) - float(source["x_nm"]), float(target["y_nm"]) - float(source["y_nm"])))


def points_in_roi(points: pd.DataFrame, roi: AnalysisROI) -> pd.DataFrame:
    if points is None or points.empty:
        return pd.DataFrame(columns=[] if points is None else points.columns)
    if roi.polygon_xy_px is None:
        return points.copy()
    polygon = np.asarray(roi.polygon_xy_px, dtype=float)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise ValueError(f"ROI {roi.roi_id!r} polygon_xy_px must contain at least three (x, y) points.")
    if not np.allclose(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0]])
    path = MplPath(polygon)
    coords = points[["x_px", "y_px"]].to_numpy(dtype=float)
    mask = path.contains_points(coords, radius=1e-9)
    return points.loc[mask].copy()


def assign_points_to_rois(points: pd.DataFrame, rois: list[AnalysisROI] | tuple[AnalysisROI, ...] | None) -> pd.DataFrame:
    points = points.copy()
    if "x_px" not in points.columns or "y_px" not in points.columns:
        raise ValueError("points must contain x_px and y_px columns.")
    if "point_id" not in points.columns:
        if "atom_id" in points.columns:
            points["point_id"] = [f"atom:{int(value)}" if pd.notna(value) else f"point:{index}" for index, value in enumerate(points["atom_id"])]
        else:
            points["point_id"] = [f"point:{index}" for index in range(len(points))]
    if "class_id" not in points.columns:
        points["class_id"] = np.nan
    if "class_name" not in points.columns:
        points["class_name"] = [_default_class_name(value) for value in points["class_id"]]
    rows: list[pd.DataFrame] = []
    for roi in _default_rois(rois):
        roi_points = points_in_roi(points, roi)
        if roi.class_ids is not None and "class_id" in roi_points.columns:
            roi_points = roi_points.loc[roi_points["class_id"].isin({int(value) for value in roi.class_ids})].copy()
        if roi.class_names is not None and "class_name" in roi_points.columns:
            roi_points = roi_points.loc[
                roi_points["class_name"].astype(str).isin({str(value) for value in roi.class_names})
            ].copy()
        if roi_points.empty:
            continue
        roi_points = roi_points.copy()
        roi_points["roi_id"] = str(roi.roi_id)
        roi_points["roi_name"] = str(roi.roi_name or roi.roi_id)
        roi_points["roi_color"] = str(roi.color)
        rows.append(roi_points)
    if not rows:
        result = points.iloc[0:0].copy()
        for column in ("roi_id", "roi_name", "roi_color"):
            if column not in result.columns:
                result[column] = pd.Series(dtype="object")
        return result.reset_index(drop=True)
    return pd.concat(rows, ignore_index=True)


def prepare_analysis_points(
    session: Any,
    source_table: str = "curated",
    use_keep_only: bool = True,
    class_filter: tuple[str, ...] | list[str] | set[str] | None = None,
    class_id_filter: tuple[int, ...] | list[int] | set[int] | None = None,
    rois: list[AnalysisROI] | tuple[AnalysisROI, ...] | None = None,
) -> pd.DataFrame:
    table, actual_source = _selected_source_table(session, source_table)
    if "x_px" not in table.columns or "y_px" not in table.columns:
        raise ValueError("Atom coordinate table must contain x_px and y_px columns.")

    points = pd.DataFrame(index=table.index)
    if "atom_id" in table.columns:
        atom_ids = pd.to_numeric(table["atom_id"], errors="coerce")
        fallback_ids = pd.Series(np.arange(len(table), dtype=int), index=table.index)
        atom_ids = atom_ids.where(atom_ids.notna(), fallback_ids)
        points["atom_id"] = atom_ids.astype(int)
    else:
        points["atom_id"] = np.arange(len(table), dtype=int)
    points["point_id"] = [f"atom:{int(value)}" for value in points["atom_id"]]
    points["source_type"] = "atom"
    points["point_set"] = "atoms"
    for column in (
        "parent_source_point_id",
        "parent_target_point_id",
        "parent_source_atom_id",
        "parent_target_atom_id",
    ):
        points[column] = pd.NA
    points["x_px"] = pd.to_numeric(table["x_px"], errors="coerce").astype(float)
    points["y_px"] = pd.to_numeric(table["y_px"], errors="coerce").astype(float)
    points["class_id"] = pd.to_numeric(table["class_id"], errors="coerce") if "class_id" in table.columns else np.nan
    if "class_name" in table.columns:
        points["class_name"] = table["class_name"].astype("object")
    else:
        points["class_name"] = [_default_class_name(value) for value in points["class_id"]]
    missing = points["class_name"].isna()
    if missing.any():
        points.loc[missing, "class_name"] = [_default_class_name(value) for value in points.loc[missing, "class_id"]]
    points["class_color"] = table["class_color"].astype("object") if "class_color" in table.columns else pd.NA
    points["column_role"] = table["column_role"].astype("object") if "column_role" in table.columns else pd.NA
    points["keep"] = table["keep"].astype(bool) if "keep" in table.columns else True
    points["quality_score"] = (
        pd.to_numeric(table["quality_score"], errors="coerce") if "quality_score" in table.columns else np.nan
    )
    points["source_table"] = actual_source

    pixel_to_nm = _pixel_to_nm_from_calibration(getattr(session, "pixel_calibration", None))
    if pixel_to_nm is None:
        points["x_nm"] = np.nan
        points["y_nm"] = np.nan
    else:
        points["x_nm"] = points["x_px"] * pixel_to_nm
        points["y_nm"] = points["y_px"] * pixel_to_nm
    if use_keep_only and "keep" in points.columns:
        points = points.loc[points["keep"] == True].copy()  # noqa: E712
    if class_filter is not None:
        points = points.loc[points["class_name"].astype(str).isin({str(value) for value in class_filter})].copy()
    if class_id_filter is not None:
        points = points.loc[points["class_id"].isin({int(value) for value in class_id_filter})].copy()

    points = assign_points_to_rois(points.reset_index(drop=True), rois)
    if "scope_id" not in points.columns:
        points["scope_id"] = (
            points["roi_id"].astype(str) + ":" + points["point_set"].astype(str) + ":" + points["source_table"].astype(str)
        )
    points.attrs["pixel_to_nm"] = pixel_to_nm
    return points.reset_index(drop=True)


def _atom_row(points: pd.DataFrame, atom_id: int) -> pd.Series:
    matches = points.loc[points["atom_id"].astype(int) == int(atom_id)]
    if matches.empty:
        raise ValueError(f"atom_id {atom_id!r} was not found in quant_points.")
    return matches.iloc[0]


def _nearest_atom(points: pd.DataFrame, xy: tuple[float, float]) -> pd.Series:
    if points.empty:
        raise ValueError("Cannot snap direction points because quant_points is empty.")
    tree = cKDTree(points[["x_px", "y_px"]].to_numpy(dtype=float))
    _, index = tree.query(np.asarray(xy, dtype=float), k=1)
    return points.iloc[int(index)]


def _nearest_point(points: pd.DataFrame, xy: tuple[float, float]) -> pd.Series:
    if points.empty:
        raise ValueError("Cannot snap basis vector points because analysis_points is empty.")
    tree = cKDTree(points[["x_px", "y_px"]].to_numpy(dtype=float))
    _, index = tree.query(np.asarray(xy, dtype=float), k=1)
    return points.iloc[int(index)]


def _point_row(points: pd.DataFrame, point_id: str) -> pd.Series:
    matches = points.loc[points["point_id"].astype(str) == str(point_id)]
    if matches.empty:
        raise ValueError(f"point_id {point_id!r} was not found in analysis_points.")
    return matches.iloc[0]


def _basis_row(basis_vector_table: pd.DataFrame, basis_name: str) -> pd.Series:
    matches = basis_vector_table.loc[basis_vector_table["basis_name"].astype(str) == str(basis_name)]
    if matches.empty:
        raise ValueError(f"Basis vector {basis_name!r} was not found in basis_vector_table.")
    return matches.iloc[0]


def _infer_basis_role(name: str, roi_id: str | None = None) -> str | None:
    value = str(name)
    if roi_id is not None:
        prefix = f"{roi_id}_"
        if value.startswith(prefix) and len(value) > len(prefix):
            return value[len(prefix) :]
    parts = value.split("_")
    if len(parts) >= 2 and parts[-1]:
        return parts[-1]
    return value if value in {"a", "b", "c", "u", "v"} else None


def resolve_basis_vector_specs(
    points: pd.DataFrame,
    basis_specs: list[BasisVectorSpec] | tuple[BasisVectorSpec, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in basis_specs:
        source_type = "vector"
        from_point_id_1 = pd.NA
        from_point_id_2 = pd.NA
        from_atom_id_1 = pd.NA
        from_atom_id_2 = pd.NA
        from_x1_px = np.nan
        from_y1_px = np.nan
        from_x2_px = np.nan
        from_y2_px = np.nan
        snapped = False
        if spec.vector_px is not None:
            dx, dy = (float(spec.vector_px[0]), float(spec.vector_px[1]))
        elif spec.from_point_ids is not None:
            source_type = "point_ids"
            row_1 = _point_row(points, spec.from_point_ids[0])
            row_2 = _point_row(points, spec.from_point_ids[1])
            from_point_id_1 = _point_id(row_1)
            from_point_id_2 = _point_id(row_2)
            from_atom_id_1 = _atom_id_or_na(row_1)
            from_atom_id_2 = _atom_id_or_na(row_2)
            from_x1_px, from_y1_px = float(row_1["x_px"]), float(row_1["y_px"])
            from_x2_px, from_y2_px = float(row_2["x_px"]), float(row_2["y_px"])
            dx, dy = from_x2_px - from_x1_px, from_y2_px - from_y1_px
        elif spec.from_atom_ids is not None:
            source_type = "atom_ids"
            row_1 = _atom_row(points, int(spec.from_atom_ids[0]))
            row_2 = _atom_row(points, int(spec.from_atom_ids[1]))
            from_point_id_1 = _point_id(row_1)
            from_point_id_2 = _point_id(row_2)
            from_atom_id_1 = _atom_id_or_na(row_1)
            from_atom_id_2 = _atom_id_or_na(row_2)
            from_x1_px, from_y1_px = float(row_1["x_px"]), float(row_1["y_px"])
            from_x2_px, from_y2_px = float(row_2["x_px"]), float(row_2["y_px"])
            dx, dy = from_x2_px - from_x1_px, from_y2_px - from_y1_px
        elif spec.from_xy_px is not None:
            source_type = "xy"
            xy_1 = (float(spec.from_xy_px[0][0]), float(spec.from_xy_px[0][1]))
            xy_2 = (float(spec.from_xy_px[1][0]), float(spec.from_xy_px[1][1]))
            if spec.snap_to_nearest_points:
                row_1 = _nearest_point(points, xy_1)
                row_2 = _nearest_point(points, xy_2)
                from_point_id_1 = _point_id(row_1)
                from_point_id_2 = _point_id(row_2)
                from_atom_id_1 = _atom_id_or_na(row_1)
                from_atom_id_2 = _atom_id_or_na(row_2)
                xy_1 = (float(row_1["x_px"]), float(row_1["y_px"]))
                xy_2 = (float(row_2["x_px"]), float(row_2["y_px"]))
                source_type = "xy_snapped"
                snapped = True
            from_x1_px, from_y1_px = xy_1
            from_x2_px, from_y2_px = xy_2
            dx, dy = from_x2_px - from_x1_px, from_y2_px - from_y1_px
        else:
            raise ValueError(f"BasisVectorSpec {spec.name!r} must define vector_px, from_point_ids, from_atom_ids, or from_xy_px.")

        vector = np.asarray([dx, dy], dtype=float)
        length = float(np.linalg.norm(vector))
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError(f"BasisVectorSpec {spec.name!r} has a zero or invalid vector.")
        ux, uy = vector / length
        period_px = float(spec.period_px) if spec.period_px is not None else float(length)
        rows.append(
            {
                "basis_name": str(spec.name),
                "roi_id": spec.roi_id,
                "basis_role": spec.basis_role or _infer_basis_role(str(spec.name), spec.roi_id),
                "is_global": spec.roi_id is None,
                "vector_x_px": float(dx),
                "vector_y_px": float(dy),
                "length_px": float(length),
                "ux": float(ux),
                "uy": float(uy),
                "vx": float(-uy),
                "vy": float(ux),
                "angle_deg": float(np.degrees(np.arctan2(uy, ux))),
                "period_px": period_px,
                "source_type": source_type,
                "from_point_id_1": from_point_id_1,
                "from_point_id_2": from_point_id_2,
                "from_atom_id_1": from_atom_id_1,
                "from_atom_id_2": from_atom_id_2,
                "from_x1_px": from_x1_px,
                "from_y1_px": from_y1_px,
                "from_x2_px": from_x2_px,
                "from_y2_px": from_y2_px,
                "snapped": snapped,
            }
        )
    return pd.DataFrame(rows)


def _roi_records(rois: list[AnalysisROI] | tuple[AnalysisROI, ...] | pd.DataFrame | None) -> list[dict[str, Any]]:
    if isinstance(rois, pd.DataFrame):
        if rois.empty:
            return [{"roi_id": "global", "roi_name": "global", "roi_color": "#ff9f1c", "enabled": True}]
        records = []
        for _, row in rois.drop_duplicates("roi_id").iterrows():
            records.append(
                {
                    "roi_id": str(row.get("roi_id", "global")),
                    "roi_name": str(row.get("roi_name", row.get("roi_id", "global"))),
                    "roi_color": str(row.get("roi_color", "#ff9f1c")),
                    "enabled": True,
                }
            )
        return records
    records = []
    for roi in _default_rois(rois):
        records.append(
            {
                "roi_id": str(roi.roi_id),
                "roi_name": str(roi.roi_name or roi.roi_id),
                "roi_color": str(roi.color),
                "enabled": bool(roi.enabled),
            }
        )
    return records


def build_roi_basis_table(
    rois: list[AnalysisROI] | tuple[AnalysisROI, ...] | pd.DataFrame | None,
    basis_vector_table: pd.DataFrame,
    *,
    basis_roles: tuple[str, ...] = ("a", "b"),
    global_fallback: bool = True,
) -> pd.DataFrame:
    basis_table = basis_vector_table.copy() if basis_vector_table is not None else pd.DataFrame()
    records: list[dict[str, Any]] = []
    for roi in _roi_records(rois):
        if not roi.get("enabled", True):
            continue
        roi_id = str(roi["roi_id"])
        for role in basis_roles:
            role = str(role)
            selected: pd.Series | None = None
            if not basis_table.empty and {"roi_id", "basis_role"}.issubset(basis_table.columns):
                specific = basis_table.loc[
                    (basis_table["roi_id"].astype("object").astype(str) == roi_id)
                    & (basis_table["basis_role"].astype("object").astype(str) == role)
                ]
                if not specific.empty:
                    selected = specific.iloc[0]
            if selected is None and global_fallback and not basis_table.empty:
                global_rows = basis_table.loc[
                    (basis_table.get("is_global", pd.Series(False, index=basis_table.index)).astype(bool))
                    & (basis_table.get("basis_role", pd.Series(pd.NA, index=basis_table.index)).astype("object").astype(str) == role)
                ]
                if global_rows.empty:
                    global_rows = basis_table.loc[basis_table["basis_name"].astype(str) == role]
                if not global_rows.empty:
                    selected = global_rows.iloc[0]
            records.append(
                {
                    "roi_id": roi_id,
                    "roi_name": roi.get("roi_name", roi_id),
                    "basis_role": role,
                    "basis_name": pd.NA if selected is None else selected["basis_name"],
                    "is_global": pd.NA if selected is None else bool(selected.get("is_global", False)),
                    "found": selected is not None,
                }
            )
    return pd.DataFrame(records)


def make_basis_specs_for_rois(
    rois: list[AnalysisROI] | tuple[AnalysisROI, ...] | pd.DataFrame | None,
    basis_roles: tuple[str, ...] = ("a", "b"),
) -> list[BasisVectorSpec]:
    specs: list[BasisVectorSpec] = []
    for roi in _roi_records(rois):
        if not roi.get("enabled", True):
            continue
        roi_id = str(roi["roi_id"])
        if roi_id == "global":
            for role in basis_roles:
                specs.append(BasisVectorSpec(name=str(role), roi_id=None, basis_role=str(role)))
        else:
            for role in basis_roles:
                specs.append(BasisVectorSpec(name=f"{roi_id}_{role}", roi_id=roi_id, basis_role=str(role)))
    return specs


def expand_tasks_by_roi_basis(
    rois: list[AnalysisROI] | tuple[AnalysisROI, ...] | pd.DataFrame | None,
    roi_basis_table: pd.DataFrame,
    *,
    task_kind: str,
    basis_role: str = "a",
    template_name: str,
    point_set: str = "atoms",
    class_ids: tuple[int, ...] | None = None,
    class_names: tuple[str, ...] | None = None,
    nearest_forward_kwargs: dict[str, Any] | None = None,
    periodic_kwargs: dict[str, Any] | None = None,
    line_guide_kwargs: dict[str, Any] | None = None,
) -> list[Any]:
    del rois  # roi_basis_table is the source of truth after mapping.
    if roi_basis_table is None or roi_basis_table.empty:
        return []
    task_kind = str(task_kind)
    rows = roi_basis_table.loc[
        (roi_basis_table["basis_role"].astype(str) == str(basis_role))
        & (roi_basis_table["found"].astype(bool))
    ]
    tasks: list[Any] = []
    for _, row in rows.iterrows():
        roi_id = str(row["roi_id"])
        basis_name = str(row["basis_name"])
        name = f"{roi_id}_{template_name}_{basis_role}"
        common = {"name": name, "basis": basis_name, "point_set": point_set, "roi_ids": (roi_id,)}
        if task_kind == "nearest_forward":
            kwargs = dict(nearest_forward_kwargs or {})
            tasks.append(
                NearestForwardTask(
                    **common,
                    source_class_ids=class_ids,
                    target_class_ids=class_ids,
                    source_class_names=class_names,
                    target_class_names=class_names,
                    **kwargs,
                )
            )
        elif task_kind == "periodic_vector":
            kwargs = dict(periodic_kwargs or {})
            tasks.append(PeriodicVectorTask(**common, class_ids=class_ids, class_names=class_names, **kwargs))
        elif task_kind == "line_guide":
            kwargs = dict(line_guide_kwargs or {})
            tasks.append(LineGuideTask(**common, class_ids=class_ids, class_names=class_names, **kwargs))
        else:
            raise ValueError("task_kind must be 'nearest_forward', 'periodic_vector', or 'line_guide'.")
    return tasks


def flip_basis_vectors(
    basis_vector_table: pd.DataFrame,
    basis_names: tuple[str, ...] | list[str] | set[str],
) -> pd.DataFrame:
    table = basis_vector_table.copy()
    if table.empty or not basis_names:
        return table
    names = {str(value) for value in basis_names}
    mask = table["basis_name"].astype(str).isin(names)
    for index in table.index[mask]:
        table.loc[index, "vector_x_px"] = -float(table.loc[index, "vector_x_px"])
        table.loc[index, "vector_y_px"] = -float(table.loc[index, "vector_y_px"])
        table.loc[index, "ux"] = -float(table.loc[index, "ux"])
        table.loc[index, "uy"] = -float(table.loc[index, "uy"])
        table.loc[index, "vx"] = -float(table.loc[index, "uy"])
        table.loc[index, "vy"] = float(table.loc[index, "ux"])
        table.loc[index, "angle_deg"] = float(np.degrees(np.arctan2(float(table.loc[index, "uy"]), float(table.loc[index, "ux"]))))
        for left, right in (
            ("from_x1_px", "from_x2_px"),
            ("from_y1_px", "from_y2_px"),
            ("from_point_id_1", "from_point_id_2"),
            ("from_atom_id_1", "from_atom_id_2"),
        ):
            if left in table.columns and right in table.columns:
                old_left = table.loc[index, left]
                table.loc[index, left] = table.loc[index, right]
                table.loc[index, right] = old_left
    return table


def resolve_direction_specs(quant_points: pd.DataFrame, directions: list[DirectionSpec] | tuple[DirectionSpec, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in directions:
        source_type = "vector"
        from_atom_id_1 = np.nan
        from_atom_id_2 = np.nan
        from_x1_px = np.nan
        from_y1_px = np.nan
        from_x2_px = np.nan
        from_y2_px = np.nan
        snapped = bool(spec.snap_to_nearest_atoms)
        if spec.vector_px is not None:
            dx, dy = (float(spec.vector_px[0]), float(spec.vector_px[1]))
            snapped = False
        elif spec.from_atom_ids is not None:
            source_type = "atom_ids"
            row_1 = _atom_row(quant_points, int(spec.from_atom_ids[0]))
            row_2 = _atom_row(quant_points, int(spec.from_atom_ids[1]))
            from_atom_id_1 = int(row_1["atom_id"])
            from_atom_id_2 = int(row_2["atom_id"])
            from_x1_px, from_y1_px = float(row_1["x_px"]), float(row_1["y_px"])
            from_x2_px, from_y2_px = float(row_2["x_px"]), float(row_2["y_px"])
            dx, dy = from_x2_px - from_x1_px, from_y2_px - from_y1_px
        elif spec.from_xy_px is not None:
            source_type = "xy"
            xy_1 = (float(spec.from_xy_px[0][0]), float(spec.from_xy_px[0][1]))
            xy_2 = (float(spec.from_xy_px[1][0]), float(spec.from_xy_px[1][1]))
            if spec.snap_to_nearest_atoms:
                row_1 = _nearest_atom(quant_points, xy_1)
                row_2 = _nearest_atom(quant_points, xy_2)
                from_atom_id_1 = int(row_1["atom_id"])
                from_atom_id_2 = int(row_2["atom_id"])
                xy_1 = (float(row_1["x_px"]), float(row_1["y_px"]))
                xy_2 = (float(row_2["x_px"]), float(row_2["y_px"]))
                source_type = "xy_snapped"
            from_x1_px, from_y1_px = xy_1
            from_x2_px, from_y2_px = xy_2
            dx, dy = from_x2_px - from_x1_px, from_y2_px - from_y1_px
        else:
            raise ValueError(f"DirectionSpec {spec.name!r} must define vector_px, from_atom_ids, or from_xy_px.")

        vector = np.asarray([dx, dy], dtype=float)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"DirectionSpec {spec.name!r} has a zero or invalid direction vector.")
        ux, uy = vector / norm
        rows.append(
            {
                "direction_name": str(spec.name),
                "ux": float(ux),
                "uy": float(uy),
                "vx": float(-uy),
                "vy": float(ux),
                "angle_deg": float(np.degrees(np.arctan2(uy, ux))),
                "source_type": source_type,
                "from_atom_id_1": from_atom_id_1,
                "from_atom_id_2": from_atom_id_2,
                "from_x1_px": from_x1_px,
                "from_y1_px": from_y1_px,
                "from_x2_px": from_x2_px,
                "from_y2_px": from_y2_px,
                "snapped": snapped,
            }
        )
    return pd.DataFrame(rows)


def _direction_row(direction_table: pd.DataFrame, direction_name: str) -> pd.Series:
    matches = direction_table.loc[direction_table["direction_name"].astype(str) == str(direction_name)]
    if matches.empty:
        raise ValueError(f"Direction {direction_name!r} was not found in direction_table.")
    return matches.iloc[0]


def _filter_points(
    points: pd.DataFrame,
    *,
    classes: tuple[str, ...] | None = None,
    class_ids: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    result = points
    if classes is not None:
        allowed = {str(value) for value in classes}
        result = result.loc[result["class_name"].astype(str).isin(allowed)]
    if class_ids is not None:
        allowed_ids = {int(value) for value in class_ids}
        result = result.loc[result["class_id"].isin(allowed_ids)]
    return result.copy().reset_index(drop=True)


def _angle_error_deg(dx: float, dy: float, ux: float, uy: float, vx: float, vy: float) -> float:
    parallel = dx * ux + dy * uy
    perpendicular = dx * vx + dy * vy
    return float(np.degrees(np.arctan2(abs(perpendicular), parallel)))


def _measurement_record(
    *,
    prefix: str,
    source: pd.Series,
    target: pd.Series,
    direction_name: str | None,
    dx: float,
    dy: float,
    distance_px: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_x_px = float(source["x_px"])
    source_y_px = float(source["y_px"])
    target_x_px = float(target["x_px"])
    target_y_px = float(target["y_px"])
    source_x_nm = float(source["x_nm"]) if "x_nm" in source.index and pd.notna(source["x_nm"]) else np.nan
    source_y_nm = float(source["y_nm"]) if "y_nm" in source.index and pd.notna(source["y_nm"]) else np.nan
    target_x_nm = float(target["x_nm"]) if "x_nm" in target.index and pd.notna(target["x_nm"]) else np.nan
    target_y_nm = float(target["y_nm"]) if "y_nm" in target.index and pd.notna(target["y_nm"]) else np.nan
    if np.isfinite([source_x_nm, source_y_nm, target_x_nm, target_y_nm]).all():
        distance_nm = float(np.hypot(target_x_nm - source_x_nm, target_y_nm - source_y_nm))
    else:
        distance_nm = np.nan
    record = {
        f"{prefix}_atom_id": int(source["atom_id"]),
        "target_atom_id": int(target["atom_id"]),
        f"{prefix}_class_id": source.get("class_id", np.nan),
        "target_class_id": target.get("class_id", np.nan),
        f"{prefix}_class_name": source.get("class_name", pd.NA),
        "target_class_name": target.get("class_name", pd.NA),
        f"{prefix}_x_px": source_x_px,
        f"{prefix}_y_px": source_y_px,
        "target_x_px": target_x_px,
        "target_y_px": target_y_px,
        "mid_x_px": (source_x_px + target_x_px) / 2.0,
        "mid_y_px": (source_y_px + target_y_px) / 2.0,
        "dx_px": float(dx),
        "dy_px": float(dy),
        "distance_px": float(distance_px),
        "distance_nm": distance_nm,
        "distance_pm": distance_nm * 1000.0 if np.isfinite(distance_nm) else np.nan,
        "direction_name": direction_name,
        "status": "ok",
    }
    if extra:
        record.update(extra)
    return record


def compute_directional_spacing(
    quant_points: pd.DataFrame,
    direction_table: pd.DataFrame,
    tasks: list[DirectionalSpacingTask] | tuple[DirectionalSpacingTask, ...],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for task in tasks:
        direction = _direction_row(direction_table, task.direction)
        ux, uy, vx, vy = (float(direction[key]) for key in ("ux", "uy", "vx", "vy"))
        sources = _filter_points(quant_points, classes=task.source_classes, class_ids=task.source_class_ids)
        targets = _filter_points(quant_points, classes=task.target_classes, class_ids=task.target_class_ids)
        seen: set[tuple[int, int]] = set()
        for _, source in sources.iterrows():
            best: tuple[float, pd.Series, float, float, float, float, float] | None = None
            for _, target in targets.iterrows():
                dx = float(target["x_px"] - source["x_px"])
                dy = float(target["y_px"] - source["y_px"])
                distance_px = float(np.hypot(dx, dy))
                if not np.isfinite(distance_px) or distance_px <= 0.0:
                    continue
                parallel = dx * ux + dy * uy
                perpendicular = dx * vx + dy * vy
                angle_error = _angle_error_deg(dx, dy, ux, uy, vx, vy)
                if parallel <= float(task.min_parallel_distance_px):
                    continue
                if task.max_parallel_distance_px is not None and parallel > float(task.max_parallel_distance_px):
                    continue
                if abs(perpendicular) > float(task.perpendicular_tolerance_px):
                    continue
                if angle_error > float(task.angle_tolerance_deg):
                    continue
                if best is None or parallel < best[0]:
                    best = (parallel, target, perpendicular, distance_px, dx, dy, angle_error)
            if best is None:
                continue
            parallel, target, perpendicular, distance_px, dx, dy, angle_error = best
            pair_key = tuple(sorted((int(source["atom_id"]), int(target["atom_id"]))))
            if task.unique_pairs and pair_key in seen:
                continue
            seen.add(pair_key)
            records.append(
                {
                    "measurement_name": task.name,
                    **_measurement_record(
                        prefix="source",
                        source=source,
                        target=target,
                        direction_name=task.direction,
                        dx=dx,
                        dy=dy,
                        distance_px=distance_px,
                        extra={
                            "parallel_distance_px": float(parallel),
                            "perpendicular_offset_px": float(perpendicular),
                            "angle_error_deg": float(angle_error),
                        },
                    ),
                }
            )
    return pd.DataFrame(records)


def compute_pair_distances(
    quant_points: pd.DataFrame,
    tasks: list[PairDistanceTask] | tuple[PairDistanceTask, ...],
    direction_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for task in tasks:
        if str(task.mode).lower() != "nearest":
            raise ValueError("PairDistanceTask.mode currently supports only 'nearest'.")
        direction = None
        if task.direction is not None:
            if direction_table is None:
                raise ValueError("direction_table is required when PairDistanceTask.direction is set.")
            direction = _direction_row(direction_table, task.direction)
        if task.explicit_atom_pairs is not None:
            pair_rows = [(_atom_row(quant_points, a), _atom_row(quant_points, b)) for a, b in task.explicit_atom_pairs]
        else:
            sources = _filter_points(quant_points, classes=task.source_classes, class_ids=task.source_class_ids)
            targets = _filter_points(quant_points, classes=task.target_classes, class_ids=task.target_class_ids)
            pair_rows = []
            for _, source in sources.iterrows():
                best_target: pd.Series | None = None
                best_distance = np.inf
                for _, target in targets.iterrows():
                    if int(source["atom_id"]) == int(target["atom_id"]):
                        continue
                    dx = float(target["x_px"] - source["x_px"])
                    dy = float(target["y_px"] - source["y_px"])
                    distance_px = float(np.hypot(dx, dy))
                    if task.max_distance_px is not None and distance_px > float(task.max_distance_px):
                        continue
                    if direction is not None and task.angle_tolerance_deg is not None:
                        angle_error = _angle_error_deg(
                            dx,
                            dy,
                            float(direction["ux"]),
                            float(direction["uy"]),
                            float(direction["vx"]),
                            float(direction["vy"]),
                        )
                        if angle_error > float(task.angle_tolerance_deg):
                            continue
                    if distance_px < best_distance:
                        best_distance = distance_px
                        best_target = target
                if best_target is not None:
                    pair_rows.append((source, best_target))

        seen: set[tuple[int, int]] = set()
        for source, target in pair_rows:
            pair_key = tuple(sorted((int(source["atom_id"]), int(target["atom_id"]))))
            if task.unique_pairs and pair_key in seen:
                continue
            seen.add(pair_key)
            dx = float(target["x_px"] - source["x_px"])
            dy = float(target["y_px"] - source["y_px"])
            distance_px = float(np.hypot(dx, dy))
            if task.max_distance_px is not None and distance_px > float(task.max_distance_px):
                continue
            angle_error = np.nan
            if direction is not None:
                angle_error = _angle_error_deg(
                    dx,
                    dy,
                    float(direction["ux"]),
                    float(direction["uy"]),
                    float(direction["vx"]),
                    float(direction["vy"]),
                )
                if task.angle_tolerance_deg is not None and angle_error > float(task.angle_tolerance_deg):
                    continue
            records.append(
                {
                    "pair_name": task.name,
                    **_measurement_record(
                        prefix="source",
                        source=source,
                        target=target,
                        direction_name=task.direction,
                        dx=dx,
                        dy=dy,
                        distance_px=distance_px,
                        extra={
                            "angle_deg": float(np.degrees(np.arctan2(dy, dx))),
                            "angle_error_deg": float(angle_error) if np.isfinite(angle_error) else np.nan,
                        },
                    ),
                }
            )
    return pd.DataFrame(records)


def assign_lines_by_projection(
    quant_points: pd.DataFrame,
    direction_table: pd.DataFrame,
    task: LineGroupingTask,
) -> pd.DataFrame:
    group_axis = str(task.group_axis).lower()
    if group_axis not in {"s", "t"}:
        raise ValueError("LineGroupingTask.group_axis must be 's' or 't'.")
    sort_axis = str(task.sort_axis or ("s" if group_axis == "t" else "t")).lower()
    if sort_axis not in {"s", "t"}:
        raise ValueError("LineGroupingTask.sort_axis must be 's', 't', or None.")
    direction = _direction_row(direction_table, task.direction)
    ux, uy, vx, vy = (float(direction[key]) for key in ("ux", "uy", "vx", "vy"))
    points = _filter_points(quant_points, classes=task.classes, class_ids=task.class_ids)
    if points.empty:
        return pd.DataFrame()
    points = points.copy()
    points["s_coord_px"] = points["x_px"] * ux + points["y_px"] * uy
    points["t_coord_px"] = points["x_px"] * vx + points["y_px"] * vy
    points["group_coord_px"] = points[f"{group_axis}_coord_px"]
    points["sort_coord_px"] = points[f"{sort_axis}_coord_px"]
    ordered = points.sort_values("group_coord_px").reset_index(drop=True)

    raw_groups: list[list[int]] = []
    current: list[int] = []
    previous_coord: float | None = None
    for index, row in ordered.iterrows():
        coord = float(row["group_coord_px"])
        if previous_coord is None or coord - previous_coord <= float(task.line_tolerance_px):
            current.append(index)
        else:
            raw_groups.append(current)
            current = [index]
        previous_coord = coord
    if current:
        raw_groups.append(current)

    records: list[dict[str, Any]] = []
    line_id = 0
    for group in raw_groups:
        if len(group) < int(task.min_atoms_per_line):
            continue
        group_frame = ordered.iloc[group].copy()
        line_center = float(group_frame["group_coord_px"].mean())
        group_frame = group_frame.sort_values("sort_coord_px")
        atom_count = int(len(group_frame))
        for _, row in group_frame.iterrows():
            records.append(
                {
                    "line_task_name": task.name,
                    "direction_name": task.direction,
                    "group_axis": group_axis,
                    "line_id": line_id,
                    "atom_id": int(row["atom_id"]),
                    "class_id": row.get("class_id", np.nan),
                    "class_name": row.get("class_name", pd.NA),
                    "x_px": float(row["x_px"]),
                    "y_px": float(row["y_px"]),
                    "s_coord_px": float(row["s_coord_px"]),
                    "t_coord_px": float(row["t_coord_px"]),
                    "group_coord_px": float(row["group_coord_px"]),
                    "sort_coord_px": float(row["sort_coord_px"]),
                    "line_center_px": line_center,
                    "line_atom_count": atom_count,
                    "status": "ok",
                }
            )
        line_id += 1
    return pd.DataFrame(records)


def _line_width(values: pd.Series, method: str) -> float:
    coords = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if coords.size == 0:
        return np.nan
    if method == "p95_p5":
        return float(np.percentile(coords, 95) - np.percentile(coords, 5))
    if method == "std":
        return float(np.std(coords, ddof=0))
    raise ValueError("line_width_method must be 'p95_p5' or 'std'.")


def compute_line_spacing(
    quant_points: pd.DataFrame,
    line_assignments: pd.DataFrame,
    direction_table: pd.DataFrame,
    task: LineGroupingTask,
) -> pd.DataFrame:
    if line_assignments.empty:
        return pd.DataFrame()
    _direction_row(direction_table, task.direction)
    pixel_to_nm = _pixel_to_nm_from_points(quant_points)
    by_atom = quant_points.set_index("atom_id", drop=False)
    records: list[dict[str, Any]] = []
    for line_id, line in line_assignments.groupby("line_id", sort=True):
        line = line.sort_values("sort_coord_px").reset_index(drop=True)
        width_px = _line_width(line["group_coord_px"], task.line_width_method)
        width_nm = width_px * pixel_to_nm if pixel_to_nm is not None and np.isfinite(width_px) else np.nan
        spacings_px: list[float] = []
        pair_values: list[tuple[float, float, float, float, int, float, float, str]] = []
        for index, row in line.iterrows():
            if index == len(line) - 1:
                pair_values.append((np.nan, np.nan, np.nan, np.nan, -1, np.nan, np.nan, "last_in_line"))
                continue
            next_row = line.iloc[index + 1]
            dx = float(next_row["x_px"] - row["x_px"])
            dy = float(next_row["y_px"] - row["y_px"])
            spacing_px = float(np.hypot(dx, dy))
            status = "ok"
            if task.max_in_line_gap_px is not None and spacing_px > float(task.max_in_line_gap_px):
                status = "gap_exceeds_max"
                spacing_px = np.nan
            if np.isfinite(spacing_px):
                spacings_px.append(spacing_px)
            pair_values.append(
                (
                    float(next_row["x_px"]),
                    float(next_row["y_px"]),
                    dx,
                    dy,
                    int(next_row["atom_id"]),
                    spacing_px,
                    float(next_row["sort_coord_px"]),
                    status,
                )
            )
        mean_spacing_px = float(np.mean(spacings_px)) if spacings_px else np.nan
        std_spacing_px = float(np.std(spacings_px, ddof=1)) if len(spacings_px) > 1 else np.nan
        for row, pair in zip(line.to_dict(orient="records"), pair_values, strict=True):
            next_x, next_y, dx, dy, next_atom_id, spacing_px, _next_sort, status = pair
            source_row = by_atom.loc[row["atom_id"]] if row["atom_id"] in by_atom.index else pd.Series(row)
            if next_atom_id >= 0 and next_atom_id in by_atom.index:
                target_row = by_atom.loc[next_atom_id]
                spacing_record = _measurement_record(
                    prefix="source",
                    source=source_row,
                    target=target_row,
                    direction_name=task.direction,
                    dx=dx,
                    dy=dy,
                    distance_px=spacing_px if np.isfinite(spacing_px) else np.nan,
                )
                spacing_nm = spacing_record["distance_nm"]
                spacing_pm = spacing_record["distance_pm"]
            else:
                spacing_nm = np.nan
                spacing_pm = np.nan
            records.append(
                {
                    "line_task_name": task.name,
                    "direction_name": task.direction,
                    "group_axis": task.group_axis,
                    "line_id": int(line_id),
                    "atom_id": int(row["atom_id"]),
                    "class_id": row.get("class_id", np.nan),
                    "class_name": row.get("class_name", pd.NA),
                    "x_px": float(row["x_px"]),
                    "y_px": float(row["y_px"]),
                    "s_coord_px": float(row["s_coord_px"]),
                    "t_coord_px": float(row["t_coord_px"]),
                    "group_coord_px": float(row["group_coord_px"]),
                    "sort_coord_px": float(row["sort_coord_px"]),
                    "next_atom_id": next_atom_id if next_atom_id >= 0 else np.nan,
                    "next_x_px": next_x,
                    "next_y_px": next_y,
                    "spacing_to_next_px": spacing_px,
                    "spacing_to_next_nm": spacing_nm,
                    "spacing_to_next_pm": spacing_pm,
                    "line_center_px": float(row["line_center_px"]),
                    "line_width_px": width_px,
                    "line_width_nm": width_nm,
                    "line_width_pm": width_nm * 1000.0 if np.isfinite(width_nm) else np.nan,
                    "line_atom_count": int(row["line_atom_count"]),
                    "line_mean_spacing_px": mean_spacing_px,
                    "line_mean_spacing_pm": mean_spacing_px * pixel_to_nm * 1000.0
                    if pixel_to_nm is not None and np.isfinite(mean_spacing_px)
                    else np.nan,
                    "line_std_spacing_px": std_spacing_px,
                    "line_std_spacing_pm": std_spacing_px * pixel_to_nm * 1000.0
                    if pixel_to_nm is not None and np.isfinite(std_spacing_px)
                    else np.nan,
                    "status": status,
                }
            )
    return pd.DataFrame(records)


_SEGMENT_COLUMNS = [
    "segment_id",
    "task_name",
    "task_type",
    "roi_id",
    "roi_name",
    "roi_color",
    "scope_id",
    "point_set",
    "source_point_id",
    "target_point_id",
    "source_atom_id",
    "target_atom_id",
    "source_class_id",
    "target_class_id",
    "source_class_name",
    "target_class_name",
    "source_x_px",
    "source_y_px",
    "target_x_px",
    "target_y_px",
    "source_x_nm",
    "source_y_nm",
    "target_x_nm",
    "target_y_nm",
    "mid_x_px",
    "mid_y_px",
    "dx_px",
    "dy_px",
    "distance_px",
    "distance_nm",
    "distance_pm",
    "basis_name",
    "parallel_distance_px",
    "perpendicular_offset_px",
    "angle_error_deg",
    "period_residual_px",
    "line_id",
    "chain_id",
    "period_index",
    "status",
]


def _empty_segments() -> pd.DataFrame:
    return pd.DataFrame(columns=_SEGMENT_COLUMNS)


def _filter_points_v2(
    points: pd.DataFrame,
    *,
    point_set: str | None = None,
    roi_ids: tuple[str, ...] | None = None,
    class_ids: tuple[int, ...] | None = None,
    class_names: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    result = points.copy()
    if point_set is not None and "point_set" in result.columns:
        result = result.loc[result["point_set"].astype(str) == str(point_set)].copy()
    if roi_ids is not None and "roi_id" in result.columns:
        result = result.loc[result["roi_id"].astype(str).isin({str(value) for value in roi_ids})].copy()
    if class_ids is not None and "class_id" in result.columns:
        result = result.loc[result["class_id"].isin({int(value) for value in class_ids})].copy()
    if class_names is not None and "class_name" in result.columns:
        result = result.loc[result["class_name"].astype(str).isin({str(value) for value in class_names})].copy()
    return result.reset_index(drop=True)


def _segment_record(
    *,
    task_name: str,
    task_type: str,
    source: pd.Series,
    target: pd.Series,
    basis: pd.Series | None = None,
    line_id: Any = pd.NA,
    chain_id: Any = pd.NA,
    period_index: Any = pd.NA,
    period_residual_px: float = np.nan,
    status: str = "ok",
) -> dict[str, Any]:
    source_x = float(source["x_px"])
    source_y = float(source["y_px"])
    target_x = float(target["x_px"])
    target_y = float(target["y_px"])
    source_x_nm = source.get("x_nm", np.nan)
    source_y_nm = source.get("y_nm", np.nan)
    target_x_nm = target.get("x_nm", np.nan)
    target_y_nm = target.get("y_nm", np.nan)
    dx = target_x - source_x
    dy = target_y - source_y
    distance_px = float(np.hypot(dx, dy))
    distance_nm = _pixel_to_nm_from_row_pair(source, target)
    parallel = np.nan
    perpendicular = np.nan
    angle_error = np.nan
    basis_name = pd.NA
    if basis is not None:
        ux, uy, vx, vy = (float(basis[key]) for key in ("ux", "uy", "vx", "vy"))
        parallel = float(dx * ux + dy * uy)
        perpendicular = float(dx * vx + dy * vy)
        angle_error = _angle_error_deg(dx, dy, ux, uy, vx, vy)
        basis_name = str(basis["basis_name"])
    return {
        "task_name": str(task_name),
        "task_type": str(task_type),
        "roi_id": source.get("roi_id", pd.NA),
        "roi_name": source.get("roi_name", pd.NA),
        "roi_color": source.get("roi_color", pd.NA),
        "scope_id": source.get("scope_id", pd.NA),
        "point_set": source.get("point_set", pd.NA),
        "source_point_id": _point_id(source),
        "target_point_id": _point_id(target),
        "source_atom_id": _atom_id_or_na(source),
        "target_atom_id": _atom_id_or_na(target),
        "source_class_id": source.get("class_id", pd.NA),
        "target_class_id": target.get("class_id", pd.NA),
        "source_class_name": source.get("class_name", pd.NA),
        "target_class_name": target.get("class_name", pd.NA),
        "source_x_px": source_x,
        "source_y_px": source_y,
        "target_x_px": target_x,
        "target_y_px": target_y,
        "source_x_nm": source_x_nm,
        "source_y_nm": source_y_nm,
        "target_x_nm": target_x_nm,
        "target_y_nm": target_y_nm,
        "mid_x_px": (source_x + target_x) / 2.0,
        "mid_y_px": (source_y + target_y) / 2.0,
        "dx_px": dx,
        "dy_px": dy,
        "distance_px": distance_px,
        "distance_nm": distance_nm,
        "distance_pm": distance_nm * 1000.0 if np.isfinite(distance_nm) else np.nan,
        "basis_name": basis_name,
        "parallel_distance_px": parallel,
        "perpendicular_offset_px": perpendicular,
        "angle_error_deg": angle_error,
        "period_residual_px": period_residual_px,
        "line_id": line_id,
        "chain_id": chain_id,
        "period_index": period_index,
        "status": status,
    }


def _finalize_segments(records: list[dict[str, Any]], task_name: str) -> pd.DataFrame:
    if not records:
        return _empty_segments()
    table = pd.DataFrame(records)
    table.insert(0, "segment_id", [f"{task_name}:{index}" for index in range(len(table))])
    for column in _SEGMENT_COLUMNS:
        if column not in table.columns:
            table[column] = pd.NA
    return table[_SEGMENT_COLUMNS]


def _scope_groups(points: pd.DataFrame):
    if points.empty:
        return []
    if "scope_id" not in points.columns:
        points = points.copy()
        point_set = points["point_set"].astype(str) if "point_set" in points.columns else pd.Series("points", index=points.index)
        points["scope_id"] = "global:" + point_set.astype(str)
    return points.groupby("scope_id", dropna=False, sort=False)


def compute_nearest_forward_segments(
    points: pd.DataFrame,
    basis_vector_table: pd.DataFrame,
    task: NearestForwardTask,
) -> pd.DataFrame:
    basis = _basis_row(basis_vector_table, task.basis)
    candidates = _filter_points_v2(points, point_set=task.point_set, roi_ids=task.roi_ids)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    ux, uy, vx, vy = (float(basis[key]) for key in ("ux", "uy", "vx", "vy"))
    for scope_id, group in _scope_groups(candidates):
        sources = _filter_points_v2(
            group,
            class_ids=task.source_class_ids,
            class_names=task.source_class_names,
        )
        targets = _filter_points_v2(
            group,
            class_ids=task.target_class_ids,
            class_names=task.target_class_names,
        )
        for _, source in sources.iterrows():
            best: tuple[float, float, float, pd.Series] | None = None
            for _, target in targets.iterrows():
                if _point_id(source) == _point_id(target):
                    continue
                dx = float(target["x_px"] - source["x_px"])
                dy = float(target["y_px"] - source["y_px"])
                parallel = dx * ux + dy * uy
                if parallel <= float(task.min_parallel_distance_px):
                    continue
                if task.max_parallel_distance_px is not None and parallel > float(task.max_parallel_distance_px):
                    continue
                perpendicular = dx * vx + dy * vy
                if abs(perpendicular) > float(task.perpendicular_tolerance_px):
                    continue
                angle_error = _angle_error_deg(dx, dy, ux, uy, vx, vy)
                if angle_error > float(task.angle_tolerance_deg):
                    continue
                if best is None or parallel < best[0]:
                    best = (float(parallel), float(perpendicular), float(angle_error), target)
            if best is None:
                continue
            _, _, _, target = best
            pair_key = tuple(sorted((_point_id(source), _point_id(target))))
            seen_key = (str(scope_id), pair_key[0], pair_key[1])
            if task.unique_pairs and seen_key in seen:
                continue
            seen.add(seen_key)
            records.append(
                _segment_record(
                    task_name=task.name,
                    task_type="nearest_forward",
                    source=source,
                    target=target,
                    basis=basis,
                )
            )
    return _finalize_segments(records, task.name)


def compute_pair_segments(
    points: pd.DataFrame,
    basis_vector_table: pd.DataFrame | None,
    task: PairSegmentTask,
) -> pd.DataFrame:
    if str(task.mode).lower() != "nearest":
        raise ValueError("PairSegmentTask.mode currently supports only 'nearest'.")
    basis = _basis_row(basis_vector_table, task.basis) if task.basis is not None and basis_vector_table is not None else None
    candidates = _filter_points_v2(points, point_set=task.point_set, roi_ids=task.roi_ids)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for scope_id, group in _scope_groups(candidates):
        pair_rows: list[tuple[pd.Series, pd.Series]] = []
        if task.explicit_point_pairs is not None:
            for source_id, target_id in task.explicit_point_pairs:
                pair_rows.append((_point_row(group, source_id), _point_row(group, target_id)))
        elif task.explicit_atom_pairs is not None:
            for source_id, target_id in task.explicit_atom_pairs:
                pair_rows.append((_atom_row(group, int(source_id)), _atom_row(group, int(target_id))))
        else:
            sources = _filter_points_v2(group, class_ids=task.source_class_ids, class_names=task.source_class_names)
            targets = _filter_points_v2(group, class_ids=task.target_class_ids, class_names=task.target_class_names)
            for _, source in sources.iterrows():
                best: tuple[float, pd.Series] | None = None
                for _, target in targets.iterrows():
                    if _point_id(source) == _point_id(target):
                        continue
                    dx = float(target["x_px"] - source["x_px"])
                    dy = float(target["y_px"] - source["y_px"])
                    distance = float(np.hypot(dx, dy))
                    if task.max_distance_px is not None and distance > float(task.max_distance_px):
                        continue
                    if basis is not None and task.angle_tolerance_deg is not None:
                        angle_error = _angle_error_deg(
                            dx,
                            dy,
                            float(basis["ux"]),
                            float(basis["uy"]),
                            float(basis["vx"]),
                            float(basis["vy"]),
                        )
                        if angle_error > float(task.angle_tolerance_deg):
                            continue
                    if best is None or distance < best[0]:
                        best = (distance, target)
                if best is not None:
                    pair_rows.append((source, best[1]))

        for source, target in pair_rows:
            distance = float(np.hypot(float(target["x_px"] - source["x_px"]), float(target["y_px"] - source["y_px"])))
            if task.max_distance_px is not None and distance > float(task.max_distance_px):
                continue
            pair_key = tuple(sorted((_point_id(source), _point_id(target))))
            seen_key = (str(scope_id), pair_key[0], pair_key[1])
            if task.unique_pairs and seen_key in seen:
                continue
            seen.add(seen_key)
            records.append(
                _segment_record(
                    task_name=task.name,
                    task_type="explicit_pair" if task.explicit_point_pairs or task.explicit_atom_pairs else "nearest_class_pair",
                    source=source,
                    target=target,
                    basis=basis,
                )
            )
    return _finalize_segments(records, task.name)


def compute_periodic_vector_segments(
    points: pd.DataFrame,
    basis_vector_table: pd.DataFrame,
    task: PeriodicVectorTask,
) -> pd.DataFrame:
    basis = _basis_row(basis_vector_table, task.basis)
    candidates = _filter_points_v2(
        points,
        point_set=task.point_set,
        roi_ids=task.roi_ids,
        class_ids=task.class_ids,
        class_names=task.class_names,
    )
    period_px = float(basis.get("period_px", basis["length_px"]))
    vector = np.asarray([float(basis["ux"]), float(basis["uy"])], dtype=float) * period_px
    match_radius = float(task.match_radius_px) if task.match_radius_px is not None else period_px * float(task.match_radius_fraction)
    records: list[dict[str, Any]] = []
    chain_counter = 0
    for scope_id, group in _scope_groups(candidates):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            continue
        coords = group[["x_px", "y_px"]].to_numpy(dtype=float)
        tree = cKDTree(coords)
        candidate_rows: list[dict[str, Any]] = []
        for source_index, source_xy in enumerate(coords):
            expected = source_xy + vector
            residual, target_index = tree.query(expected, k=1)
            residual = float(residual)
            target_index = int(target_index)
            if target_index == source_index or residual > match_radius:
                continue
            source = group.iloc[source_index]
            target = group.iloc[target_index]
            candidate_rows.append(
                {
                    "source_index": source_index,
                    "target_index": target_index,
                    "residual": residual,
                    "source": source,
                    "target": target,
                }
            )
        if task.one_to_one and candidate_rows:
            best_by_target: dict[int, dict[str, Any]] = {}
            for row in candidate_rows:
                target_index = int(row["target_index"])
                if target_index not in best_by_target or row["residual"] < best_by_target[target_index]["residual"]:
                    best_by_target[target_index] = row
            candidate_rows = list(best_by_target.values())
        by_source = {int(row["source_index"]): row for row in candidate_rows}
        predecessor = {int(row["target_index"]): int(row["source_index"]) for row in candidate_rows}
        projections = coords @ np.asarray([float(basis["ux"]), float(basis["uy"])])
        starts = [idx for idx in by_source if idx not in predecessor]
        if not starts:
            starts = list(by_source)
        starts = sorted(starts, key=lambda idx: float(projections[idx]))
        emitted: set[int] = set()
        for start in starts:
            if start in emitted:
                continue
            current = int(start)
            seen_in_chain: set[int] = set()
            period_index = 0
            chain_id = f"{task.name}:{chain_counter}"
            chain_counter += 1
            while current in by_source:
                if current in seen_in_chain:
                    break
                if task.max_steps_per_chain is not None and period_index >= int(task.max_steps_per_chain):
                    break
                row = by_source[current]
                target_index = int(row["target_index"])
                status = "loop_detected" if target_index in seen_in_chain else "ok"
                records.append(
                    _segment_record(
                        task_name=task.name,
                        task_type="periodic_vector",
                        source=row["source"],
                        target=row["target"],
                        basis=basis,
                        chain_id=chain_id,
                        period_index=period_index,
                        period_residual_px=float(row["residual"]),
                        status=status,
                    )
                )
                emitted.add(current)
                seen_in_chain.add(current)
                if status == "loop_detected":
                    break
                current = target_index
                period_index += 1
    return _finalize_segments(records, task.name)


def _xy_from_st(s_coord: float, t_coord: float, basis: pd.Series) -> tuple[float, float]:
    ux, uy, vx, vy = (float(basis[key]) for key in ("ux", "uy", "vx", "vy"))
    return (s_coord * ux + t_coord * vx, s_coord * uy + t_coord * vy)


def _estimate_point_spacing_px(points: pd.DataFrame) -> float:
    data = points[["x_px", "y_px"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 2:
        return 8.0
    coords = data.to_numpy(dtype=float)
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
    return float(np.median(nearest)) if nearest.size else 8.0


def compute_line_guides(points: pd.DataFrame, basis_vector_table: pd.DataFrame, task: LineGuideTask) -> pd.DataFrame:
    basis = _basis_row(basis_vector_table, task.basis)
    group_axis = str(task.group_axis).lower()
    if group_axis not in {"s", "t"}:
        raise ValueError("LineGuideTask.group_axis must be 's' or 't'.")
    sort_axis = "s" if group_axis == "t" else "t"
    candidates = _filter_points_v2(
        points,
        point_set=task.point_set,
        roi_ids=task.roi_ids,
        class_ids=task.class_ids,
        class_names=task.class_names,
    )
    records: list[dict[str, Any]] = []
    ux, uy, vx, vy = (float(basis[key]) for key in ("ux", "uy", "vx", "vy"))
    for scope_id, group in _scope_groups(candidates):
        if group.empty:
            continue
        group = group.copy()
        group["s_coord_px"] = group["x_px"] * ux + group["y_px"] * uy
        group["t_coord_px"] = group["x_px"] * vx + group["y_px"] * vy
        group["group_coord_px"] = group[f"{group_axis}_coord_px"]
        group["sort_coord_px"] = group[f"{sort_axis}_coord_px"]
        ordered = group.sort_values("group_coord_px").reset_index(drop=True)
        raw_groups: list[list[int]] = []
        current_group: list[int] = []
        previous_coord: float | None = None
        for index, row in ordered.iterrows():
            coord = float(row["group_coord_px"])
            if previous_coord is None or coord - previous_coord <= float(task.line_tolerance_px):
                current_group.append(index)
            else:
                raw_groups.append(current_group)
                current_group = [index]
            previous_coord = coord
        if current_group:
            raw_groups.append(current_group)
        margin = max(float(_estimate_point_spacing_px(group)), 4.0)
        sort_min = float(group["sort_coord_px"].min()) - margin
        sort_max = float(group["sort_coord_px"].max()) + margin
        line_id = 0
        for raw_group in raw_groups:
            if len(raw_group) < int(task.min_points_per_line):
                continue
            line = ordered.iloc[raw_group].copy()
            center = float(line["group_coord_px"].mean())
            if group_axis == "t":
                start = _xy_from_st(sort_min, center, basis)
                end = _xy_from_st(sort_max, center, basis)
                label = _xy_from_st(sort_min - margin, center, basis)
            else:
                start = _xy_from_st(center, sort_min, basis)
                end = _xy_from_st(center, sort_max, basis)
                label = _xy_from_st(center, sort_min - margin, basis)
            first = line.iloc[0]
            records.append(
                {
                    "line_id": int(line_id),
                    "task_name": str(task.name),
                    "roi_id": first.get("roi_id", pd.NA),
                    "roi_name": first.get("roi_name", pd.NA),
                    "roi_color": first.get("roi_color", pd.NA),
                    "basis_name": str(task.basis),
                    "point_set": str(task.point_set),
                    "scope_id": first.get("scope_id", scope_id),
                    "group_axis": group_axis,
                    "line_center_px": center,
                    "line_start_x_px": float(start[0]),
                    "line_start_y_px": float(start[1]),
                    "line_end_x_px": float(end[0]),
                    "line_end_y_px": float(end[1]),
                    "line_label_x_px": float(label[0]),
                    "line_label_y_px": float(label[1]),
                    "line_point_count": int(len(line)),
                    "status": "ok",
                }
            )
            line_id += 1
    return pd.DataFrame(records)


def compute_line_guides_and_segments(
    points: pd.DataFrame,
    basis_vector_table: pd.DataFrame,
    task: LineGuideTask,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    guides = compute_line_guides(points, basis_vector_table, task)
    if not task.generate_consecutive_segments or guides.empty:
        return guides, _empty_segments()
    basis = _basis_row(basis_vector_table, task.basis)
    candidates = _filter_points_v2(
        points,
        point_set=task.point_set,
        roi_ids=task.roi_ids,
        class_ids=task.class_ids,
        class_names=task.class_names,
    )
    group_axis = str(task.group_axis).lower()
    sort_axis = "s" if group_axis == "t" else "t"
    records: list[dict[str, Any]] = []
    ux, uy, vx, vy = (float(basis[key]) for key in ("ux", "uy", "vx", "vy"))
    for _, guide in guides.iterrows():
        group = candidates.loc[candidates["scope_id"].astype(str) == str(guide["scope_id"])].copy()
        group["s_coord_px"] = group["x_px"] * ux + group["y_px"] * uy
        group["t_coord_px"] = group["x_px"] * vx + group["y_px"] * vy
        line = group.loc[
            (group[f"{group_axis}_coord_px"] - float(guide["line_center_px"])).abs() <= float(task.line_tolerance_px)
        ].sort_values(f"{sort_axis}_coord_px")
        if len(line) < 2:
            continue
        rows = list(line.iterrows())
        for index in range(len(rows) - 1):
            source = rows[index][1]
            target = rows[index + 1][1]
            distance = float(np.hypot(float(target["x_px"] - source["x_px"]), float(target["y_px"] - source["y_px"])))
            status = "ok"
            if task.max_in_line_gap_px is not None and distance > float(task.max_in_line_gap_px):
                status = "gap_exceeds_max"
            records.append(
                _segment_record(
                    task_name=task.name,
                    task_type="line_consecutive",
                    source=source,
                    target=target,
                    basis=basis,
                    line_id=int(guide["line_id"]),
                    status=status,
                )
            )
    return guides, _finalize_segments(records, task.name)


def make_pair_center_points(
    pair_segments: pd.DataFrame,
    *,
    pair_center_name: str | None = None,
    class_name: str | None = None,
    class_id: int | None = None,
    class_color: str | None = None,
) -> pd.DataFrame:
    if pair_segments is None or pair_segments.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in pair_segments.iterrows():
        pair_name = pair_center_name or str(row.get("task_name", row.get("pair_name", "pair")))
        source_point_id = str(row.get("source_point_id", f"atom:{row.get('source_atom_id', 'na')}"))
        target_point_id = str(row.get("target_point_id", f"atom:{row.get('target_atom_id', 'na')}"))
        source_x = float(row["source_x_px"])
        source_y = float(row["source_y_px"])
        target_x = float(row["target_x_px"])
        target_y = float(row["target_y_px"])
        source_x_nm = row.get("source_x_nm", np.nan)
        source_y_nm = row.get("source_y_nm", np.nan)
        target_x_nm = row.get("target_x_nm", np.nan)
        target_y_nm = row.get("target_y_nm", np.nan)
        if np.isfinite(pd.to_numeric(pd.Series([source_x_nm, source_y_nm, target_x_nm, target_y_nm]), errors="coerce")).all():
            x_nm = (float(source_x_nm) + float(target_x_nm)) / 2.0
            y_nm = (float(source_y_nm) + float(target_y_nm)) / 2.0
        else:
            x_nm = np.nan
            y_nm = np.nan
        roi_id = row.get("roi_id", "global")
        rows.append(
            {
                "point_id": f"pair_center:{pair_name}:{source_point_id}->{target_point_id}",
                "source_type": "pair_center",
                "point_set": "pair_centers",
                "atom_id": pd.NA,
                "parent_source_point_id": source_point_id,
                "parent_target_point_id": target_point_id,
                "parent_source_atom_id": row.get("source_atom_id", pd.NA),
                "parent_target_atom_id": row.get("target_atom_id", pd.NA),
                "x_px": (source_x + target_x) / 2.0,
                "y_px": (source_y + target_y) / 2.0,
                "x_nm": x_nm,
                "y_nm": y_nm,
                "class_id": class_id if class_id is not None else pd.NA,
                "class_name": class_name or f"pair_center:{pair_name}",
                "class_color": class_color if class_color is not None else pd.NA,
                "column_role": pd.NA,
                "keep": True,
                "quality_score": np.nan,
                "roi_id": roi_id,
                "roi_name": row.get("roi_name", roi_id),
                "roi_color": row.get("roi_color", pd.NA),
                "scope_id": f"{roi_id}:pair_centers",
                "source_table": row.get("source_table", "measurement_segments"),
            }
        )
    return pd.DataFrame(rows)


def combine_analysis_points(*point_tables: pd.DataFrame) -> pd.DataFrame:
    tables = [table for table in point_tables if table is not None and not table.empty]
    if not tables:
        return pd.DataFrame()
    result = pd.concat(tables, ignore_index=True, sort=False)
    for table in tables:
        pixel_to_nm = getattr(table, "attrs", {}).get("pixel_to_nm")
        if pixel_to_nm is not None:
            result.attrs["pixel_to_nm"] = pixel_to_nm
            break
    return result


def run_measurement_tasks(
    points: pd.DataFrame,
    basis_vector_table: pd.DataFrame,
    tasks: list[Any] | tuple[Any, ...],
) -> dict[str, pd.DataFrame]:
    segment_tables: list[pd.DataFrame] = []
    line_guide_tables: list[pd.DataFrame] = []
    pair_center_tables: list[pd.DataFrame] = []
    for task in tasks or []:
        if isinstance(task, NearestForwardTask):
            table = compute_nearest_forward_segments(points, basis_vector_table, task)
            segment_tables.append(table)
        elif isinstance(task, PairSegmentTask):
            table = compute_pair_segments(points, basis_vector_table, task)
            segment_tables.append(table)
            if task.create_pair_centers:
                pair_center_tables.append(
                    make_pair_center_points(table, class_name=task.pair_center_class_name)
                )
        elif isinstance(task, PeriodicVectorTask):
            segment_tables.append(compute_periodic_vector_segments(points, basis_vector_table, task))
        elif isinstance(task, LineGuideTask):
            guides, segments = compute_line_guides_and_segments(points, basis_vector_table, task)
            line_guide_tables.append(guides)
            segment_tables.append(segments)
        else:
            raise TypeError(f"Unsupported measurement task type: {type(task).__name__}")
    measurement_segments = (
        pd.concat([table for table in segment_tables if table is not None and not table.empty], ignore_index=True)
        if segment_tables
        else _empty_segments()
    )
    line_guides = (
        pd.concat([table for table in line_guide_tables if table is not None and not table.empty], ignore_index=True)
        if line_guide_tables
        else pd.DataFrame()
    )
    pair_center_points = (
        pd.concat([table for table in pair_center_tables if table is not None and not table.empty], ignore_index=True)
        if pair_center_tables
        else pd.DataFrame()
    )
    return {
        "measurement_segments": measurement_segments,
        "line_guides": line_guides,
        "pair_center_points": pair_center_points,
    }


def summarize_simple_quant_table(
    table: pd.DataFrame,
    group_columns: list[str] | tuple[str, ...] | None,
    value_column: str,
) -> pd.DataFrame:
    columns = list(group_columns or [])
    summary_columns = columns + ["count", "mean", "std", "median", "min", "max"]
    if table is None or table.empty or value_column not in table.columns:
        return pd.DataFrame(columns=summary_columns)
    data = table.copy()
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data = data.loc[data[value_column].notna()]
    if data.empty:
        return pd.DataFrame(columns=summary_columns)
    if columns:
        result = (
            data.groupby(columns, dropna=False)[value_column]
            .agg(count="count", mean="mean", std="std", median="median", min="min", max="max")
            .reset_index()
        )
    else:
        stats = data[value_column].agg(["count", "mean", "std", "median", "min", "max"])
        result = pd.DataFrame([stats.to_dict()])
    return result[summary_columns]


def _as_tuple_or_none(values: Any) -> tuple[Any, ...] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return (values,)
    try:
        return tuple(values)
    except TypeError:
        return (values,)


def _clean_class_ids(values: Any) -> tuple[int, ...] | None:
    values = _as_tuple_or_none(values)
    if values is None:
        return None
    result = tuple(int(value) for value in values if pd.notna(value))
    return result or None


def _class_selection_label(class_ids: tuple[int, ...] | None, class_names: tuple[str, ...] | None = None) -> str:
    if class_ids:
        return "+".join(f"class_id:{int(value)}" for value in class_ids)
    if class_names:
        return "+".join(f"class_name:{value}" for value in class_names)
    return "all_classes"


def _class_group_records(
    points: pd.DataFrame,
    *,
    roi_id: str,
    selected_class_ids: tuple[int, ...] | None = None,
    selected_class_names: tuple[str, ...] | None = None,
    class_group_mode: str = "per_class",
) -> list[dict[str, Any]]:
    roi_points = _filter_points_v2(
        points,
        point_set="atoms",
        roi_ids=(str(roi_id),),
        class_ids=selected_class_ids,
        class_names=selected_class_names,
    )
    if roi_points.empty:
        return [
            {
                "class_ids": selected_class_ids,
                "class_names": selected_class_names,
                "class_selection": _class_selection_label(selected_class_ids, selected_class_names),
                "n_points": 0,
            }
        ]
    mode = str(class_group_mode).lower()
    if mode not in {"per_class", "union"}:
        raise ValueError("class_group_mode must be 'per_class' or 'union'.")
    if mode == "union":
        return [
            {
                "class_ids": selected_class_ids,
                "class_names": selected_class_names,
                "class_selection": _class_selection_label(selected_class_ids, selected_class_names),
                "n_points": int(len(roi_points)),
            }
        ]

    records: list[dict[str, Any]] = []
    if selected_class_ids is not None or "class_id" in roi_points.columns:
        class_ids = selected_class_ids
        if class_ids is None:
            class_ids = tuple(int(value) for value in sorted(roi_points["class_id"].dropna().unique()))
        for class_id in class_ids:
            count = int((roi_points["class_id"] == int(class_id)).sum()) if "class_id" in roi_points.columns else 0
            records.append(
                {
                    "class_ids": (int(class_id),),
                    "class_names": None,
                    "class_selection": _class_selection_label((int(class_id),), None),
                    "n_points": count,
                }
            )
    elif selected_class_names is not None or "class_name" in roi_points.columns:
        class_names = selected_class_names
        if class_names is None:
            class_names = tuple(str(value) for value in sorted(roi_points["class_name"].dropna().astype(str).unique()))
        for class_name in class_names:
            count = int((roi_points["class_name"].astype(str) == str(class_name)).sum())
            records.append(
                {
                    "class_ids": None,
                    "class_names": (str(class_name),),
                    "class_selection": _class_selection_label(None, (str(class_name),)),
                    "n_points": count,
                }
            )
    return records or [
        {
            "class_ids": selected_class_ids,
            "class_names": selected_class_names,
            "class_selection": _class_selection_label(selected_class_ids, selected_class_names),
            "n_points": int(len(roi_points)),
        }
    ]


def select_points_by_roi_and_class(
    points: pd.DataFrame,
    *,
    roi_ids: tuple[str, ...] | list[str] | set[str] | None = None,
    class_ids: tuple[int, ...] | list[int] | set[int] | None = None,
    class_names: tuple[str, ...] | list[str] | set[str] | None = None,
    point_set: str | None = "atoms",
) -> pd.DataFrame:
    """Return a copy of points filtered by task-local ROI and class settings."""

    return _filter_points_v2(
        points,
        point_set=point_set,
        roi_ids=None if roi_ids is None else tuple(str(value) for value in roi_ids),
        class_ids=None if class_ids is None else tuple(int(value) for value in class_ids),
        class_names=None if class_names is None else tuple(str(value) for value in class_names),
    )


def angle_delta_deg(angle_deg: float | np.ndarray, target_angle_deg: float | np.ndarray) -> float | np.ndarray:
    """Wrapped signed angular difference in degrees, returned in [-180, 180)."""

    return (np.asarray(angle_deg, dtype=float) - np.asarray(target_angle_deg, dtype=float) + 180.0) % 360.0 - 180.0


def _robust_std(values: pd.Series | np.ndarray) -> float:
    data = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if data.size == 0:
        return np.nan
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    return 1.4826 * mad


def _sem(values: pd.Series | np.ndarray) -> float:
    data = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if data.size <= 1:
        return np.nan
    return float(np.std(data, ddof=1) / np.sqrt(data.size))


def _period_segment_columns() -> list[str]:
    return [
        "roi_id",
        "roi_name",
        "direction",
        "class_selection",
        "p0_id",
        "p1_id",
        "p0_x",
        "p0_y",
        "p1_x",
        "p1_y",
        "length_px",
        "length_nm",
        "length_A",
        "length_pm",
        "segment_angle_deg",
        "target_angle_deg",
        "angle_delta_deg",
        "valid",
        "invalid_reason",
        "basis_name",
        "task_name",
        "chain_id",
        "period_index",
        "period_residual_px",
        "source_class_id",
        "target_class_id",
        "source_class_name",
        "target_class_name",
    ]


def _normalize_period_segments(
    segments: pd.DataFrame,
    *,
    direction: str,
    class_selection: str,
    target_angle_deg: float,
) -> pd.DataFrame:
    columns = _period_segment_columns()
    if segments is None or segments.empty:
        return pd.DataFrame(columns=columns)
    table = pd.DataFrame(
        {
            "roi_id": segments.get("roi_id", pd.Series(pd.NA, index=segments.index)),
            "roi_name": segments.get("roi_name", pd.Series(pd.NA, index=segments.index)),
            "direction": str(direction),
            "class_selection": str(class_selection),
            "p0_id": segments.get("source_point_id", pd.Series(pd.NA, index=segments.index)),
            "p1_id": segments.get("target_point_id", pd.Series(pd.NA, index=segments.index)),
            "p0_x": pd.to_numeric(segments.get("source_x_px", np.nan), errors="coerce"),
            "p0_y": pd.to_numeric(segments.get("source_y_px", np.nan), errors="coerce"),
            "p1_x": pd.to_numeric(segments.get("target_x_px", np.nan), errors="coerce"),
            "p1_y": pd.to_numeric(segments.get("target_y_px", np.nan), errors="coerce"),
            "length_px": pd.to_numeric(segments.get("distance_px", np.nan), errors="coerce"),
            "length_nm": pd.to_numeric(segments.get("distance_nm", np.nan), errors="coerce"),
            "basis_name": segments.get("basis_name", pd.Series(pd.NA, index=segments.index)),
            "task_name": segments.get("task_name", pd.Series(pd.NA, index=segments.index)),
            "chain_id": segments.get("chain_id", pd.Series(pd.NA, index=segments.index)),
            "period_index": segments.get("period_index", pd.Series(pd.NA, index=segments.index)),
            "period_residual_px": pd.to_numeric(segments.get("period_residual_px", np.nan), errors="coerce"),
            "source_class_id": segments.get("source_class_id", pd.Series(pd.NA, index=segments.index)),
            "target_class_id": segments.get("target_class_id", pd.Series(pd.NA, index=segments.index)),
            "source_class_name": segments.get("source_class_name", pd.Series(pd.NA, index=segments.index)),
            "target_class_name": segments.get("target_class_name", pd.Series(pd.NA, index=segments.index)),
        }
    )
    table["length_A"] = table["length_nm"] * 10.0
    table["length_pm"] = table["length_nm"] * 1000.0
    table["segment_angle_deg"] = np.degrees(np.arctan2(table["p1_y"] - table["p0_y"], table["p1_x"] - table["p0_x"]))
    table["target_angle_deg"] = float(target_angle_deg)
    table["angle_delta_deg"] = angle_delta_deg(table["segment_angle_deg"].to_numpy(dtype=float), float(target_angle_deg))
    status = segments.get("status", pd.Series("ok", index=segments.index)).astype(str)
    table["valid"] = status.eq("ok")
    table["invalid_reason"] = np.where(table["valid"], "", status)
    return table[columns]


def summarize_period_segments(period_segment_table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "roi_id",
        "roi_name",
        "direction",
        "class_selection",
        "n_segments",
        "length_mean_px",
        "length_median_px",
        "length_std_px",
        "length_robust_std_px",
        "length_sem_px",
        "length_min_px",
        "length_max_px",
        "length_mean_A",
        "length_median_A",
        "length_std_A",
        "length_robust_std_A",
        "length_sem_A",
        "length_min_A",
        "length_max_A",
        "angle_delta_mean_deg",
        "angle_delta_median_deg",
        "angle_delta_std_deg",
        "angle_delta_robust_std_deg",
    ]
    if period_segment_table is None or period_segment_table.empty:
        return pd.DataFrame(columns=columns)
    data = period_segment_table.copy()
    data = data.loc[data.get("valid", True).astype(bool)].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    for column in ("length_px", "length_A", "angle_delta_deg"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    rows: list[dict[str, Any]] = []
    for keys, group in data.groupby(["roi_id", "roi_name", "direction", "class_selection"], dropna=False, sort=False):
        length_px = group["length_px"].dropna()
        length_A = group["length_A"].dropna()
        angle = group["angle_delta_deg"].dropna()
        rows.append(
            {
                "roi_id": keys[0],
                "roi_name": keys[1],
                "direction": keys[2],
                "class_selection": keys[3],
                "n_segments": int(len(group)),
                "length_mean_px": float(length_px.mean()) if len(length_px) else np.nan,
                "length_median_px": float(length_px.median()) if len(length_px) else np.nan,
                "length_std_px": float(length_px.std(ddof=1)) if len(length_px) > 1 else np.nan,
                "length_robust_std_px": _robust_std(length_px),
                "length_sem_px": _sem(length_px),
                "length_min_px": float(length_px.min()) if len(length_px) else np.nan,
                "length_max_px": float(length_px.max()) if len(length_px) else np.nan,
                "length_mean_A": float(length_A.mean()) if len(length_A) else np.nan,
                "length_median_A": float(length_A.median()) if len(length_A) else np.nan,
                "length_std_A": float(length_A.std(ddof=1)) if len(length_A) > 1 else np.nan,
                "length_robust_std_A": _robust_std(length_A),
                "length_sem_A": _sem(length_A),
                "length_min_A": float(length_A.min()) if len(length_A) else np.nan,
                "length_max_A": float(length_A.max()) if len(length_A) else np.nan,
                "angle_delta_mean_deg": float(angle.mean()) if len(angle) else np.nan,
                "angle_delta_median_deg": float(angle.median()) if len(angle) else np.nan,
                "angle_delta_std_deg": float(angle.std(ddof=1)) if len(angle) > 1 else np.nan,
                "angle_delta_robust_std_deg": _robust_std(angle),
            }
        )
    return pd.DataFrame(rows)[columns]


def run_period_statistics_ab(
    analysis_points: pd.DataFrame,
    basis_vector_table: pd.DataFrame,
    roi_basis_table: pd.DataFrame,
    *,
    roi_class_selection: dict[str, Any] | None = None,
    basis_roles: tuple[str, ...] = ("a", "b"),
    class_group_mode: str = "per_class",
    match_radius_fraction: float = 0.30,
    match_radius_px: float | None = None,
    one_to_one: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run ROI-resolved a/b periodic-vector statistics with per-task class groups."""

    if roi_basis_table is None or roi_basis_table.empty:
        raise ValueError("roi_basis_table is empty; choose or define Task 1 basis vectors first.")
    selection = dict(roi_class_selection or {})
    segment_tables: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for _, basis_map in roi_basis_table.iterrows():
        if str(basis_map.get("basis_role")) not in {str(role) for role in basis_roles}:
            continue
        if not bool(basis_map.get("found", False)):
            continue
        roi_id = str(basis_map["roi_id"])
        direction = str(basis_map["basis_role"])
        basis_name = str(basis_map["basis_name"])
        basis = _basis_row(basis_vector_table, basis_name)
        roi_selection = selection.get(roi_id, selection.get("default", None))
        if isinstance(roi_selection, dict):
            selected_class_ids = _clean_class_ids(roi_selection.get("class_ids"))
            selected_class_names = (
                tuple(str(value) for value in _as_tuple_or_none(roi_selection.get("class_names")) or ())
                or None
            )
        else:
            selected_class_ids = _clean_class_ids(roi_selection)
            selected_class_names = None
        class_groups = _class_group_records(
            analysis_points,
            roi_id=roi_id,
            selected_class_ids=selected_class_ids,
            selected_class_names=selected_class_names,
            class_group_mode=class_group_mode,
        )
        for class_group in class_groups:
            class_ids = class_group["class_ids"]
            class_names = class_group["class_names"]
            class_selection = class_group["class_selection"]
            task_name = f"task1A_{roi_id}_{direction}_{str(class_selection).replace(':', '_').replace('+', '_')}"
            task = PeriodicVectorTask(
                name=task_name,
                basis=basis_name,
                point_set="atoms",
                roi_ids=(roi_id,),
                class_ids=class_ids,
                class_names=class_names,
                match_radius_fraction=match_radius_fraction,
                match_radius_px=match_radius_px,
                one_to_one=one_to_one,
            )
            raw_segments = compute_periodic_vector_segments(analysis_points, basis_vector_table, task)
            period_segments = _normalize_period_segments(
                raw_segments,
                direction=direction,
                class_selection=class_selection,
                target_angle_deg=float(basis["angle_deg"]),
            )
            segment_tables.append(period_segments)
            selection_rows.append(
                {
                    "roi_id": roi_id,
                    "roi_name": basis_map.get("roi_name", roi_id),
                    "direction": direction,
                    "class_selection": class_selection,
                    "class_group_mode": class_group_mode,
                    "class_ids": class_ids,
                    "class_names": class_names,
                    "n_points": int(class_group.get("n_points", 0)),
                }
            )
            task_rows.append({"task_name": task_name, **task.__dict__})
    period_segment_table = (
        pd.concat(segment_tables, ignore_index=True)
        if any(table is not None and not table.empty for table in segment_tables)
        else pd.DataFrame(columns=_period_segment_columns())
    )
    period_summary_table = summarize_period_segments(period_segment_table)
    return {
        "period_segment_table": period_segment_table,
        "period_summary_table": period_summary_table,
        "roi_class_selection": pd.DataFrame(selection_rows),
        "task1A_tasks": pd.DataFrame(task_rows),
    }


def fit_single_gaussian_to_histogram(values: pd.Series | np.ndarray) -> dict[str, float]:
    data = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "robust_std": np.nan}
    std = float(np.std(data, ddof=1)) if data.size > 1 else 0.0
    return {
        "n": int(data.size),
        "mean": float(np.mean(data)),
        "std": std,
        "median": float(np.median(data)),
        "robust_std": _robust_std(data),
    }


def _pixel_to_nm_from_table(table: pd.DataFrame) -> float | None:
    value = getattr(table, "attrs", {}).get("pixel_to_nm")
    if value is not None and np.isfinite(float(value)) and float(value) > 0:
        return float(value)
    return _pixel_to_nm_from_points(table)


def _distance_A_to_px(distance_A: float | None, pixel_to_nm: float | None) -> float | None:
    if distance_A is None or pixel_to_nm is None:
        return None
    distance_A = float(distance_A)
    if not np.isfinite(distance_A) or distance_A <= 0.0:
        return None
    return distance_A * 0.1 / float(pixel_to_nm)


def find_strict_mutual_nearest_pairs(
    analysis_points: pd.DataFrame,
    *,
    roi_class_selection: dict[str, Any] | None = None,
    pair_mode: str = "within_class",
    source_class_ids: tuple[int, ...] | list[int] | None = None,
    target_class_ids: tuple[int, ...] | list[int] | None = None,
    max_pair_distance_px: float | None = None,
    max_pair_distance_A: float | None = None,
) -> pd.DataFrame:
    """Find strict mutual nearest-neighbor pairs for Task 2."""

    mode = str(pair_mode).lower()
    if mode not in {"within_class", "between_classes"}:
        raise ValueError("pair_mode must be 'within_class' or 'between_classes'.")
    pixel_to_nm = _pixel_to_nm_from_table(analysis_points)
    max_from_A_px = _distance_A_to_px(max_pair_distance_A, pixel_to_nm)
    thresholds = [value for value in (max_pair_distance_px, max_from_A_px) if value is not None]
    max_distance_px = min(float(value) for value in thresholds) if thresholds else None
    roi_ids = (
        tuple(str(value) for value in analysis_points["roi_id"].dropna().astype(str).unique())
        if "roi_id" in analysis_points.columns
        else ("global",)
    )
    selection = dict(roi_class_selection or {})
    rows: list[dict[str, Any]] = []
    pair_counter = 0

    def row_record(roi_id: str, pair_mode_value: str, p1: pd.Series, p2: pd.Series, valid: bool, reason: str) -> dict[str, Any]:
        nonlocal pair_counter
        dx = float(p2["x_px"] - p1["x_px"])
        dy = float(p2["y_px"] - p1["y_px"])
        distance_px = float(np.hypot(dx, dy))
        distance_nm = _pixel_to_nm_from_row_pair(p1, p2)
        record = {
            "roi_id": roi_id,
            "roi_name": p1.get("roi_name", roi_id),
            "pair_id": f"{roi_id}:pair:{pair_counter}",
            "pair_mode": pair_mode_value,
            "p1_id": _point_id(p1),
            "p2_id": _point_id(p2),
            "p1_class": p1.get("class_id", pd.NA),
            "p2_class": p2.get("class_id", pd.NA),
            "p1_x": float(p1["x_px"]),
            "p1_y": float(p1["y_px"]),
            "p2_x": float(p2["x_px"]),
            "p2_y": float(p2["y_px"]),
            "center_x": float((p1["x_px"] + p2["x_px"]) / 2.0),
            "center_y": float((p1["y_px"] + p2["y_px"]) / 2.0),
            "vector_x_px": dx,
            "vector_y_px": dy,
            "distance_px": distance_px,
            "distance_nm": distance_nm,
            "distance_A": distance_nm * 10.0 if np.isfinite(distance_nm) else np.nan,
            "pair_angle_deg": float(np.degrees(np.arctan2(dy, dx))),
            "valid": bool(valid),
            "invalid_reason": reason,
        }
        pair_counter += 1
        return record

    for roi_id in roi_ids:
        roi_selection = selection.get(roi_id, selection.get("default", None))
        selected_ids = _clean_class_ids(roi_selection if not isinstance(roi_selection, dict) else roi_selection.get("class_ids"))
        roi_points = select_points_by_roi_and_class(
            analysis_points,
            roi_ids=(roi_id,),
            class_ids=selected_ids,
            point_set="atoms",
        )
        if mode == "within_class":
            class_ids = selected_ids
            if class_ids is None:
                class_ids = tuple(int(value) for value in sorted(roi_points["class_id"].dropna().unique()))
            for class_id in class_ids:
                group = roi_points.loc[roi_points["class_id"] == int(class_id)].reset_index(drop=True)
                if len(group) < 2:
                    continue
                coords = group[["x_px", "y_px"]].to_numpy(dtype=float)
                distances, indices = cKDTree(coords).query(coords, k=2)
                nearest = indices[:, 1].astype(int)
                seen: set[tuple[str, str]] = set()
                for source_index, target_index in enumerate(nearest):
                    if int(nearest[target_index]) != int(source_index):
                        continue
                    p1 = group.iloc[source_index]
                    p2 = group.iloc[target_index]
                    key = tuple(sorted((_point_id(p1), _point_id(p2))))
                    if key in seen:
                        continue
                    seen.add(key)
                    distance_px = float(distances[source_index, 1])
                    valid = max_distance_px is None or distance_px <= max_distance_px
                    rows.append(row_record(roi_id, mode, p1, p2, valid, "" if valid else "too_far"))
        else:
            source_ids = _clean_class_ids(source_class_ids)
            target_ids = _clean_class_ids(target_class_ids)
            if source_ids is None or target_ids is None:
                raise ValueError("between_classes mode requires source_class_ids and target_class_ids.")
            sources = select_points_by_roi_and_class(analysis_points, roi_ids=(roi_id,), class_ids=source_ids, point_set="atoms").reset_index(drop=True)
            targets = select_points_by_roi_and_class(analysis_points, roi_ids=(roi_id,), class_ids=target_ids, point_set="atoms").reset_index(drop=True)
            if sources.empty or targets.empty:
                continue
            target_tree = cKDTree(targets[["x_px", "y_px"]].to_numpy(dtype=float))
            source_tree = cKDTree(sources[["x_px", "y_px"]].to_numpy(dtype=float))
            source_to_target_dist, source_to_target = target_tree.query(sources[["x_px", "y_px"]].to_numpy(dtype=float), k=1)
            _target_to_source_dist, target_to_source = source_tree.query(targets[["x_px", "y_px"]].to_numpy(dtype=float), k=1)
            seen_between: set[tuple[str, str]] = set()
            for source_index, target_index in enumerate(source_to_target.astype(int)):
                if int(target_to_source[target_index]) != int(source_index):
                    continue
                p1 = sources.iloc[source_index]
                p2 = targets.iloc[target_index]
                key = (_point_id(p1), _point_id(p2))
                if key in seen_between:
                    continue
                seen_between.add(key)
                distance_px = float(source_to_target_dist[source_index])
                valid = max_distance_px is None or distance_px <= max_distance_px
                rows.append(row_record(roi_id, mode, p1, p2, valid, "" if valid else "too_far"))
    return pd.DataFrame(rows)


def suggest_line_tolerance_from_projection(
    pair_table: pd.DataFrame,
    projection_vector: tuple[float, float],
) -> pd.DataFrame:
    if pair_table is None or pair_table.empty:
        return pd.DataFrame(columns=["roi_id", "n_centers", "spacing_median_px", "spacing_iqr_px", "suggested_line_tolerance_px"])
    vector = np.asarray(projection_vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("projection_vector must be a non-zero 2D vector.")
    unit = vector / norm
    data = pair_table.loc[pair_table.get("valid", True).astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for roi_id, group in data.groupby("roi_id", dropna=False, sort=False):
        centers = group[["center_x", "center_y"]].to_numpy(dtype=float)
        if len(centers) < 2:
            rows.append({"roi_id": roi_id, "n_centers": int(len(centers)), "spacing_median_px": np.nan, "spacing_iqr_px": np.nan, "suggested_line_tolerance_px": np.nan})
            continue
        s = centers @ unit
        gaps = np.diff(np.sort(s))
        gaps = gaps[np.isfinite(gaps) & (gaps > 1e-9)]
        if gaps.size == 0:
            suggested = np.nan
            median = np.nan
            iqr = np.nan
        else:
            q1, q3 = np.percentile(gaps, [25, 75])
            median = float(np.median(gaps))
            iqr = float(q3 - q1)
            suggested = float(max(q3, median + 0.5 * iqr))
        rows.append(
            {
                "roi_id": roi_id,
                "n_centers": int(len(centers)),
                "spacing_median_px": median,
                "spacing_iqr_px": iqr,
                "suggested_line_tolerance_px": suggested,
            }
        )
    return pd.DataFrame(rows)


def assign_pair_center_lines_by_projection(
    pair_table: pd.DataFrame,
    *,
    projection_vector: tuple[float, float],
    line_tolerance_px: float | None = None,
    min_pairs_per_line: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign valid pair centers to 1D lines along a projection axis."""

    if pair_table is None or pair_table.empty:
        return pd.DataFrame(), pd.DataFrame()
    vector = np.asarray(projection_vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("projection_vector must be a non-zero 2D vector.")
    unit = vector / norm
    suggestion = suggest_line_tolerance_from_projection(pair_table, tuple(vector))
    table = pair_table.copy()
    table["projection_s_px"] = np.nan
    table["line_id"] = pd.NA
    table["line_valid"] = False
    table["line_invalid_reason"] = ""
    summary_rows: list[dict[str, Any]] = []
    for roi_id, group in table.groupby("roi_id", dropna=False, sort=False):
        valid_index = group.index[group.get("valid", True).astype(bool)]
        valid = table.loc[valid_index].copy()
        if valid.empty:
            continue
        centers = valid[["center_x", "center_y"]].to_numpy(dtype=float)
        raw_s = centers @ unit
        origin = float(np.nanmin(raw_s))
        valid["projection_s_px"] = raw_s - origin
        tolerance = line_tolerance_px
        if tolerance is None:
            matched = suggestion.loc[suggestion["roi_id"].astype(str) == str(roi_id)]
            if not matched.empty and np.isfinite(float(matched.iloc[0]["suggested_line_tolerance_px"])):
                tolerance = float(matched.iloc[0]["suggested_line_tolerance_px"])
            else:
                tolerance = 3.0
        ordered = valid.sort_values("projection_s_px")
        raw_groups: list[list[int]] = []
        current: list[int] = []
        previous: float | None = None
        for index, row in ordered.iterrows():
            coord = float(row["projection_s_px"])
            if previous is None or coord - previous <= float(tolerance):
                current.append(index)
            else:
                raw_groups.append(current)
                current = [index]
            previous = coord
        if current:
            raw_groups.append(current)
        line_id = 1
        for raw_group in raw_groups:
            if len(raw_group) < int(min_pairs_per_line):
                table.loc[raw_group, "line_invalid_reason"] = "line_too_short"
                continue
            line_s = valid.loc[raw_group, "projection_s_px"]
            table.loc[raw_group, "line_id"] = int(line_id)
            table.loc[raw_group, "line_valid"] = True
            summary_rows.append(
                {
                    "roi_id": roi_id,
                    "line_id": int(line_id),
                    "n_pairs": int(len(raw_group)),
                    "projection_s_median": float(line_s.median()),
                    "line_tolerance_px": float(tolerance),
                }
            )
            line_id += 1
        table.loc[valid.index, "projection_s_px"] = valid["projection_s_px"]
    return table, pd.DataFrame(summary_rows)


def summarize_pair_lines_median_iqr(pair_line_table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "roi_id",
        "line_id",
        "n_pairs",
        "projection_s_median",
        "distance_median_A",
        "distance_q1_A",
        "distance_q3_A",
        "distance_iqr_A",
        "distance_median_px",
        "distance_q1_px",
        "distance_q3_px",
        "distance_iqr_px",
    ]
    if pair_line_table is None or pair_line_table.empty:
        return pd.DataFrame(columns=columns)
    data = pair_line_table.loc[pair_line_table.get("valid", True).astype(bool) & pair_line_table.get("line_valid", False).astype(bool)].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (roi_id, line_id), group in data.groupby(["roi_id", "line_id"], dropna=False, sort=True):
        dist_A = pd.to_numeric(group.get("distance_A", np.nan), errors="coerce").dropna()
        dist_px = pd.to_numeric(group.get("distance_px", np.nan), errors="coerce").dropna()
        q1_A, q3_A = (np.nan, np.nan) if dist_A.empty else np.percentile(dist_A, [25, 75])
        q1_px, q3_px = (np.nan, np.nan) if dist_px.empty else np.percentile(dist_px, [25, 75])
        rows.append(
            {
                "roi_id": roi_id,
                "line_id": int(line_id),
                "n_pairs": int(len(group)),
                "projection_s_median": float(pd.to_numeric(group["projection_s_px"], errors="coerce").median()),
                "distance_median_A": float(dist_A.median()) if len(dist_A) else np.nan,
                "distance_q1_A": float(q1_A) if np.isfinite(q1_A) else np.nan,
                "distance_q3_A": float(q3_A) if np.isfinite(q3_A) else np.nan,
                "distance_iqr_A": float(q3_A - q1_A) if np.isfinite(q1_A) and np.isfinite(q3_A) else np.nan,
                "distance_median_px": float(dist_px.median()) if len(dist_px) else np.nan,
                "distance_q1_px": float(q1_px) if np.isfinite(q1_px) else np.nan,
                "distance_q3_px": float(q3_px) if np.isfinite(q3_px) else np.nan,
                "distance_iqr_px": float(q3_px - q1_px) if np.isfinite(q1_px) and np.isfinite(q3_px) else np.nan,
            }
        )
    return pd.DataFrame(rows)[columns]


def compute_group_centroids_by_roi(
    analysis_points: pd.DataFrame,
    *,
    center_groups: dict[str, tuple[int, ...] | list[int]],
    roi_ids: tuple[str, ...] | list[str] | None = None,
    min_points: int = 1,
) -> pd.DataFrame:
    """Compute unweighted geometric centroids for Task 3 class groups."""

    if roi_ids is None:
        roi_ids = tuple(str(value) for value in analysis_points["roi_id"].dropna().astype(str).unique())
    rows: list[dict[str, Any]] = []
    for roi_id in roi_ids:
        roi_points = select_points_by_roi_and_class(analysis_points, roi_ids=(str(roi_id),), point_set="atoms")
        roi_name = roi_points["roi_name"].iloc[0] if not roi_points.empty and "roi_name" in roi_points.columns else str(roi_id)
        for group_name, class_ids_value in center_groups.items():
            class_ids = tuple(int(value) for value in class_ids_value)
            group = roi_points.loc[roi_points["class_id"].isin(class_ids)].copy() if "class_id" in roi_points.columns else pd.DataFrame()
            valid = len(group) >= int(min_points)
            rows.append(
                {
                    "roi_id": str(roi_id),
                    "roi_name": roi_name,
                    "group_name": str(group_name),
                    "class_ids": ",".join(str(value) for value in class_ids),
                    "n_points": int(len(group)),
                    "center_x": float(group["x_px"].mean()) if valid else np.nan,
                    "center_y": float(group["y_px"].mean()) if valid else np.nan,
                    "center_x_std": float(group["x_px"].std(ddof=1)) if len(group) > 1 else np.nan,
                    "center_y_std": float(group["y_px"].std(ddof=1)) if len(group) > 1 else np.nan,
                    "valid": bool(valid),
                    "invalid_reason": "" if valid else "not_enough_points",
                }
            )
    return pd.DataFrame(rows)


def compute_group_pair_displacements(
    group_centroid_table: pd.DataFrame,
    *,
    center_pairs: tuple[tuple[str, str], ...] | list[tuple[str, str]],
    pixel_to_nm: float | None = None,
) -> pd.DataFrame:
    columns = [
        "roi_id",
        "roi_name",
        "group_A",
        "group_B",
        "center_A_x",
        "center_A_y",
        "center_B_x",
        "center_B_y",
        "dx_px",
        "dy_px",
        "distance_px",
        "distance_A",
        "angle_deg",
        "valid",
        "invalid_reason",
    ]
    if group_centroid_table is None or group_centroid_table.empty:
        return pd.DataFrame(columns=columns)
    if pixel_to_nm is None:
        pixel_to_nm = _pixel_to_nm_from_table(group_centroid_table)
    rows: list[dict[str, Any]] = []
    for roi_id, group in group_centroid_table.groupby("roi_id", dropna=False, sort=False):
        roi_name = group["roi_name"].iloc[0] if "roi_name" in group.columns and len(group) else roi_id
        indexed = group.set_index("group_name", drop=False)
        for group_A, group_B in center_pairs:
            reason = ""
            valid = str(group_A) in indexed.index and str(group_B) in indexed.index
            if valid:
                row_A = indexed.loc[str(group_A)]
                row_B = indexed.loc[str(group_B)]
                if isinstance(row_A, pd.DataFrame):
                    row_A = row_A.iloc[0]
                if isinstance(row_B, pd.DataFrame):
                    row_B = row_B.iloc[0]
                valid = bool(row_A.get("valid", False)) and bool(row_B.get("valid", False))
                if not valid:
                    reason = "invalid_group_centroid"
            else:
                row_A = pd.Series(dtype=object)
                row_B = pd.Series(dtype=object)
                reason = "missing_group"
            if valid:
                dx = float(row_B["center_x"] - row_A["center_x"])
                dy = float(row_B["center_y"] - row_A["center_y"])
                distance_px = float(np.hypot(dx, dy))
            else:
                dx = dy = distance_px = np.nan
            distance_A = distance_px * pixel_to_nm * 10.0 if pixel_to_nm is not None and np.isfinite(distance_px) else np.nan
            rows.append(
                {
                    "roi_id": roi_id,
                    "roi_name": roi_name,
                    "group_A": str(group_A),
                    "group_B": str(group_B),
                    "center_A_x": row_A.get("center_x", np.nan),
                    "center_A_y": row_A.get("center_y", np.nan),
                    "center_B_x": row_B.get("center_x", np.nan),
                    "center_B_y": row_B.get("center_y", np.nan),
                    "dx_px": dx,
                    "dy_px": dy,
                    "distance_px": distance_px,
                    "distance_A": distance_A,
                    "angle_deg": float(np.degrees(np.arctan2(dy, dx))) if valid else np.nan,
                    "valid": bool(valid),
                    "invalid_reason": reason,
                }
            )
    return pd.DataFrame(rows)[columns]
