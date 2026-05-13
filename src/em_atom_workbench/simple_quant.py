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
    vector = np.asarray([float(basis["vector_x_px"]), float(basis["vector_y_px"])], dtype=float)
    basis_length = float(basis["length_px"])
    match_radius = float(task.match_radius_px) if task.match_radius_px is not None else basis_length * float(task.match_radius_fraction)
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
