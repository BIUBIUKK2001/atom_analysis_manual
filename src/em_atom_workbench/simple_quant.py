from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
