from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .session import AnalysisSession

_ALLOWED_REFERENCE_SOURCES = {"curated", "refined", "candidate"}
_MODE_ALIASES = {
    "manual": "manual_basis",
    "manual_basis": "manual_basis",
    "global_median": "median_local_basis",
    "median_local_basis": "median_local_basis",
}
_ALLOWED_REFERENCE_MODES = set(_MODE_ALIASES)
_ALLOWED_COORDINATE_UNITS = {"px", "calibrated"}
_ALLOWED_LATTICE_UNITS = {"px", "nm"}
_SOURCE_TABLES = {
    "curated": "curated_points",
    "refined": "refined_points",
    "candidate": "candidate_points",
}
_SINGULAR_TOLERANCE = 1e-12
_BASIS_VECTOR_COLUMNS = ["basis_a_x", "basis_a_y", "basis_b_x", "basis_b_y"]
_CANONICAL_BASIS_CONVENTION = "canonical_reduced_columns_are_a_b_vectors_in_xy"
_CLUSTER_FEATURE_MODE = "log_lengths_folded_gamma_orientation_axis"


def _coerce_manual_basis(value: Any) -> np.ndarray:
    basis_vectors = np.asarray(value, dtype=float)
    if basis_vectors.shape != (2, 2):
        raise ValueError("manual_basis 必须包含两个二维基矢量。")
    if not np.isfinite(basis_vectors).all():
        raise ValueError("manual_basis 必须全部为有限数值。")
    basis = np.column_stack((basis_vectors[0], basis_vectors[1]))
    determinant = float(np.linalg.det(basis))
    if not np.isfinite(determinant) or abs(determinant) <= _SINGULAR_TOLERANCE:
        raise ValueError("manual_basis 必须表示非奇异的二维参考基。")
    return basis


def _coerce_origin(value: Any, field_name: str = "manual_origin") -> np.ndarray:
    origin = np.asarray(value, dtype=float)
    if origin.shape != (2,):
        raise ValueError(f"{field_name} 必须是二维坐标。")
    if not np.isfinite(origin).all():
        raise ValueError(f"{field_name} 必须全部为有限数值。")
    return origin


def _calibration_scale_to_nm(session: AnalysisSession) -> float:
    calibration = getattr(session, "pixel_calibration", None)
    size = getattr(calibration, "size", None)
    unit = str(getattr(calibration, "unit", "") or "").strip().lower()
    if size is None:
        raise ValueError("局域仿射应变需要显式 nm 坐标或有效像素标定；当前缺少 pixel_calibration.size。")
    size = float(size)
    if not np.isfinite(size) or size <= 0.0:
        raise ValueError("局域仿射应变需要有效像素标定；pixel_calibration.size 必须为正的有限数值。")

    if unit in {"nm", "nanometer", "nanometers"}:
        return size
    if unit in {"a", "å", "angstrom", "angstroms", "ångström", "ångströms"}:
        return size * 0.1
    if unit in {"pm", "picometer", "picometers"}:
        return size * 0.001
    raise ValueError(
        "局域仿射应变需要显式 nm 坐标或可转换到 nm 的像素标定；"
        f"当前 pixel_calibration.unit={getattr(calibration, 'unit', None)!r} 不受支持。"
    )


def _atom_ids_for(table: pd.DataFrame) -> pd.Series:
    if "atom_id" in table.columns:
        return table["atom_id"].reset_index(drop=True)
    return pd.Series(np.arange(len(table), dtype=int), name="atom_id")


def _copy_optional_atom_columns(target: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    optional_columns = ["role", "column_role", "channel", "seed_channel", "keep"]
    optional_columns.extend(column for column in table.columns if str(column).startswith("flag_"))
    for column in optional_columns:
        if column in table.columns and column not in target.columns:
            target[column] = table[column].reset_index(drop=True)
    return target


def _get_atom_coordinate_table(session: AnalysisSession, source: str) -> pd.DataFrame:
    if source not in _SOURCE_TABLES:
        raise ValueError(f"source 必须是 {sorted(_SOURCE_TABLES)} 之一。")
    table_name = _SOURCE_TABLES[source]
    table = getattr(session, table_name, None)
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError(f"source='{source}' 对应的 {table_name} 为空，无法构建参考晶格。")
    return table.copy()


def _extract_xy_nm(session: AnalysisSession, table: pd.DataFrame, source: str) -> pd.DataFrame:
    if {"x_nm", "y_nm"}.issubset(table.columns):
        x_nm = table["x_nm"].to_numpy(dtype=float)
        y_nm = table["y_nm"].to_numpy(dtype=float)
    elif {"x_px", "y_px"}.issubset(table.columns):
        scale_nm_per_px = _calibration_scale_to_nm(session)
        x_nm = table["x_px"].to_numpy(dtype=float) * scale_nm_per_px
        y_nm = table["y_px"].to_numpy(dtype=float) * scale_nm_per_px
    else:
        raise ValueError(
            f"source='{source}' 缺少 x_nm/y_nm，也缺少可通过像素标定转换的 x_px/y_px 坐标。"
        )

    if not np.isfinite(x_nm).all() or not np.isfinite(y_nm).all():
        raise ValueError(f"source='{source}' 的 nm 坐标必须全部为有限数值。")

    result = pd.DataFrame(
        {
            "atom_id": _atom_ids_for(table),
            "x_nm": x_nm,
            "y_nm": y_nm,
        }
    )
    if {"x_px", "y_px"}.issubset(table.columns):
        result["x_px"] = table["x_px"].reset_index(drop=True)
        result["y_px"] = table["y_px"].reset_index(drop=True)
    return _copy_optional_atom_columns(result, table)


def _extract_xy_px(table: pd.DataFrame, source: str) -> pd.DataFrame:
    if not {"x_px", "y_px"}.issubset(table.columns):
        raise ValueError(f"source='{source}' 缺少 x_px/y_px；默认像素路径不能使用其他单位。")
    x_px = table["x_px"].to_numpy(dtype=float)
    y_px = table["y_px"].to_numpy(dtype=float)
    if not np.isfinite(x_px).all() or not np.isfinite(y_px).all():
        raise ValueError(f"source='{source}' 的 px 坐标必须全部为有限数值。")

    result = pd.DataFrame(
        {
            "atom_id": _atom_ids_for(table),
            "x": x_px,
            "y": y_px,
            "x_px": x_px,
            "y_px": y_px,
        }
    )
    return _copy_optional_atom_columns(result, table)


def _extract_coordinate_table(
    session: AnalysisSession,
    table: pd.DataFrame,
    source: str,
    coordinate_unit: str,
) -> tuple[pd.DataFrame, str]:
    if coordinate_unit == "px":
        return _extract_xy_px(table, source), "px"
    if coordinate_unit == "calibrated":
        nm_table = _extract_xy_nm(session, table, source)
        result = nm_table.copy()
        result["x"] = result["x_nm"]
        result["y"] = result["y_nm"]
        return result, "nm"
    raise ValueError(f"coordinate_unit 必须是 {sorted(_ALLOWED_COORDINATE_UNITS)} 之一。")


def _role_column(table: pd.DataFrame) -> str | None:
    if "role" in table.columns:
        return "role"
    if "column_role" in table.columns:
        return "column_role"
    return None


def _keep_mask(values: pd.Series) -> pd.Series:
    def is_keep(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        if pd.isna(value):
            return False
        try:
            return bool(int(value))
        except (TypeError, ValueError):
            return bool(value)

    return values.map(is_keep)


def _select_reference_points(
    session: AnalysisSession,
    config: "ReferenceLatticeConfig",
) -> tuple[pd.DataFrame, str]:
    source_table = _get_atom_coordinate_table(session, config.source)
    coordinates, unit = _extract_coordinate_table(session, source_table, config.source, config.coordinate_unit)

    selected = coordinates.copy()
    if config.role_filter is not None:
        role_column = _role_column(selected)
        if role_column is None:
            raise ValueError("已指定 role_filter/atom_role，但坐标表中没有 role 或 column_role 列。")
        selected = selected[selected[role_column].astype(str) == str(config.role_filter)]

    if config.use_keep and "keep" in selected.columns:
        selected = selected[_keep_mask(selected["keep"])]

    selected = selected.reset_index(drop=True)
    if len(selected) < int(config.min_points):
        raise ValueError(
            "参考晶格构建需要至少 "
            f"{int(config.min_points)} 个筛选后的原子点；当前仅有 {len(selected)} 个。"
        )
    return selected, unit


def _ensure_basis_atom_ids(rows: pd.DataFrame, atom_ids: pd.Series) -> pd.DataFrame:
    rows = rows.copy()
    if "atom_id" in rows.columns:
        return rows
    atom_ids = pd.Series(atom_ids).reset_index(drop=True)
    if len(rows) != len(atom_ids):
        return pd.DataFrame()
    rows = rows.reset_index(drop=True)
    rows.insert(0, "atom_id", atom_ids)
    return rows


def _basis_rows_from_explicit_columns_with_atom_id(basis_table: pd.DataFrame, atom_ids: pd.Series) -> pd.DataFrame:
    if not set(_BASIS_VECTOR_COLUMNS).issubset(basis_table.columns):
        return pd.DataFrame()
    rows = _ensure_basis_atom_ids(basis_table, atom_ids)
    if rows.empty:
        return pd.DataFrame()
    rows = rows[rows["atom_id"].isin(set(atom_ids))]
    rows = rows.dropna(subset=_BASIS_VECTOR_COLUMNS)
    finite_mask = np.isfinite(rows[_BASIS_VECTOR_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    return rows.loc[finite_mask, ["atom_id", *_BASIS_VECTOR_COLUMNS]].copy()


def _valid_basis_rows_from_explicit_columns(basis_table: pd.DataFrame, atom_ids: pd.Series) -> pd.DataFrame:
    rows = _basis_rows_from_explicit_columns_with_atom_id(basis_table, atom_ids)
    if rows.empty:
        return pd.DataFrame()
    return rows[_BASIS_VECTOR_COLUMNS].copy()


def _basis_rows_from_local_metrics_with_atom_id(local_metrics: pd.DataFrame, atom_ids: pd.Series) -> pd.DataFrame:
    required = ["basis_a_length_px", "basis_b_length_px", "basis_angle_deg", "local_orientation_deg"]
    if not set(required).issubset(local_metrics.columns):
        return pd.DataFrame()
    rows = _ensure_basis_atom_ids(local_metrics, atom_ids)
    if rows.empty:
        return pd.DataFrame()
    rows = rows[rows["atom_id"].isin(set(atom_ids))]
    rows = rows.dropna(subset=required)
    values = rows[required].to_numpy(dtype=float)
    if len(values) == 0:
        return pd.DataFrame()
    finite_mask = np.isfinite(values).all(axis=1)
    rows = rows.loc[finite_mask].copy()
    values = values[finite_mask]
    if len(values) == 0:
        return pd.DataFrame()

    a_length = values[:, 0]
    b_length = values[:, 1]
    basis_angle = np.deg2rad(values[:, 2])
    orientation = np.deg2rad(values[:, 3])
    basis_a_x = a_length * np.cos(orientation)
    basis_a_y = a_length * np.sin(orientation)
    basis_b_x = b_length * np.cos(orientation + basis_angle)
    basis_b_y = b_length * np.sin(orientation + basis_angle)
    return pd.DataFrame(
        {
            "atom_id": rows["atom_id"].to_numpy(),
            "basis_a_x": basis_a_x,
            "basis_a_y": basis_a_y,
            "basis_b_x": basis_b_x,
            "basis_b_y": basis_b_y,
        }
    )


def _basis_rows_from_local_metrics(local_metrics: pd.DataFrame, atom_ids: pd.Series) -> pd.DataFrame:
    rows = _basis_rows_from_local_metrics_with_atom_id(local_metrics, atom_ids)
    if rows.empty:
        return pd.DataFrame()
    return rows[_BASIS_VECTOR_COLUMNS].copy()


def _robust_median_basis(basis_rows: pd.DataFrame, config: "ReferenceLatticeConfig") -> np.ndarray:
    values = basis_rows[["basis_a_x", "basis_a_y", "basis_b_x", "basis_b_y"]].to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("没有可用于 global_median 的有效局域基矢量。")

    low = float(config.robust_percentile_low)
    high = float(config.robust_percentile_high)
    lower = np.percentile(values, low, axis=0)
    upper = np.percentile(values, high, axis=0)
    mask = ((values >= lower) & (values <= upper)).all(axis=1)
    trimmed = values[mask]
    if len(trimmed) == 0:
        trimmed = values

    median = np.median(trimmed, axis=0)
    return np.array(
        [
            [median[0], median[2]],
            [median[1], median[3]],
        ],
        dtype=float,
    )


def _global_median_basis(
    session: AnalysisSession,
    selected_points: pd.DataFrame,
    config: "ReferenceLatticeConfig",
    unit: str,
) -> np.ndarray:
    basis_table = pd.DataFrame()
    neighbor_graph = getattr(session, "neighbor_graph", {}) or {}
    graph_basis_table = neighbor_graph.get("basis_table", pd.DataFrame())
    if isinstance(graph_basis_table, pd.DataFrame) and not graph_basis_table.empty:
        basis_table = _valid_basis_rows_from_explicit_columns(graph_basis_table, selected_points["atom_id"])

    if basis_table.empty:
        local_metrics = getattr(session, "local_metrics", pd.DataFrame())
        if isinstance(local_metrics, pd.DataFrame) and not local_metrics.empty:
            basis_table = _basis_rows_from_local_metrics(local_metrics, selected_points["atom_id"])

    if basis_table.empty:
        raise ValueError(
            "global_median 需要已有局域基矢量；请先运行 compute_local_metrics(session)，"
            "或改用 mode='manual' 提供 manual_basis。"
        )
    if len(basis_table) < int(config.min_points):
        raise ValueError(
            "global_median 需要至少 "
            f"{int(config.min_points)} 条有效局域基矢量；当前仅有 {len(basis_table)} 条。"
        )

    basis = _robust_median_basis(basis_table, config)
    if unit == "nm":
        basis = basis * _calibration_scale_to_nm(session)
    return basis


@dataclass
class ReferenceLatticeConfig:
    role_filter: str | None = None
    source: str = "curated"
    use_keep: bool = True
    mode: str = "manual_basis"
    manual_basis: tuple[tuple[float, float], tuple[float, float]] | None = None
    manual_origin: tuple[float, float] | None = None
    coordinate_unit: str = "px"
    min_points: int = 10
    robust_percentile_low: float = 5.0
    robust_percentile_high: float = 95.0
    atom_role: str | None = None
    manual_basis_nm: tuple[tuple[float, float], tuple[float, float]] | None = None
    manual_origin_nm: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        self.source = str(self.source)
        if self.source not in _ALLOWED_REFERENCE_SOURCES:
            raise ValueError(f"source 必须是 {sorted(_ALLOWED_REFERENCE_SOURCES)} 之一。")

        mode = str(self.mode)
        if mode not in _ALLOWED_REFERENCE_MODES:
            raise ValueError(f"mode 必须是 {sorted(_ALLOWED_REFERENCE_MODES)} 之一。")
        self.mode = _MODE_ALIASES[mode]

        self.coordinate_unit = str(self.coordinate_unit)
        if self.coordinate_unit not in _ALLOWED_COORDINATE_UNITS:
            raise ValueError(f"coordinate_unit 必须是 {sorted(_ALLOWED_COORDINATE_UNITS)} 之一。")
        if int(self.min_points) <= 0:
            raise ValueError("min_points 必须为正数。")

        low = float(self.robust_percentile_low)
        high = float(self.robust_percentile_high)
        if not np.isfinite(low) or not np.isfinite(high) or low < 0.0 or high > 100.0 or low >= high:
            raise ValueError("robust_percentile_low/high 必须满足 0 <= low < high <= 100。")

        if self.role_filter is not None and self.atom_role is not None:
            if str(self.role_filter) != str(self.atom_role):
                raise ValueError("role_filter 与 atom_role 不能指定为不同的值。")
        if self.role_filter is None and self.atom_role is not None:
            self.role_filter = self.atom_role
        if self.atom_role is None and self.role_filter is not None:
            self.atom_role = self.role_filter

        if self.manual_basis is not None and self.manual_basis_nm is not None:
            raise ValueError("manual_basis 与 manual_basis_nm 不能同时指定。")
        if self.manual_origin is not None and self.manual_origin_nm is not None:
            raise ValueError("manual_origin 与 manual_origin_nm 不能同时指定。")
        if self.manual_basis_nm is not None:
            if self.coordinate_unit != "calibrated":
                raise ValueError("manual_basis_nm 仅可在 coordinate_unit='calibrated' 时作为兼容别名使用。")
            self.manual_basis = self.manual_basis_nm
        if self.manual_origin_nm is not None:
            if self.coordinate_unit != "calibrated":
                raise ValueError("manual_origin_nm 仅可在 coordinate_unit='calibrated' 时作为兼容别名使用。")
            self.manual_origin = self.manual_origin_nm

        if self.manual_basis is not None:
            _coerce_manual_basis(self.manual_basis)
        if self.manual_origin is not None:
            _coerce_origin(self.manual_origin)


@dataclass
class ReferenceLattice:
    basis: np.ndarray
    origin: np.ndarray
    unit: str
    role_filter: str | None = None
    mode: str = "manual_basis"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        basis = np.asarray(self.basis, dtype=float)
        if basis.shape != (2, 2):
            raise ValueError("ReferenceLattice.basis 必须是形状为 (2, 2) 的数组。")
        if not np.isfinite(basis).all():
            raise ValueError("ReferenceLattice.basis 必须全部为有限数值。")
        determinant = float(np.linalg.det(basis))
        if not np.isfinite(determinant) or abs(determinant) <= _SINGULAR_TOLERANCE:
            raise ValueError("ReferenceLattice.basis 必须非奇异。")

        origin = np.asarray(self.origin, dtype=float)
        if origin.shape != (2,):
            raise ValueError("ReferenceLattice.origin 必须是形状为 (2,) 的数组。")
        if not np.isfinite(origin).all():
            raise ValueError("ReferenceLattice.origin 必须全部为有限数值。")

        unit = str(self.unit)
        if unit not in _ALLOWED_LATTICE_UNITS:
            raise ValueError(f"ReferenceLattice.unit 必须是 {sorted(_ALLOWED_LATTICE_UNITS)} 之一。")

        self.basis = basis
        self.origin = origin
        self.unit = unit
        self.metadata = dict(self.metadata or {})

    @property
    def basis_nm(self) -> np.ndarray | None:
        if self.unit != "nm":
            return None
        return self.basis

    @property
    def origin_nm(self) -> np.ndarray | None:
        if self.unit != "nm":
            return None
        return self.origin


@dataclass
class ReferenceLatticeSuggestionConfig:
    role_filter: str | None = None
    source: str = "curated"
    use_keep: bool = True
    coordinate_unit: str = "px"
    n_candidates: int = 3
    random_state: int = 0
    min_points: int = 10
    robust_percentile_low: float = 5.0
    robust_percentile_high: float = 95.0
    roi: tuple[float, float, float, float] | None = None
    atom_role: str | None = None

    def __post_init__(self) -> None:
        self.source = str(self.source)
        if self.source not in _ALLOWED_REFERENCE_SOURCES:
            raise ValueError(f"source 必须是 {sorted(_ALLOWED_REFERENCE_SOURCES)} 之一。")
        self.coordinate_unit = str(self.coordinate_unit)
        if self.coordinate_unit not in _ALLOWED_COORDINATE_UNITS:
            raise ValueError(f"coordinate_unit 必须是 {sorted(_ALLOWED_COORDINATE_UNITS)} 之一。")
        if int(self.n_candidates) <= 0:
            raise ValueError("n_candidates 必须为正数。")
        if int(self.min_points) <= 0:
            raise ValueError("min_points 必须为正数。")
        low = float(self.robust_percentile_low)
        high = float(self.robust_percentile_high)
        if not np.isfinite(low) or not np.isfinite(high) or low < 0.0 or high > 100.0 or low >= high:
            raise ValueError("robust_percentile_low/high 必须满足 0 <= low < high <= 100。")
        if self.role_filter is not None and self.atom_role is not None:
            if str(self.role_filter) != str(self.atom_role):
                raise ValueError("role_filter 与 atom_role 不能指定为不同的值。")
        if self.role_filter is None and self.atom_role is not None:
            self.role_filter = self.atom_role
        if self.atom_role is None and self.role_filter is not None:
            self.atom_role = self.role_filter
        if self.roi is not None:
            roi = tuple(float(value) for value in self.roi)
            if len(roi) != 4 or not np.isfinite(roi).all():
                raise ValueError("roi 必须是 (x_min, x_max, y_min, y_max) 四个有限数值。")
            if roi[0] >= roi[1] or roi[2] >= roi[3]:
                raise ValueError("roi 必须满足 x_min < x_max 且 y_min < y_max。")
            self.roi = roi


@dataclass
class ReferenceLatticeSuggestion:
    candidates: pd.DataFrame
    assignments: pd.DataFrame
    unit: str
    coordinate_unit: str
    source: str
    role_filter: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.candidates = pd.DataFrame(self.candidates).copy()
        self.assignments = pd.DataFrame(self.assignments).copy()
        self.unit = str(self.unit)
        if self.unit not in _ALLOWED_LATTICE_UNITS:
            raise ValueError(f"ReferenceLatticeSuggestion.unit 必须是 {sorted(_ALLOWED_LATTICE_UNITS)} 之一。")
        self.coordinate_unit = str(self.coordinate_unit)
        self.source = str(self.source)
        self.metadata = dict(self.metadata or {})


def _apply_reference_roi(selected_points: pd.DataFrame, config: ReferenceLatticeSuggestionConfig) -> pd.DataFrame:
    if config.roi is None:
        return selected_points
    x_min, x_max, y_min, y_max = config.roi
    selected = selected_points[
        (selected_points["x"] >= x_min)
        & (selected_points["x"] <= x_max)
        & (selected_points["y"] >= y_min)
        & (selected_points["y"] <= y_max)
    ].copy()
    if len(selected) < int(config.min_points):
        raise ValueError(
            "reference ROI 内可用原子数不足；需要至少 "
            f"{int(config.min_points)} 个，当前只有 {len(selected)} 个。"
        )
    return selected.reset_index(drop=True)


def _basis_rows_for_selected_points(session: AnalysisSession, selected_points: pd.DataFrame) -> pd.DataFrame:
    neighbor_graph = getattr(session, "neighbor_graph", {}) or {}
    graph_basis_table = neighbor_graph.get("basis_table", pd.DataFrame())
    if isinstance(graph_basis_table, pd.DataFrame) and not graph_basis_table.empty:
        basis_rows = _basis_rows_from_explicit_columns_with_atom_id(graph_basis_table, selected_points["atom_id"])
        if not basis_rows.empty:
            return basis_rows

    local_metrics = getattr(session, "local_metrics", pd.DataFrame())
    if isinstance(local_metrics, pd.DataFrame) and not local_metrics.empty:
        basis_rows = _basis_rows_from_local_metrics_with_atom_id(local_metrics, selected_points["atom_id"])
        if not basis_rows.empty:
            return basis_rows

    raise ValueError("自动 reference 建议需要已有局域基矢量；请先运行 compute_local_metrics(session)。")


def _basis_summary_from_matrix(basis: np.ndarray) -> dict[str, float]:
    a = basis[:, 0]
    b = basis[:, 1]
    a_length = float(np.linalg.norm(a))
    b_length = float(np.linalg.norm(b))
    cosine = float(np.dot(a, b) / max(a_length * b_length, np.finfo(float).tiny))
    gamma = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    orientation = float(np.degrees(np.arctan2(a[1], a[0])) % 180.0)
    return {
        "basis_a_length": a_length,
        "basis_b_length": b_length,
        "basis_gamma_deg": gamma,
        "basis_orientation_deg": orientation,
    }


def _basis_matrix_from_components(values: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [float(values[0]), float(values[2])],
            [float(values[1]), float(values[3])],
        ],
        dtype=float,
    )


def _components_from_basis_matrix(basis: np.ndarray) -> np.ndarray:
    return np.array([basis[0, 0], basis[1, 0], basis[0, 1], basis[1, 1]], dtype=float)


def _axis_orientation_deg(vector: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(float(vector[1]), float(vector[0]))) % 180.0)


def _canonicalize_axis_vector(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=float).copy()
    if abs(result[0]) >= abs(result[1]):
        if result[0] < 0:
            result *= -1.0
    elif result[1] < 0:
        result *= -1.0
    return result


def _gauss_reduce_2d_basis(basis: np.ndarray, max_iterations: int = 12) -> np.ndarray:
    a = np.asarray(basis[:, 0], dtype=float).copy()
    b = np.asarray(basis[:, 1], dtype=float).copy()
    tiny = np.finfo(float).tiny
    for _ in range(max_iterations):
        changed = False
        if float(np.dot(b, b)) < float(np.dot(a, a)):
            a, b = b, a
            changed = True
        denom = float(np.dot(a, a))
        if denom <= tiny:
            break
        mu = int(np.rint(float(np.dot(a, b)) / denom))
        if mu != 0:
            b = b - mu * a
            changed = True
        if not changed:
            break
    return np.column_stack([a, b])


def _canonicalize_basis_matrix(basis: np.ndarray) -> np.ndarray:
    basis = np.asarray(basis, dtype=float)
    if (
        basis.shape != (2, 2)
        or not np.isfinite(basis).all()
        or abs(float(np.linalg.det(basis))) <= _SINGULAR_TOLERANCE
    ):
        return basis.copy()

    reduced = _gauss_reduce_2d_basis(basis)
    a = _canonicalize_axis_vector(reduced[:, 0])
    b = _canonicalize_axis_vector(reduced[:, 1])
    if _axis_orientation_deg(a) > _axis_orientation_deg(b):
        a, b = b, a
    canonical = np.column_stack([a, b])
    if float(np.linalg.det(canonical)) < 0.0:
        canonical[:, 1] *= -1.0
    return canonical


def _canonicalize_basis_dataframe(basis_data: pd.DataFrame) -> pd.DataFrame:
    result = basis_data.copy()
    values = result[_BASIS_VECTOR_COLUMNS].to_numpy(dtype=float)
    canonical_values = [
        _components_from_basis_matrix(_canonicalize_basis_matrix(_basis_matrix_from_components(row)))
        for row in values
    ]
    result.loc[:, _BASIS_VECTOR_COLUMNS] = np.asarray(canonical_values, dtype=float)
    return result


def _basis_physical_feature_values(values: np.ndarray) -> np.ndarray:
    records: list[list[float]] = []
    tiny = np.finfo(float).tiny
    for row in np.asarray(values, dtype=float):
        basis = _basis_matrix_from_components(row)
        summary = _basis_summary_from_matrix(basis)
        folded_gamma = min(summary["basis_gamma_deg"], 180.0 - summary["basis_gamma_deg"])
        theta = np.deg2rad(2.0 * summary["basis_orientation_deg"])
        records.append(
            [
                float(np.log(max(summary["basis_a_length"], tiny))),
                float(np.log(max(summary["basis_b_length"], tiny))),
                float(folded_gamma),
                float(np.cos(theta)),
                float(np.sin(theta)),
            ]
        )
    return np.asarray(records, dtype=float)


def _robust_standardized_basis_features(values: np.ndarray, config: ReferenceLatticeSuggestionConfig) -> np.ndarray:
    center = np.median(values, axis=0)
    low = np.percentile(values, float(config.robust_percentile_low), axis=0)
    high = np.percentile(values, float(config.robust_percentile_high), axis=0)
    scale = (high - low) / 2.0
    fallback = np.std(values, axis=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (values - center) / scale


def _candidate_table_from_labels(
    basis_data: pd.DataFrame,
    labels: np.ndarray,
    config: ReferenceLatticeSuggestionConfig,
) -> tuple[pd.DataFrame, dict[int, int]]:
    total = int(len(basis_data))
    records: list[dict[str, object]] = []
    for cluster_label in sorted(set(int(label) for label in labels)):
        rows = basis_data.loc[labels == cluster_label].copy()
        basis = _robust_median_basis(rows[_BASIS_VECTOR_COLUMNS], config)
        values = rows[_BASIS_VECTOR_COLUMNS].to_numpy(dtype=float)
        median_components = np.array([basis[0, 0], basis[1, 0], basis[0, 1], basis[1, 1]], dtype=float)
        basis_spread = float(np.sqrt(np.mean(np.sum((values - median_components) ** 2, axis=1))))
        summary = _basis_summary_from_matrix(basis)
        records.append(
            {
                "cluster_label": cluster_label,
                "n_points": int(len(rows)),
                "fraction": float(len(rows) / total),
                "basis_a_x": float(basis[0, 0]),
                "basis_a_y": float(basis[1, 0]),
                "basis_b_x": float(basis[0, 1]),
                "basis_b_y": float(basis[1, 1]),
                "basis_spread": basis_spread,
                **summary,
            }
        )

    candidates = pd.DataFrame(records).sort_values(["n_points", "basis_spread"], ascending=[False, True]).reset_index(drop=True)
    label_to_candidate = {int(row["cluster_label"]): int(index) for index, row in candidates.iterrows()}
    candidates.insert(0, "candidate_id", np.arange(len(candidates), dtype=int))
    candidates = candidates.drop(columns=["cluster_label"])
    return candidates, label_to_candidate


def suggest_reference_lattices(
    session: AnalysisSession,
    config: ReferenceLatticeSuggestionConfig | None = None,
) -> ReferenceLatticeSuggestion:
    config = config or ReferenceLatticeSuggestionConfig()
    reference_config = ReferenceLatticeConfig(
        role_filter=config.role_filter,
        source=config.source,
        use_keep=config.use_keep,
        mode="global_median",
        coordinate_unit=config.coordinate_unit,
        min_points=config.min_points,
        robust_percentile_low=config.robust_percentile_low,
        robust_percentile_high=config.robust_percentile_high,
    )
    selected_points, unit = _select_reference_points(session, reference_config)
    selected_points = _apply_reference_roi(selected_points, config)
    basis_rows = _basis_rows_for_selected_points(session, selected_points)
    basis_data = selected_points.merge(basis_rows, on="atom_id", how="inner").reset_index(drop=True)
    if len(basis_data) < int(config.min_points):
        raise ValueError(
            "自动 reference 建议需要至少 "
            f"{int(config.min_points)} 条有效局域基矢量；当前只有 {len(basis_data)} 条。"
        )
    if unit == "nm":
        scale = _calibration_scale_to_nm(session)
        basis_data.loc[:, _BASIS_VECTOR_COLUMNS] = basis_data[_BASIS_VECTOR_COLUMNS].to_numpy(dtype=float) * scale

    basis_data = _canonicalize_basis_dataframe(basis_data)
    values = basis_data[_BASIS_VECTOR_COLUMNS].to_numpy(dtype=float)
    physical_values = _basis_physical_feature_values(values)
    features = _robust_standardized_basis_features(physical_values, config)
    distinct_count = len(np.unique(np.round(features, decimals=12), axis=0))
    n_clusters = max(1, min(int(config.n_candidates), len(basis_data), distinct_count))
    if n_clusters == 1:
        labels = np.zeros(len(basis_data), dtype=int)
    else:
        import os

        os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=n_clusters, random_state=int(config.random_state), n_init=10).fit_predict(features)

    candidates, label_to_candidate = _candidate_table_from_labels(basis_data, labels, config)
    assignments = basis_data[
        ["atom_id", "x", "y", *([column for column in ("x_px", "y_px", "x_nm", "y_nm") if column in basis_data.columns])]
    ].copy()
    assignments["candidate_id"] = [label_to_candidate[int(label)] for label in labels]
    assignments = assignments.sort_values(["candidate_id", "atom_id"]).reset_index(drop=True)

    return ReferenceLatticeSuggestion(
        candidates=candidates,
        assignments=assignments,
        unit=unit,
        coordinate_unit=config.coordinate_unit,
        source=config.source,
        role_filter=config.role_filter,
        metadata={
            "n_points_selected": int(len(selected_points)),
            "n_basis_rows": int(len(basis_data)),
            "n_candidates_requested": int(config.n_candidates),
            "n_candidates_found": int(len(candidates)),
            "roi": config.roi,
            "use_keep": bool(config.use_keep),
            "basis_convention": _CANONICAL_BASIS_CONVENTION,
            "cluster_feature_mode": _CLUSTER_FEATURE_MODE,
        },
    )


def build_reference_lattice_from_suggestion(
    session: AnalysisSession,
    suggestion: ReferenceLatticeSuggestion,
    candidate_id: int = 0,
    key: str = "default",
) -> AnalysisSession:
    candidates = pd.DataFrame(suggestion.candidates)
    if candidates.empty or "candidate_id" not in candidates.columns:
        raise ValueError("suggestion.candidates 为空，无法构建 reference lattice。")
    candidate_id = int(candidate_id)
    rows = candidates.loc[candidates["candidate_id"].astype(int) == candidate_id]
    if rows.empty:
        raise ValueError(f"suggestion 中没有 candidate_id={candidate_id} 的候选。")
    row = rows.iloc[0]
    basis = np.array(
        [
            [float(row["basis_a_x"]), float(row["basis_b_x"])],
            [float(row["basis_a_y"]), float(row["basis_b_y"])],
        ],
        dtype=float,
    )
    assigned = suggestion.assignments.loc[suggestion.assignments["candidate_id"].astype(int) == candidate_id]
    if assigned.empty:
        raise ValueError(f"candidate_id={candidate_id} 没有对应的原子分配记录。")
    origin = np.median(assigned[["x", "y"]].to_numpy(dtype=float), axis=0)
    lattice = ReferenceLattice(
        basis=basis,
        origin=origin,
        unit=suggestion.unit,
        role_filter=suggestion.role_filter,
        mode="suggested_cluster",
        metadata={
            "source": suggestion.source,
            "mode": "suggested_cluster",
            "candidate_id": candidate_id,
            "n_points_used": int(row["n_points"]),
            "fraction": float(row["fraction"]),
            "basis_spread": float(row["basis_spread"]),
            "coordinate_unit": suggestion.coordinate_unit,
            "basis_convention": _CANONICAL_BASIS_CONVENTION,
            "cluster_feature_mode": suggestion.metadata.get("cluster_feature_mode", _CLUSTER_FEATURE_MODE),
            "suggestion_metadata": dict(suggestion.metadata),
        },
    )

    if not hasattr(session, "reference_lattice") or session.reference_lattice is None:
        session.reference_lattice = {}
    session.reference_lattice[str(key)] = lattice
    session.record_step(
        "build_reference_lattice_from_suggestion",
        parameters={"candidate_id": candidate_id, "key": str(key)},
        notes={"unit": suggestion.unit, "n_points_used": int(row["n_points"])},
    )
    return session


def build_reference_lattice(
    session: AnalysisSession,
    config: ReferenceLatticeConfig,
    key: str = "default",
) -> AnalysisSession:
    if config.mode == "manual_basis" and config.manual_basis is None:
        raise ValueError("mode='manual_basis' 时必须提供 manual_basis；不会使用隐式默认基。")

    selected_points, unit = _select_reference_points(session, config)
    if config.mode == "manual_basis":
        basis = _coerce_manual_basis(config.manual_basis)
    elif config.mode == "median_local_basis":
        basis = _global_median_basis(session, selected_points, config, unit)
    else:
        raise ValueError(f"不支持的参考晶格模式: {config.mode}")

    if config.manual_origin is not None:
        origin = _coerce_origin(config.manual_origin)
    else:
        origin = np.median(selected_points[["x", "y"]].to_numpy(dtype=float), axis=0)

    lattice = ReferenceLattice(
        basis=basis,
        origin=origin,
        unit=unit,
        role_filter=config.role_filter,
        mode=config.mode,
        metadata={
            "n_points_used": int(len(selected_points)),
            "source": config.source,
            "mode": config.mode,
            "atom_role": config.role_filter,
            "coordinate_unit": config.coordinate_unit,
            "basis_convention": "columns_are_a_b_vectors_in_xy",
            "use_keep": bool(config.use_keep),
            "min_points": int(config.min_points),
            "robust_percentile_low": float(config.robust_percentile_low),
            "robust_percentile_high": float(config.robust_percentile_high),
            "preserved_flag_columns": [column for column in selected_points.columns if str(column).startswith("flag_")],
        },
    )

    if not hasattr(session, "reference_lattice") or session.reference_lattice is None:
        session.reference_lattice = {}
    session.reference_lattice[str(key)] = lattice
    session.record_step(
        "build_reference_lattice",
        parameters=config,
        notes={"key": str(key), "mode": config.mode, "unit": unit, "n_points_used": len(selected_points)},
    )
    return session
