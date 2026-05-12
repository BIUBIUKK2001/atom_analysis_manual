from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .session import AnalysisSession

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - 仅作为无 scipy 环境的兜底
    cKDTree = None

from .reference import (
    _extract_coordinate_table,
    _get_atom_coordinate_table,
    _keep_mask,
    _role_column,
)

_ALLOWED_STRAIN_SOURCES = {"curated", "refined", "candidate"}
_ALLOWED_STRAIN_TYPES = {"small", "green"}
_ALLOWED_OUTPUT_FRAMES = {"image"}
_SINGULAR_TOLERANCE = 1e-12


def _as_vector_pairs(value: object, name: str) -> np.ndarray:
    vectors = np.asarray(value, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError(f"{name} 必须是形状为 (N, 2) 的数组。")
    if not np.isfinite(vectors).all():
        raise ValueError(f"{name} 必须全部为有限数值。")
    return vectors


def _as_matrix_2x2(value: object, name: str, *, require_non_singular: bool = False) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError(f"{name} 必须是形状为 (2, 2) 的数组。")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} 必须全部为有限数值。")
    if require_non_singular:
        determinant = float(np.linalg.det(matrix))
        if not np.isfinite(determinant) or abs(determinant) <= _SINGULAR_TOLERANCE:
            raise ValueError(f"{name} 必须非奇异。")
    return matrix


def _validate_neighbor_shells(neighbor_shells: int) -> int:
    if isinstance(neighbor_shells, (bool, np.bool_)) or not isinstance(neighbor_shells, (int, np.integer)):
        raise ValueError("neighbor_shells 必须是大于等于 1 的整数。")
    neighbor_shells = int(neighbor_shells)
    if neighbor_shells < 1:
        raise ValueError("neighbor_shells 必须是大于等于 1 的整数。")
    return neighbor_shells


def _validate_tolerance(tolerance: float) -> float:
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance 必须为有限且大于 0 的数值。")
    return tolerance


def _assign_reference_pair_vectors(
    obs_vectors: np.ndarray,
    basis: np.ndarray,
    neighbor_shells: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs_vectors = _as_vector_pairs(obs_vectors, "obs_vectors")
    basis = _as_matrix_2x2(basis, "basis", require_non_singular=True)
    neighbor_shells = _validate_neighbor_shells(neighbor_shells)
    tolerance = _validate_tolerance(tolerance)

    if len(obs_vectors) == 0:
        return (
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=int),
            np.empty((0,), dtype=float),
        )

    fractional = np.linalg.solve(basis, obs_vectors.T).T
    best_by_index: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
    for q, observed in zip(fractional, obs_vectors, strict=True):
        h = np.rint(q).astype(int)
        if h[0] == 0 and h[1] == 0:
            continue
        if int(np.max(np.abs(h))) > neighbor_shells:
            continue
        residual = float(np.linalg.norm(q - h))
        if residual > tolerance:
            continue
        key = (int(h[0]), int(h[1]))
        existing = best_by_index.get(key)
        if existing is None or residual < existing[0]:
            best_by_index[key] = (residual, observed.copy())

    if not best_by_index:
        return (
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=int),
            np.empty((0,), dtype=float),
        )

    ordered_keys = sorted(best_by_index)
    miller_like_indices = np.asarray(ordered_keys, dtype=int)
    ref_vectors = (basis @ miller_like_indices.T).T
    obs_vectors_kept = np.asarray([best_by_index[key][1] for key in ordered_keys], dtype=float)
    assignment_residuals = np.asarray([best_by_index[key][0] for key in ordered_keys], dtype=float)
    return ref_vectors, obs_vectors_kept, miller_like_indices, assignment_residuals


def _validate_weights(weights: object, count: int) -> np.ndarray:
    weights_array = np.asarray(weights, dtype=float)
    if weights_array.shape != (count,):
        raise ValueError("weights 长度必须与向量对数量一致。")
    if not np.isfinite(weights_array).all():
        raise ValueError("weights 必须全部为有限数值。")
    if np.any(weights_array < 0.0):
        raise ValueError("weights 必须为非负数值。")
    if float(np.sum(weights_array)) <= 0.0:
        raise ValueError("weights 的总和必须大于 0。")
    return weights_array


def _fit_affine_gradient(
    ref_vectors: np.ndarray,
    obs_vectors: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    ref_vectors = _as_vector_pairs(ref_vectors, "ref_vectors")
    obs_vectors = _as_vector_pairs(obs_vectors, "obs_vectors")
    if ref_vectors.shape != obs_vectors.shape:
        raise ValueError("ref_vectors 与 obs_vectors 必须具有相同形状。")
    if len(ref_vectors) < 2:
        raise ValueError("拟合仿射梯度至少需要 2 对向量。")

    R = ref_vectors.T
    O = obs_vectors.T
    weights_array = None
    if weights is None:
        fit_R = R
        fit_O = O
    else:
        weights_array = _validate_weights(weights, len(ref_vectors))
        scale = np.sqrt(weights_array)[np.newaxis, :]
        fit_R = R * scale
        fit_O = O * scale

    if np.linalg.matrix_rank(fit_R) < 2:
        raise ValueError("参考向量不足以拟合二维仿射梯度；向量数量不足或共线。")

    condition_number = float(np.linalg.cond(fit_R))
    F = fit_O @ np.linalg.pinv(fit_R)
    residual = obs_vectors - (F @ R).T
    residual_sq_norm = np.sum(residual**2, axis=1)
    if weights_array is None:
        residual_rms = float(np.sqrt(np.mean(residual_sq_norm)))
    else:
        residual_rms = float(np.sqrt(np.sum(weights_array * residual_sq_norm) / np.sum(weights_array)))
    return F, condition_number, residual_rms


def _strain_from_deformation_gradient(
    F: np.ndarray,
    strain_type: str = "small",
) -> tuple[np.ndarray, float]:
    F = _as_matrix_2x2(F, "F")
    if strain_type == "small":
        epsilon = 0.5 * (F + F.T) - np.eye(2)
    elif strain_type == "green":
        epsilon = 0.5 * (F.T @ F - np.eye(2))
    else:
        raise ValueError(f"strain_type 必须是 {_ALLOWED_STRAIN_TYPES} 之一。")
    rotation_rad = float(0.5 * (F[1, 0] - F[0, 1]))
    return epsilon, rotation_rad


def _local_lattice_from_F(
    F: np.ndarray,
    basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    F = _as_matrix_2x2(F, "F")
    basis = _as_matrix_2x2(basis, "basis", require_non_singular=True)
    local_basis = F @ basis
    local_a = local_basis[:, 0]
    local_b = local_basis[:, 1]
    denominator = float(np.linalg.norm(local_a) * np.linalg.norm(local_b))
    if denominator <= _SINGULAR_TOLERANCE:
        raise ValueError("局域基矢量长度不能为零。")
    cosine = float(np.clip(np.dot(local_a, local_b) / denominator, -1.0, 1.0))
    local_gamma_deg = float(np.degrees(np.arccos(cosine)))
    return local_a, local_b, local_gamma_deg


@dataclass
class LocalAffineStrainConfig:
    role_filter: str | None = None
    atom_role: str | None = None
    source: str = "curated"
    use_keep: bool = True
    reference_key: str = "default"
    neighbor_shells: int = 2
    k_neighbors: int = 12
    min_pairs: int = 6
    pair_assignment_tolerance: float = 0.35
    max_condition_number: float = 100.0
    strain_type: str = "small"
    output_frame: str = "image"
    weight_power: float = 1.0

    def __post_init__(self) -> None:
        if self.role_filter is not None and self.atom_role is not None:
            if str(self.role_filter) != str(self.atom_role):
                raise ValueError("role_filter 与 atom_role 不能指定为不同的值。")
        if self.role_filter is None and self.atom_role is not None:
            self.role_filter = self.atom_role
        if self.atom_role is None and self.role_filter is not None:
            self.atom_role = self.role_filter
        if self.source not in _ALLOWED_STRAIN_SOURCES:
            raise ValueError(f"source 必须是 {_ALLOWED_STRAIN_SOURCES} 之一。")
        if int(self.neighbor_shells) <= 0:
            raise ValueError("neighbor_shells 必须为正数。")
        if int(self.k_neighbors) <= 0:
            raise ValueError("k_neighbors 必须为正数。")
        if int(self.min_pairs) <= 0:
            raise ValueError("min_pairs 必须为正数。")
        pair_assignment_tolerance = float(self.pair_assignment_tolerance)
        if not np.isfinite(pair_assignment_tolerance) or pair_assignment_tolerance <= 0.0:
            raise ValueError("pair_assignment_tolerance 必须为正数。")
        max_condition_number = float(self.max_condition_number)
        if not np.isfinite(max_condition_number) or max_condition_number <= 0.0:
            raise ValueError("max_condition_number 必须为正数。")
        if self.strain_type not in _ALLOWED_STRAIN_TYPES:
            raise ValueError(f"strain_type 必须是 {_ALLOWED_STRAIN_TYPES} 之一。")
        if self.output_frame not in _ALLOWED_OUTPUT_FRAMES:
            raise ValueError("当前实现仅支持 output_frame='image'。")
        weight_power = float(self.weight_power)
        if not np.isfinite(weight_power) or weight_power < 0.0:
            raise ValueError("weight_power 必须为有限且非负的数值。")


_RESULT_NUMERIC_COLUMNS = (
    "eps_xx",
    "eps_yy",
    "eps_xy",
    "rotation_rad",
    "rotation_deg",
    "principal_eps_1",
    "principal_eps_2",
    "dilatation",
    "shear_magnitude",
    "F_xx",
    "F_xy",
    "F_yx",
    "F_yy",
    "local_a_x",
    "local_a_y",
    "local_b_x",
    "local_b_y",
    "local_a_length",
    "local_b_length",
    "local_gamma_deg",
    "assignment_residual_mean",
    "assignment_residual_max",
    "condition_number",
    "affine_residual",
)


def _coordinate_unit_for_reference(reference: object) -> str:
    unit = str(getattr(reference, "unit", ""))
    if unit == "px":
        return "px"
    if unit == "nm":
        return "calibrated"
    raise ValueError(f"参考晶格 unit={unit!r} 暂不支持；当前仅支持 'px' 或 'nm'。")


def _coordinate_table_for_strain(
    session: AnalysisSession,
    config: LocalAffineStrainConfig,
    reference: object,
) -> tuple[pd.DataFrame, str]:
    source_table = _get_atom_coordinate_table(session, config.source)
    coordinates, unit = _extract_coordinate_table(
        session,
        source_table,
        config.source,
        _coordinate_unit_for_reference(reference),
    )
    source_table = source_table.reset_index(drop=True)
    coordinates = coordinates.reset_index(drop=True)

    for column in ("x_px", "y_px"):
        if column in source_table.columns and column not in coordinates.columns:
            coordinates[column] = source_table[column].to_numpy()

    if "role" not in coordinates.columns:
        role_column = _role_column(coordinates)
        if role_column is not None:
            coordinates["role"] = coordinates[role_column]
    if "channel" not in coordinates.columns and "seed_channel" in coordinates.columns:
        coordinates["channel"] = coordinates["seed_channel"]

    if config.role_filter is not None:
        if "role" not in coordinates.columns:
            raise ValueError("已指定 role_filter/atom_role，但坐标表中没有 role 或 column_role 列。")
        coordinates = coordinates[coordinates["role"].astype(str) == str(config.role_filter)]

    if config.use_keep and "keep" in coordinates.columns:
        coordinates = coordinates[_keep_mask(coordinates["keep"])]

    coordinates = coordinates.reset_index(drop=True)
    if coordinates.empty:
        raise ValueError("局域仿射应变计算没有可用原子点；请检查 source、role_filter/atom_role 和 keep 过滤。")
    return coordinates, unit


def _nearest_neighbor_indices(coords: np.ndarray, k_neighbors: int) -> list[list[int]]:
    count = len(coords)
    if count == 0:
        return []
    if count == 1:
        return [[]]

    k_query = min(int(k_neighbors) + 1, count)
    if cKDTree is not None:
        distances, indices = cKDTree(coords).query(coords, k=k_query)
        if k_query == 1:
            indices = indices[:, np.newaxis]
        result: list[list[int]] = []
        for center_index, index_row in enumerate(indices):
            neighbors = [int(index) for index in index_row if int(index) != center_index]
            result.append(neighbors[: int(k_neighbors)])
        return result

    result = []
    for center_index, center in enumerate(coords):
        distances = np.linalg.norm(coords - center, axis=1)
        order = np.argsort(distances)
        neighbors = [int(index) for index in order if int(index) != center_index]
        result.append(neighbors[: int(k_neighbors)])
    return result


def _base_result_record(
    row: pd.Series,
    *,
    coordinate_unit: str,
    reference_key: str,
    reference_mode: str,
    strain_type: str,
) -> dict[str, object]:
    record: dict[str, object] = {
        "atom_id": row["atom_id"],
        "x": float(row["x"]),
        "y": float(row["y"]),
        "coordinate_unit": coordinate_unit,
        "reference_key": reference_key,
        "reference_mode": reference_mode,
        "strain_type": strain_type,
    }
    for column in ("x_px", "y_px", "x_nm", "y_nm", "role", "channel"):
        if column in row.index:
            record[column] = row[column]
    return record


def _failed_result_record(
    base_record: dict[str, object],
    *,
    qc_flag: str,
    n_neighbors: int,
    n_pairs: int,
    assignment_residuals: np.ndarray | None = None,
    condition_number: float = np.nan,
    affine_residual: float = np.nan,
    F: np.ndarray | None = None,
) -> dict[str, object]:
    record = dict(base_record)
    for column in _RESULT_NUMERIC_COLUMNS:
        record[column] = np.nan
    if F is not None:
        record.update(
            {
                "F_xx": float(F[0, 0]),
                "F_xy": float(F[0, 1]),
                "F_yx": float(F[1, 0]),
                "F_yy": float(F[1, 1]),
            }
        )
    if assignment_residuals is not None and len(assignment_residuals):
        record["assignment_residual_mean"] = float(np.mean(assignment_residuals))
        record["assignment_residual_max"] = float(np.max(assignment_residuals))
    record["condition_number"] = float(condition_number) if np.isfinite(condition_number) else np.nan
    record["affine_residual"] = float(affine_residual) if np.isfinite(affine_residual) else np.nan
    record["n_neighbors"] = int(n_neighbors)
    record["n_pairs"] = int(n_pairs)
    record["qc_flag"] = qc_flag
    return record


def _successful_result_record(
    base_record: dict[str, object],
    *,
    F: np.ndarray,
    epsilon: np.ndarray,
    rotation_rad: float,
    local_a: np.ndarray,
    local_b: np.ndarray,
    local_gamma_deg: float,
    affine_residual: float,
    assignment_residuals: np.ndarray,
    condition_number: float,
    n_neighbors: int,
    n_pairs: int,
) -> dict[str, object]:
    principal = np.linalg.eigvalsh(epsilon)
    eps_xx = float(epsilon[0, 0])
    eps_yy = float(epsilon[1, 1])
    eps_xy = float(epsilon[0, 1])
    record = dict(base_record)
    record.update(
        {
            "eps_xx": eps_xx,
            "eps_yy": eps_yy,
            "eps_xy": eps_xy,
            "rotation_rad": float(rotation_rad),
            "rotation_deg": float(np.degrees(rotation_rad)),
            "principal_eps_1": float(principal[-1]),
            "principal_eps_2": float(principal[0]),
            "dilatation": eps_xx + eps_yy,
            "shear_magnitude": float(np.sqrt((eps_xx - eps_yy) ** 2 + 4.0 * eps_xy**2)),
            "F_xx": float(F[0, 0]),
            "F_xy": float(F[0, 1]),
            "F_yx": float(F[1, 0]),
            "F_yy": float(F[1, 1]),
            "local_a_x": float(local_a[0]),
            "local_a_y": float(local_a[1]),
            "local_b_x": float(local_b[0]),
            "local_b_y": float(local_b[1]),
            "local_a_length": float(np.linalg.norm(local_a)),
            "local_b_length": float(np.linalg.norm(local_b)),
            "local_gamma_deg": float(local_gamma_deg),
            "n_neighbors": int(n_neighbors),
            "n_pairs": int(n_pairs),
            "assignment_residual_mean": float(np.mean(assignment_residuals)) if len(assignment_residuals) else np.nan,
            "assignment_residual_max": float(np.max(assignment_residuals)) if len(assignment_residuals) else np.nan,
            "condition_number": float(condition_number),
            "affine_residual": float(affine_residual),
            "qc_flag": "ok",
        }
    )
    return record


def _add_nm_aliases(strain_table: pd.DataFrame, coordinate_unit: str) -> pd.DataFrame:
    if coordinate_unit != "nm":
        return strain_table
    alias_pairs = {
        "local_a_length_nm": "local_a_length",
        "local_b_length_nm": "local_b_length",
        "affine_residual_nm": "affine_residual",
    }
    for alias, source in alias_pairs.items():
        if source in strain_table.columns:
            strain_table[alias] = strain_table[source]
    return strain_table


def compute_local_affine_strain(
    session: AnalysisSession,
    config: LocalAffineStrainConfig,
) -> AnalysisSession:
    reference_lattice = dict(getattr(session, "reference_lattice", {}) or {})
    reference_key = str(config.reference_key)
    if reference_key not in reference_lattice:
        raise ValueError("缺少参考晶格；请先运行 build_reference_lattice(session, config)。")
    reference = reference_lattice[reference_key]
    basis = _as_matrix_2x2(getattr(reference, "basis", None), "reference.basis", require_non_singular=True)
    reference_mode = str(getattr(reference, "mode", "unknown"))

    coordinates, coordinate_unit = _coordinate_table_for_strain(session, config, reference)
    coords = coordinates[["x", "y"]].to_numpy(dtype=float)
    neighbor_indices = _nearest_neighbor_indices(coords, int(config.k_neighbors))

    records: list[dict[str, object]] = []
    tiny = np.finfo(float).tiny
    for center_index, row in coordinates.iterrows():
        neighbors = neighbor_indices[center_index]
        base_record = _base_result_record(
            row,
            coordinate_unit=coordinate_unit,
            reference_key=reference_key,
            reference_mode=reference_mode,
            strain_type=config.strain_type,
        )
        if not neighbors:
            records.append(
                _failed_result_record(
                    base_record,
                    qc_flag="too_few_pairs",
                    n_neighbors=0,
                    n_pairs=0,
                )
            )
            continue

        obs_vectors = coords[neighbors] - coords[center_index]
        ref_vectors, obs_vectors_kept, _, assignment_residuals = _assign_reference_pair_vectors(
            obs_vectors,
            basis,
            neighbor_shells=config.neighbor_shells,
            tolerance=config.pair_assignment_tolerance,
        )
        n_pairs = len(ref_vectors)
        if n_pairs < int(config.min_pairs):
            records.append(
                _failed_result_record(
                    base_record,
                    qc_flag="too_few_pairs",
                    n_neighbors=len(neighbors),
                    n_pairs=n_pairs,
                    assignment_residuals=assignment_residuals,
                )
            )
            continue

        ref_norms = np.linalg.norm(ref_vectors, axis=1)
        weights = 1.0 / np.maximum(ref_norms, tiny) ** float(config.weight_power)
        try:
            F, condition_number, affine_residual = _fit_affine_gradient(ref_vectors, obs_vectors_kept, weights=weights)
        except Exception:
            records.append(
                _failed_result_record(
                    base_record,
                    qc_flag="fit_failed",
                    n_neighbors=len(neighbors),
                    n_pairs=n_pairs,
                    assignment_residuals=assignment_residuals,
                )
            )
            continue

        if condition_number > float(config.max_condition_number):
            records.append(
                _failed_result_record(
                    base_record,
                    qc_flag="ill_conditioned",
                    n_neighbors=len(neighbors),
                    n_pairs=n_pairs,
                    assignment_residuals=assignment_residuals,
                    condition_number=condition_number,
                    affine_residual=affine_residual,
                    F=F,
                )
            )
            continue

        try:
            epsilon, rotation_rad = _strain_from_deformation_gradient(F, config.strain_type)
            local_a, local_b, local_gamma_deg = _local_lattice_from_F(F, basis)
        except Exception:
            records.append(
                _failed_result_record(
                    base_record,
                    qc_flag="fit_failed",
                    n_neighbors=len(neighbors),
                    n_pairs=n_pairs,
                    assignment_residuals=assignment_residuals,
                    condition_number=condition_number,
                    affine_residual=affine_residual,
                    F=F,
                )
            )
            continue

        records.append(
            _successful_result_record(
                base_record,
                F=F,
                epsilon=epsilon,
                rotation_rad=rotation_rad,
                local_a=local_a,
                local_b=local_b,
                local_gamma_deg=local_gamma_deg,
                affine_residual=affine_residual,
                assignment_residuals=assignment_residuals,
                condition_number=condition_number,
                n_neighbors=len(neighbors),
                n_pairs=n_pairs,
            )
        )

    strain_table = pd.DataFrame(records)
    strain_table = _add_nm_aliases(strain_table, coordinate_unit)
    session.strain_table = strain_table
    session.set_stage("strain")
    qc_counts = strain_table["qc_flag"].value_counts(dropna=False).to_dict() if "qc_flag" in strain_table else {}
    session.record_step(
        "compute_local_affine_strain",
        parameters=asdict(config),
        notes={
            "row_count": int(len(strain_table)),
            "ok_count": int(qc_counts.get("ok", 0)),
            "reference_key": reference_key,
            "coordinate_unit": coordinate_unit,
            "qc_flag_counts": {str(key): int(value) for key, value in qc_counts.items()},
        },
    )
    return session
