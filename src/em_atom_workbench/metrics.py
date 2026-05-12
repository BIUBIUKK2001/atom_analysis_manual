from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .lattice import build_neighbor_graph
from .session import AnalysisSession, LatticeConfig, MetricsConfig


def _reference_basis_matrix(basis_table: pd.DataFrame, config: MetricsConfig) -> np.ndarray | None:
    if config.reference_basis is not None:
        return np.asarray(config.reference_basis, dtype=float).T

    valid = basis_table.dropna(subset=["basis_a_x", "basis_a_y", "basis_b_x", "basis_b_y"])
    if valid.empty:
        return None
    ref = np.array(
        [
            [valid["basis_a_x"].median(), valid["basis_b_x"].median()],
            [valid["basis_a_y"].median(), valid["basis_b_y"].median()],
        ],
        dtype=float,
    )
    if abs(np.linalg.det(ref)) < 1e-8:
        return None
    return ref


def _compute_bond_angle_stats(center: np.ndarray, neighbors: np.ndarray) -> tuple[float, float]:
    if len(neighbors) < 2:
        return np.nan, np.nan
    vectors = neighbors - center
    angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
    angles = np.sort((angles + 360.0) % 360.0)
    diffs = np.diff(np.concatenate([angles, [angles[0] + 360.0]]))
    return float(np.mean(diffs)), float(np.std(diffs))


def compute_local_metrics(
    session: AnalysisSession,
    config: MetricsConfig | None = None,
    lattice_config: LatticeConfig | None = None,
) -> AnalysisSession:
    config = config or MetricsConfig()
    if not session.neighbor_graph:
        build_neighbor_graph(session, lattice_config or LatticeConfig())

    points = session.get_atom_table(preferred="curated").copy()
    if points.empty:
        raise ValueError("Curated or refined points are required before metric computation.")

    if "atom_id" not in points.columns:
        points.insert(0, "atom_id", np.arange(len(points), dtype=int))

    basis_table = session.neighbor_graph.get("basis_table", pd.DataFrame()).copy()
    directed_neighbors = session.neighbor_graph.get("directed_neighbors", {})

    coords = points[["x_px", "y_px"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    nn_distances, _ = tree.query(coords, k=min(7, len(points)))
    reference_matrix = _reference_basis_matrix(basis_table, config)

    metrics_records: list[dict[str, float | int | str]] = []
    reference_lookup: dict[int, np.ndarray] = {}
    if config.reference_points is not None and not config.reference_points.empty:
        ref_table = config.reference_points.set_index(config.reference_match_column)
        for atom_id, row in ref_table.iterrows():
            reference_lookup[int(atom_id)] = np.array([float(row["x_px"]), float(row["y_px"])], dtype=float)

    for row_index, row in points.iterrows():
        atom_id = int(row["atom_id"])
        neighbors_idx = directed_neighbors.get(row_index, [])
        neighbor_coords = coords[neighbors_idx] if neighbors_idx else np.empty((0, 2), dtype=float)

        nn_values = nn_distances[row_index][1:] if nn_distances.ndim > 1 else np.array([], dtype=float)
        mean_bond_angle, std_bond_angle = _compute_bond_angle_stats(coords[row_index], neighbor_coords)
        basis_row = basis_table[basis_table["atom_id"] == atom_id]
        basis_values = basis_row.iloc[0].to_dict() if not basis_row.empty else {}

        displacement_x = np.nan
        displacement_y = np.nan
        displacement_norm = np.nan
        if atom_id in reference_lookup:
            delta = coords[row_index] - reference_lookup[atom_id]
            displacement_x = float(delta[0])
            displacement_y = float(delta[1])
            displacement_norm = float(np.linalg.norm(delta))

        exx = np.nan
        eyy = np.nan
        exy = np.nan
        if reference_matrix is not None and basis_values:
            local_matrix = np.array(
                [
                    [basis_values.get("basis_a_x", np.nan), basis_values.get("basis_b_x", np.nan)],
                    [basis_values.get("basis_a_y", np.nan), basis_values.get("basis_b_y", np.nan)],
                ],
                dtype=float,
            )
            if not np.isnan(local_matrix).any() and abs(np.linalg.det(reference_matrix)) > 1e-8:
                deformation = local_matrix @ np.linalg.inv(reference_matrix)
                strain = 0.5 * (deformation + deformation.T) - np.eye(2)
                exx = float(strain[0, 0])
                eyy = float(strain[1, 1])
                exy = float(strain[0, 1])

        metrics_records.append(
            {
                "atom_id": atom_id,
                "nn_count": int(len(nn_values)),
                "nearest_neighbor_distance_px": float(nn_values[0]) if len(nn_values) else np.nan,
                "mean_nn_distance_px": float(np.mean(nn_values)) if len(nn_values) else np.nan,
                "std_nn_distance_px": float(np.std(nn_values)) if len(nn_values) else np.nan,
                "mean_bond_angle_deg": mean_bond_angle,
                "std_bond_angle_deg": std_bond_angle,
                "basis_a_length_px": basis_values.get("basis_a_length_px", np.nan),
                "basis_b_length_px": basis_values.get("basis_b_length_px", np.nan),
                "basis_angle_deg": basis_values.get("basis_angle_deg", np.nan),
                "local_orientation_deg": basis_values.get("local_orientation_deg", np.nan),
                "interplanar_spacing_a_px": basis_values.get("basis_a_length_px", np.nan),
                "interplanar_spacing_b_px": basis_values.get("basis_b_length_px", np.nan),
                "reference_displacement_x_px": displacement_x,
                "reference_displacement_y_px": displacement_y,
                "reference_displacement_norm_px": displacement_norm,
                "strain_exx": exx,
                "strain_eyy": eyy,
                "strain_exy": exy,
            }
        )

    metrics = pd.DataFrame(metrics_records)
    if "annotation_label" in points.columns:
        metrics = metrics.merge(points[["atom_id", "annotation_label"]], on="atom_id", how="left")

    session.local_metrics = metrics
    session.set_stage("metrics")
    session.record_step("compute_local_metrics", parameters=config, notes={"metric_count": len(metrics)})
    return session
