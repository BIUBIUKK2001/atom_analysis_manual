from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .session import AnalysisSession, LatticeConfig


def _angle_between_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return float("nan")
    cosine = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _canonicalize_vector(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=float).copy()
    if abs(result[0]) >= abs(result[1]):
        if result[0] < 0:
            result *= -1.0
    else:
        if result[1] < 0:
            result *= -1.0
    return result


def _orientation_deg(vector: np.ndarray) -> float:
    return math.degrees(math.atan2(vector[1], vector[0])) % 180.0


def _basis_from_neighbors(vectors: np.ndarray, min_angle_deg: float) -> dict[str, float]:
    if len(vectors) < 2:
        return {
            "basis_a_x": np.nan,
            "basis_a_y": np.nan,
            "basis_b_x": np.nan,
            "basis_b_y": np.nan,
            "basis_a_length_px": np.nan,
            "basis_b_length_px": np.nan,
            "basis_angle_deg": np.nan,
            "local_orientation_deg": np.nan,
        }

    distances = np.linalg.norm(vectors, axis=1)
    order = np.argsort(distances)
    basis_a = vectors[order[0]]
    basis_b = None
    for idx in order[1:]:
        candidate = vectors[idx]
        angle = _angle_between_deg(basis_a, candidate)
        if np.isnan(angle):
            continue
        if angle >= min_angle_deg and angle <= 180.0 - min_angle_deg:
            basis_b = candidate
            break
    if basis_b is None:
        basis_b = vectors[order[1]]

    basis_a = _canonicalize_vector(basis_a)
    basis_b = _canonicalize_vector(basis_b)
    if _orientation_deg(basis_a) > _orientation_deg(basis_b):
        basis_a, basis_b = basis_b, basis_a
    if np.linalg.det(np.column_stack([basis_a, basis_b])) < 0:
        basis_b *= -1.0

    basis_angle = _angle_between_deg(basis_a, basis_b)
    orientation = _orientation_deg(basis_a)
    return {
        "basis_a_x": float(basis_a[0]),
        "basis_a_y": float(basis_a[1]),
        "basis_b_x": float(basis_b[0]),
        "basis_b_y": float(basis_b[1]),
        "basis_a_length_px": float(np.linalg.norm(basis_a)),
        "basis_b_length_px": float(np.linalg.norm(basis_b)),
        "basis_angle_deg": float(basis_angle),
        "local_orientation_deg": float(orientation),
    }


def build_neighbor_graph(session: AnalysisSession, config: LatticeConfig) -> AnalysisSession:
    points = session.get_atom_table(preferred="curated")
    if points.empty:
        raise ValueError("Curated or refined points are required before building the neighbor graph.")

    points = points.copy()
    if "atom_id" not in points.columns:
        points.insert(0, "atom_id", np.arange(len(points), dtype=int))

    coords = points[["x_px", "y_px"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=min(config.k_neighbors + 1, len(points)))
    directed_neighbors: dict[int, list[int]] = {}

    if distances.ndim == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]

    for atom_id, (dist_row, idx_row) in enumerate(zip(distances, indices, strict=True)):
        neighbors: list[int] = []
        for distance, index in zip(dist_row[1:], idx_row[1:], strict=True):
            if np.isinf(distance):
                continue
            if config.max_distance_px is not None and distance > config.max_distance_px:
                continue
            neighbors.append(int(index))
        directed_neighbors[atom_id] = neighbors

    edge_records: list[dict[str, float | int]] = []
    basis_records: list[dict[str, float | int]] = []

    for atom_id, neighbors in directed_neighbors.items():
        vectors = []
        for neighbor in neighbors:
            if config.mutual_only and atom_id not in directed_neighbors.get(neighbor, []):
                continue
            if atom_id < neighbor:
                delta = coords[neighbor] - coords[atom_id]
                edge_records.append(
                    {
                        "source_atom_id": int(points.iloc[atom_id]["atom_id"]),
                        "target_atom_id": int(points.iloc[neighbor]["atom_id"]),
                        "distance_px": float(np.linalg.norm(delta)),
                        "dx_px": float(delta[0]),
                        "dy_px": float(delta[1]),
                    }
                )
            vectors.append(coords[neighbor] - coords[atom_id])

        basis = _basis_from_neighbors(np.asarray(vectors, dtype=float), config.min_basis_angle_deg)
        basis["atom_id"] = int(points.iloc[atom_id]["atom_id"])
        basis_records.append(basis)

    session.neighbor_graph = {
        "edges": pd.DataFrame(edge_records),
        "basis_table": pd.DataFrame(basis_records),
        "directed_neighbors": directed_neighbors,
        "config": config,
    }
    session.record_step(
        "build_neighbor_graph",
        parameters=config,
        notes={"edge_count": len(edge_records), "point_count": len(points)},
    )
    return session
