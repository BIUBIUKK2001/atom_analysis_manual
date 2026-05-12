from __future__ import annotations

import numpy as np
import pandas as pd

from .session import AnalysisSession, VectorFieldConfig


def build_vector_field(
    origins_xy: np.ndarray,
    vectors_xy: np.ndarray,
    config: VectorFieldConfig | None = None,
) -> pd.DataFrame:
    config = config or VectorFieldConfig()
    origins_xy = np.asarray(origins_xy, dtype=float)
    vectors_xy = np.asarray(vectors_xy, dtype=float)
    magnitudes = np.linalg.norm(vectors_xy, axis=1)
    return pd.DataFrame(
        {
            "x_px": origins_xy[:, 0],
            "y_px": origins_xy[:, 1],
            "u_px": vectors_xy[:, 0] * config.scale,
            "v_px": vectors_xy[:, 1] * config.scale,
            "magnitude_px": magnitudes * config.scale,
            "unit": config.unit,
        }
    )


def vector_field_from_point_matches(
    source_points: pd.DataFrame,
    target_points: pd.DataFrame,
    match_column: str = "atom_id",
    config: VectorFieldConfig | None = None,
) -> pd.DataFrame:
    merged = source_points[[match_column, "x_px", "y_px"]].merge(
        target_points[[match_column, "x_px", "y_px"]],
        on=match_column,
        suffixes=("_source", "_target"),
        how="inner",
    )
    origins = merged[["x_px_source", "y_px_source"]].to_numpy(dtype=float)
    vectors = merged[["x_px_target", "y_px_target"]].to_numpy(dtype=float) - origins
    field = build_vector_field(origins, vectors, config=config)
    field.insert(0, match_column, merged[match_column].to_numpy())
    return field


def attach_vector_field(session: AnalysisSession, name: str, vector_field: pd.DataFrame) -> AnalysisSession:
    session.vector_fields[name] = vector_field.copy()
    session.set_stage("vector_field")
    session.record_step("attach_vector_field", parameters={"name": name}, notes={"vector_count": len(vector_field)})
    return session
