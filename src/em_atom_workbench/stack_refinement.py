from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import pandas as pd

from .refine import refine_point_table, refine_points_by_class
from .session import AnalysisSession, PixelCalibration, RefinementConfig


_PASSTHROUGH_COLUMNS = [
    "point_id",
    "atom_id",
    "original_candidate_id",
    "class_id",
    "class_name",
    "class_color",
    "roi_id",
    "roi_name",
    "roi_color",
    "scope_id",
    "coordinate_source",
    "source_table",
]

_REFINEMENT_COLUMNS = [
    "x_px",
    "y_px",
    "x_input_px",
    "y_input_px",
    "x_fit_px",
    "y_fit_px",
    "amplitude",
    "local_background",
    "sigma_x",
    "sigma_y",
    "theta",
    "fit_residual",
    "fit_success",
    "refinement_method",
    "quality_score",
    "center_shift_px",
    "attempted_center_shift_px",
    "center_shift_rejected",
    "position_source",
    "nn_distance_px",
    "adaptive_half_window_px",
    "gaussian_attempt_count",
    "gaussian_image_source",
    "refinement_path",
    "refinement_config_source",
    "nn_context_mode",
]


def _validate_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("image must be a 2D array.")
    return arr


def _validate_stack(stack: np.ndarray) -> np.ndarray:
    arr = np.asarray(stack)
    if arr.ndim != 3:
        raise ValueError("stack must be a 3D array with shape (n_slices, height, width).")
    return arr


def _resolve_slice_indices_and_labels(
    n_slices: int,
    slice_indices: Sequence[int] | None,
    slice_labels: Sequence[object] | None,
) -> tuple[list[int], list[object]]:
    indices = list(range(n_slices)) if slice_indices is None else [int(value) for value in slice_indices]
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


def _prepare_seed_points(points: pd.DataFrame, *, source_table: str) -> pd.DataFrame:
    if points is None:
        points = pd.DataFrame()
    seed = points.copy(deep=True).reset_index(drop=True)
    if "x_px" not in seed.columns or "y_px" not in seed.columns:
        raise ValueError("points must contain x_px and y_px columns.")
    if "candidate_id" in seed.columns:
        seed["original_candidate_id"] = seed["candidate_id"]
    seed["candidate_id"] = np.arange(len(seed), dtype=int)
    if "atom_id" not in seed.columns:
        seed["atom_id"] = seed["candidate_id"].astype(int)
    if "source_table" not in seed.columns:
        seed["source_table"] = source_table
    return seed


def _fallback_refinement_table(seed: pd.DataFrame, *, message: str, nn_context_mode: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in seed.iterrows():
        x = pd.to_numeric(pd.Series([row.get("x_px", np.nan)]), errors="coerce").iloc[0]
        y = pd.to_numeric(pd.Series([row.get("y_px", np.nan)]), errors="coerce").iloc[0]
        out = row.to_dict()
        out.update(
            {
                "x_px": float(x) if pd.notna(x) else np.nan,
                "y_px": float(y) if pd.notna(y) else np.nan,
                "x_input_px": float(x) if pd.notna(x) else np.nan,
                "y_input_px": float(y) if pd.notna(y) else np.nan,
                "x_fit_px": np.nan,
                "y_fit_px": np.nan,
                "amplitude": np.nan,
                "local_background": np.nan,
                "sigma_x": np.nan,
                "sigma_y": np.nan,
                "theta": np.nan,
                "fit_residual": np.nan,
                "fit_success": False,
                "refinement_method": "failed",
                "quality_score": 0.0,
                "center_shift_px": 0.0,
                "attempted_center_shift_px": np.nan,
                "center_shift_rejected": True,
                "position_source": "seed_after_refinement_failure",
                "nn_distance_px": np.nan,
                "adaptive_half_window_px": pd.NA,
                "gaussian_attempt_count": 0,
                "gaussian_image_source": pd.NA,
                "refinement_path": "failed",
                "refinement_config_source": "failed",
                "nn_context_mode": nn_context_mode,
                "status": "refinement_failed",
                "failure_message": str(message),
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def _merge_passthrough(refined: pd.DataFrame, seed: pd.DataFrame) -> pd.DataFrame:
    result = refined.copy().reset_index(drop=True)
    if "candidate_id" not in result.columns or "candidate_id" not in seed.columns:
        return result
    seed_by_id = seed.set_index("candidate_id", drop=False)
    for column in _PASSTHROUGH_COLUMNS:
        if column not in seed_by_id.columns:
            continue
        mapped = result["candidate_id"].map(seed_by_id[column])
        if column not in result.columns:
            result[column] = mapped
        elif column in {"point_id", "roi_id", "roi_name", "roi_color", "scope_id", "coordinate_source", "source_table", "original_candidate_id"}:
            result[column] = result[column].where(result[column].notna(), mapped)
    return result


def _ordered_columns(data: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    for column in preferred:
        if column not in data.columns:
            data[column] = pd.NA
    return data[preferred + [column for column in data.columns if column not in preferred]]


def refine_points_on_image(
    points: pd.DataFrame,
    image: np.ndarray,
    refinement_config: RefinementConfig,
    *,
    class_refinement_overrides: dict[int | str, RefinementConfig | dict[str, object]] | None = None,
    nn_context_mode: str = "all",
    pixel_calibration: PixelCalibration | None = None,
    contrast_mode: str = "bright_peak",
    channel_name: str = "stack_slice",
    source_table: str = "seed",
) -> pd.DataFrame:
    """Refine seed points on one 2D image using the existing Notebook01 refinement code."""

    arr = _validate_image(image)
    seed = _prepare_seed_points(points, source_table=source_table)
    session = AnalysisSession(
        name="stack_slice_refinement",
        raw_image=arr,
        pixel_calibration=pixel_calibration or PixelCalibration(),
        contrast_mode=contrast_mode,
        primary_channel=str(channel_name or "stack_slice"),
    )
    session.candidate_points = seed.copy()

    try:
        if "class_id" in seed.columns:
            refine_points_by_class(
                session,
                refinement_config,
                class_refinement_overrides=class_refinement_overrides,
                source_table="candidate",
                nn_context_mode=nn_context_mode,
            )
            refined = session.refined_points.copy()
        else:
            refined = refine_point_table(
                session,
                seed,
                refinement_config,
                nn_context_mode=nn_context_mode,
            )
    except Exception as exc:
        refined = _fallback_refinement_table(seed, message=str(exc), nn_context_mode=nn_context_mode)

    refined = _merge_passthrough(refined, seed)
    preferred = [
        "point_id",
        "atom_id",
        "candidate_id",
        "original_candidate_id",
        "class_id",
        "class_name",
        "class_color",
        "roi_id",
        "roi_name",
        "roi_color",
        "scope_id",
        "coordinate_source",
        "source_table",
        *_REFINEMENT_COLUMNS,
    ]
    return _ordered_columns(refined, preferred)


def refine_stack_point_table(
    points: pd.DataFrame,
    stack: np.ndarray,
    refinement_config: RefinementConfig,
    *,
    slice_indices: Sequence[int] | None = None,
    slice_labels: Sequence[object] | None = None,
    class_refinement_overrides: dict[int | str, RefinementConfig | dict[str, object]] | None = None,
    nn_context_mode: str = "all",
    pixel_calibration: PixelCalibration | None = None,
    contrast_mode: str = "bright_peak",
    channel_name: str = "stack",
    source_table: str = "seed",
) -> pd.DataFrame:
    """Refine the same seed coordinates independently on each selected stack slice."""

    arr = _validate_stack(stack)
    seed = _prepare_seed_points(points, source_table=source_table)
    indices, labels = _resolve_slice_indices_and_labels(arr.shape[0], slice_indices, slice_labels)
    tables: list[pd.DataFrame] = []
    seed_lookup = seed.set_index("candidate_id", drop=False)

    for slice_index, slice_label in zip(indices, labels, strict=True):
        refined = refine_points_on_image(
            seed,
            arr[int(slice_index)],
            refinement_config,
            class_refinement_overrides=class_refinement_overrides,
            nn_context_mode=nn_context_mode,
            pixel_calibration=pixel_calibration,
            contrast_mode=contrast_mode,
            channel_name=f"{channel_name}_slice_{int(slice_index)}",
            source_table=source_table,
        )
        refined["slice_index"] = int(slice_index)
        refined["slice_label"] = slice_label
        refined["x_seed_px"] = refined["candidate_id"].map(seed_lookup["x_px"])
        refined["y_seed_px"] = refined["candidate_id"].map(seed_lookup["y_px"])
        refined["coordinate_mode"] = "slice_refined"
        tables.append(refined)

    result = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    preferred = [
        "point_id",
        "atom_id",
        "candidate_id",
        "original_candidate_id",
        "slice_index",
        "slice_label",
        "x_seed_px",
        "y_seed_px",
        "x_px",
        "y_px",
        "x_input_px",
        "y_input_px",
        "x_fit_px",
        "y_fit_px",
        "class_id",
        "class_name",
        "class_color",
        "roi_id",
        "roi_name",
        "roi_color",
        "scope_id",
        "coordinate_source",
        "source_table",
        *_REFINEMENT_COLUMNS[6:],
        "coordinate_mode",
    ]
    return _ordered_columns(result, preferred)
