from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .session import AnalysisSession


@dataclass(frozen=True)
class VPCFConfig:
    r_max_px: float = 40.0
    grid_size: int = 128
    extent_px: float | None = None
    sigma_grid_px: float = 1.5
    normalize: str = "max"


@dataclass(frozen=True)
class LocalVPCFResult:
    H: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    neighbor_indices: np.ndarray
    vectors: np.ndarray
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "H": self.H,
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "neighbor_indices": self.neighbor_indices,
            "vectors": self.vectors,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BatchVPCFResult:
    average_H: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    center_indices: np.ndarray
    center_summary: pd.DataFrame
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "average_H": self.average_H,
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "center_indices": self.center_indices,
            "center_summary": self.center_summary,
            "metadata": dict(self.metadata),
        }


def compute_local_vpcf(
    coords: np.ndarray,
    center_index: int,
    config: VPCFConfig | None = None,
) -> LocalVPCFResult:
    config = _validate_config(config or VPCFConfig())
    coords = _validate_coords(coords)
    if center_index < 0 or center_index >= len(coords):
        raise IndexError(f"center_index is out of range: {center_index}")

    center = coords[int(center_index)]
    deltas = coords - center
    distances = np.linalg.norm(deltas, axis=1)
    mask = (distances > 0.0) & (distances <= config.r_max_px)
    neighbor_indices = np.flatnonzero(mask).astype(int)
    vectors = deltas[neighbor_indices].astype(float)

    x_axis, y_axis, qx, qy, sigma = _grid(config)
    H = np.zeros((config.grid_size, config.grid_size), dtype=float)
    for vx, vy in vectors:
        H += np.exp(-((qx - vx) ** 2 + (qy - vy) ** 2) / (2.0 * sigma**2))

    H = _normalize_H(H, config.normalize)
    metadata = {
        "center_index": int(center_index),
        "center_x_px": float(center[0]),
        "center_y_px": float(center[1]),
        "neighbor_count": int(len(neighbor_indices)),
        "r_max_px": float(config.r_max_px),
        "extent_px": _extent(config),
        "grid_size": int(config.grid_size),
        "sigma_px": float(sigma),
        "normalize": config.normalize,
    }
    return LocalVPCFResult(H, x_axis, y_axis, neighbor_indices, vectors, metadata)


def compute_batch_vpcf(
    coords: np.ndarray,
    center_indices: np.ndarray | list[int] | tuple[int, ...],
    config: VPCFConfig | None = None,
) -> BatchVPCFResult:
    config = _validate_config(config or VPCFConfig())
    coords = _validate_coords(coords)
    center_indices = np.asarray(center_indices, dtype=int)
    if center_indices.ndim != 1:
        raise ValueError("center_indices must be a 1D sequence.")
    if np.any(center_indices < 0) or np.any(center_indices >= len(coords)):
        raise IndexError("center_indices contains values outside the coords range.")

    x_axis, y_axis, _, _, _ = _grid(config)
    H_sum = np.zeros((config.grid_size, config.grid_size), dtype=float)
    rows: list[dict[str, Any]] = []
    for center_index in center_indices:
        local = compute_local_vpcf(coords, int(center_index), config)
        H_sum += local.H
        rows.append(
            {
                "center_index": int(center_index),
                "center_x_px": local.metadata["center_x_px"],
                "center_y_px": local.metadata["center_y_px"],
                "neighbor_count": local.metadata["neighbor_count"],
            }
        )

    if len(center_indices):
        average_H = H_sum / float(len(center_indices))
    else:
        average_H = H_sum
    average_H = _normalize_H(average_H, config.normalize)

    metadata = {
        "center_count": int(len(center_indices)),
        "r_max_px": float(config.r_max_px),
        "extent_px": _extent(config),
        "grid_size": int(config.grid_size),
        "normalize": config.normalize,
    }
    return BatchVPCFResult(average_H, x_axis, y_axis, center_indices, pd.DataFrame(rows), metadata)


def compute_session_vpcf(
    session: AnalysisSession,
    config: VPCFConfig | None = None,
    *,
    center_atom_id: int | None = None,
    roi_x_range: tuple[float, float] | None = None,
    roi_y_range: tuple[float, float] | None = None,
    use_keep: bool = True,
) -> AnalysisSession:
    config = _validate_config(config or VPCFConfig())
    points = points_for_vpcf(session, use_keep=use_keep)
    unit_info = _session_vpcf_unit_info(session)
    coords = points[["x_px", "y_px"]].to_numpy(dtype=float)
    center_indices = np.arange(len(points), dtype=int)
    if len(center_indices) == 0:
        raise ValueError("No points are available for vPCF.")

    if center_atom_id is None:
        example_index = int(center_indices[len(center_indices) // 2])
    else:
        matches = np.flatnonzero(points["atom_id"].to_numpy(dtype=int) == int(center_atom_id))
        if len(matches) == 0:
            raise ValueError(f"CENTER_ATOM_ID was not found in the vPCF point table: {center_atom_id}")
        example_index = int(matches[0])

    example = compute_local_vpcf(coords, example_index, config)
    global_result = compute_batch_vpcf(coords, center_indices, config)
    center_summary = global_result.center_summary.copy()
    center_summary["atom_id"] = points.iloc[center_summary["center_index"].to_numpy(dtype=int)]["atom_id"].to_numpy()
    center_summary = _point_table_to_nm(center_summary, unit_info.pixel_to_nm)
    points_used = _point_table_to_nm(points.reset_index(drop=True), unit_info.pixel_to_nm)

    result: dict[str, Any] = {
        "config": _config_to_nm(config, unit_info.pixel_to_nm),
        "input_config_px": asdict(config),
        "use_keep": bool(use_keep),
        "coordinate_unit": "nm",
        "pixel_to_nm": unit_info.pixel_to_nm,
        "source_pixel_calibration": unit_info.source_calibration,
        "points_used": points_used,
        "x_axis": global_result.x_axis * unit_info.pixel_to_nm,
        "y_axis": global_result.y_axis * unit_info.pixel_to_nm,
        "global_average_H": global_result.average_H,
        "example_local_H": example.H,
        "example_neighbor_indices": example.neighbor_indices,
        "example_vectors": example.vectors * unit_info.pixel_to_nm,
        "example_metadata": {
            **_metadata_to_nm(example.metadata, unit_info.pixel_to_nm),
            "atom_id": int(points.iloc[example_index]["atom_id"]),
        },
        "center_summary": center_summary.reset_index(drop=True),
    }

    region_indices = _roi_center_indices(points, roi_x_range, roi_y_range)
    if region_indices is not None:
        region_result = compute_batch_vpcf(coords, region_indices, config)
        result["region_average_H"] = region_result.average_H
        result["region_center_summary"] = _point_table_to_nm(
            region_result.center_summary,
            unit_info.pixel_to_nm,
        ).reset_index(drop=True)
        result["region_metadata"] = {
            **_metadata_to_nm(region_result.metadata, unit_info.pixel_to_nm),
            "roi_x_range_nm": _range_to_nm(roi_x_range, unit_info.pixel_to_nm),
            "roi_y_range_nm": _range_to_nm(roi_y_range, unit_info.pixel_to_nm),
        }

    session.vpcf_results = result
    session.set_stage("vpcf")
    session.record_step(
        "compute_session_vpcf",
        parameters={"config": config, "use_keep": use_keep, "center_atom_id": center_atom_id},
        notes={
            "point_count": len(points),
            "example_neighbor_count": int(example.metadata["neighbor_count"]),
            "region_enabled": region_indices is not None,
        },
    )
    return session


def points_for_vpcf(session: AnalysisSession, *, use_keep: bool = True) -> pd.DataFrame:
    if session.workflow_mode != "single_channel":
        raise ValueError("The current vPCF 03 workflow is single-channel only.")
    points = session.get_atom_table(preferred="curated").copy()
    if points.empty:
        raise ValueError("Curated points are required before vPCF computation.")
    if use_keep and "keep" in points.columns:
        original_count = int(len(points))
        keep_mask = points["keep"].astype(bool)
        kept_count = int(keep_mask.sum())
        points = points.loc[keep_mask].copy()
        if points.empty:
            raise ValueError(
                "No points remain for vPCF after applying keep == True. "
                f"curated_points rows: {original_count}; keep == True rows: {kept_count}. "
                "Set USE_KEEP_POINTS = False to inspect all curated points, or return to the curation stage "
                "and relax the filters / manually keep valid atoms."
            )
    required = {"x_px", "y_px"}
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError(f"vPCF points are missing required columns: {missing}")
    if "atom_id" not in points.columns:
        points.insert(0, "atom_id", np.arange(len(points), dtype=int))
    points["atom_id"] = points["atom_id"].astype(int)
    return points.reset_index(drop=True)


def plot_vpcf(
    H: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    *,
    ax=None,
    title: str = "vPCF",
    cmap: str = "magma",
    unit: str = "px",
):
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.5))
    image = ax.imshow(
        H,
        extent=[float(x_axis[0]), float(x_axis[-1]), float(y_axis[0]), float(y_axis[-1])],
        origin="lower",
        cmap=cmap,
        aspect="equal",
    )
    ax.axhline(0.0, color="white", linewidth=0.6, alpha=0.55)
    ax.axvline(0.0, color="white", linewidth=0.6, alpha=0.55)
    ax.set_xlabel(f"vx ({unit})")
    ax.set_ylabel(f"vy ({unit})")
    ax.set_title(title)
    return ax.figure, ax, image


@dataclass(frozen=True)
class _VPCFUnitInfo:
    pixel_to_nm: float
    source_calibration: dict[str, Any]


def _session_vpcf_unit_info(session: AnalysisSession) -> _VPCFUnitInfo:
    calibration = getattr(session, "pixel_calibration", None)
    pixel_size = getattr(calibration, "size", None)
    source_unit = str(getattr(calibration, "unit", "px") or "px").strip()
    source = str(getattr(calibration, "source", "unknown") or "unknown")
    if pixel_size is None:
        raise ValueError(
            "vPCF session output requires calibrated pixel size so result coordinates can be saved in nm. "
            "Provide MANUAL_CALIBRATION in the import notebook or rerun data loading with calibration metadata."
        )
    unit_factor = _unit_to_nm_factor(source_unit)
    pixel_to_nm = float(pixel_size) * unit_factor
    if pixel_to_nm <= 0 or not np.isfinite(pixel_to_nm):
        raise ValueError(f"Invalid pixel calibration for vPCF: size={pixel_size!r}, unit={source_unit!r}")
    return _VPCFUnitInfo(
        pixel_to_nm=pixel_to_nm,
        source_calibration={
            "size": float(pixel_size),
            "unit": source_unit,
            "source": source,
            "pixel_to_nm": pixel_to_nm,
        },
    )


def _unit_to_nm_factor(unit: str) -> float:
    normalized = (
        unit.strip()
        .lower()
        .replace("\u00b5", "u")
        .replace("\u03bc", "u")
    )
    normalized = unicodedata.normalize("NFKD", normalized).encode("ascii", "ignore").decode("ascii")
    factors = {
        "nm": 1.0,
        "nanometer": 1.0,
        "nanometers": 1.0,
        "nanometre": 1.0,
        "nanometres": 1.0,
        "a": 0.1,
        "angstrom": 0.1,
        "angstroms": 0.1,
        "å": 0.1,
        "pm": 0.001,
        "picometer": 0.001,
        "picometers": 0.001,
        "um": 1000.0,
        "micrometer": 1000.0,
        "micrometers": 1000.0,
        "micrometre": 1000.0,
        "micrometres": 1000.0,
    }
    if normalized not in factors:
        raise ValueError(
            f"Cannot convert pixel calibration unit to nm: {unit!r}. "
            "Use a calibration unit such as 'nm', 'A', 'angstrom', 'pm', or 'um'."
        )
    return factors[normalized]


def _point_table_to_nm(table: pd.DataFrame, pixel_to_nm: float) -> pd.DataFrame:
    result = table.copy()
    for stale_column in ("x_phys", "y_phys", "unit"):
        if stale_column in result.columns:
            result = result.drop(columns=stale_column)

    rename_map: dict[str, str] = {}
    for column in result.columns:
        if column.endswith("_px"):
            rename_map[column] = f"{column[:-3]}_nm"
    for column, new_column in rename_map.items():
        result[new_column] = result[column].astype(float) * pixel_to_nm
    if rename_map:
        result = result.drop(columns=list(rename_map))

    for column in ("sigma_x", "sigma_y"):
        if column in result.columns:
            result[f"{column}_nm"] = result[column].astype(float) * pixel_to_nm
            result = result.drop(columns=column)
    result["unit"] = "nm"
    return result


def _metadata_to_nm(metadata: dict[str, Any], pixel_to_nm: float) -> dict[str, Any]:
    key_map = {
        "center_x_px": "center_x_nm",
        "center_y_px": "center_y_nm",
        "r_max_px": "r_max_nm",
        "extent_px": "extent_nm",
        "sigma_px": "sigma_nm",
    }
    converted: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in key_map:
            converted[key_map[key]] = float(value) * pixel_to_nm
        else:
            converted[key] = value
    converted["coordinate_unit"] = "nm"
    return converted


def _config_to_nm(config: VPCFConfig, pixel_to_nm: float) -> dict[str, Any]:
    _, _, _, _, sigma_px = _grid(config)
    return {
        "r_max_nm": float(config.r_max_px) * pixel_to_nm,
        "grid_size": int(config.grid_size),
        "extent_nm": _extent(config) * pixel_to_nm,
        "sigma_nm": float(sigma_px) * pixel_to_nm,
        "sigma_grid_pixels": float(config.sigma_grid_px),
        "normalize": config.normalize,
        "coordinate_unit": "nm",
    }


def _range_to_nm(value: tuple[float, float] | None, pixel_to_nm: float) -> tuple[float, float] | None:
    if value is None:
        return None
    return (float(value[0]) * pixel_to_nm, float(value[1]) * pixel_to_nm)


def _validate_config(config: VPCFConfig) -> VPCFConfig:
    if config.r_max_px <= 0:
        raise ValueError("r_max_px must be positive.")
    if config.grid_size < 2:
        raise ValueError("grid_size must be at least 2.")
    if config.extent_px is not None and config.extent_px <= 0:
        raise ValueError("extent_px must be positive when provided.")
    if config.sigma_grid_px <= 0:
        raise ValueError("sigma_grid_px must be positive.")
    if config.normalize not in {"max", "none"}:
        raise ValueError("normalize must be either 'max' or 'none'.")
    return config


def _validate_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2).")
    return coords


def _extent(config: VPCFConfig) -> float:
    return float(config.r_max_px if config.extent_px is None else config.extent_px)


def _grid(config: VPCFConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    extent = _extent(config)
    axis = np.linspace(-extent, extent, int(config.grid_size), dtype=float)
    qx, qy = np.meshgrid(axis, axis)
    grid_step = 2.0 * extent / float(config.grid_size - 1)
    sigma = float(config.sigma_grid_px) * grid_step
    return axis, axis.copy(), qx, qy, sigma


def _normalize_H(H: np.ndarray, normalize: str) -> np.ndarray:
    if normalize == "none":
        return H
    max_value = float(np.max(H)) if H.size else 0.0
    if max_value <= 0.0:
        return H
    return H / max_value


def _roi_center_indices(
    points: pd.DataFrame,
    roi_x_range: tuple[float, float] | None,
    roi_y_range: tuple[float, float] | None,
) -> np.ndarray | None:
    if roi_x_range is None and roi_y_range is None:
        return None
    mask = pd.Series(True, index=points.index)
    if roi_x_range is not None:
        x0, x1 = sorted((float(roi_x_range[0]), float(roi_x_range[1])))
        mask &= points["x_px"].between(x0, x1, inclusive="both")
    if roi_y_range is not None:
        y0, y1 = sorted((float(roi_y_range[0]), float(roi_y_range[1])))
        mask &= points["y_px"].between(y0, y1, inclusive="both")
    return np.flatnonzero(mask.to_numpy())
