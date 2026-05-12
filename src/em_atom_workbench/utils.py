from __future__ import annotations

from dataclasses import is_dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import math
import re

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

LATEST_SESSION_POINTER = "_latest_session.json"
ACTIVE_SESSION_PICKLE = "_active_session.pkl"
ACTIVE_SESSION_INFO = "_active_session.json"
STAGE_ORDER = {
    "loaded": 0,
    "heavy_reviewed": 1,
    "detected": 2,
    "candidate_reviewed": 3,
    "classified": 4,
    "curated": 5,
    "strain": 6,
    "vpcf": 7,
    "metrics": 8,
    "annotated": 9,
    "vector_field": 10,
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "analysis_session"


def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_directory(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def ensure_analysis_output_dir(root: str | Path, session_name: str, create: bool = True) -> Path:
    target = Path(root) / f"{slugify(session_name)}_{timestamp_string()}"
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return target


def calibration_to_physical(values_px: np.ndarray | pd.Series, pixel_size: float | None) -> np.ndarray:
    if pixel_size is None:
        return np.asarray(values_px, dtype=float)
    return np.asarray(values_px, dtype=float) * float(pixel_size)


def extract_patch(
    image: np.ndarray,
    center_xy_global: tuple[float, float],
    half_window: int | float,
    origin_xy: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if half_window is None:
        raise ValueError("half_window must be a finite non-negative number.")
    half_window_value = float(half_window)
    if not math.isfinite(half_window_value) or half_window_value < 0:
        raise ValueError("half_window must be a finite non-negative number.")
    half_window_px = int(round(half_window_value))
    origin_x, origin_y = origin_xy
    x_global, y_global = center_xy_global
    x_local = float(x_global) - origin_x
    y_local = float(y_global) - origin_y
    x0 = max(int(math.floor(x_local)) - half_window_px, 0)
    x1 = min(int(math.floor(x_local)) + half_window_px + 1, int(image.shape[1]))
    y0 = max(int(math.floor(y_local)) - half_window_px, 0)
    y1 = min(int(math.floor(y_local)) + half_window_px + 1, int(image.shape[0]))
    return image[y0:y1, x0:x1], (x0, x1, y0, y1)


def border_values(patch: np.ndarray) -> np.ndarray:
    if patch.size == 0:
        return np.array([], dtype=float)
    if patch.shape[0] < 2 or patch.shape[1] < 2:
        return patch.ravel()
    top = patch[0, :]
    bottom = patch[-1, :]
    left = patch[1:-1, 0]
    right = patch[1:-1, -1]
    return np.concatenate([top, bottom, left, right]).astype(float)


def serializable(obj: Any) -> Any:
    if is_dataclass(obj):
        return serializable(asdict(obj))
    if isinstance(obj, dict):
        return {str(key): serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serializable(item) for item in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(serializable(payload), handle, indent=2, ensure_ascii=False)
    return target


def stage_rank(stage: str | None) -> int:
    return STAGE_ORDER.get(str(stage or "loaded"), -1)


def _ensure_required_stage(session: Any, required_stage: str | None, source_label: str) -> None:
    if required_stage is not None and stage_rank(session.current_stage) < stage_rank(required_stage):
        raise ValueError(
            f"{source_label} stage '{session.current_stage}' does not satisfy the required stage '{required_stage}'."
        )


def _load_active_session_info(results_root: Path) -> dict[str, Any]:
    info_path = results_root / ACTIVE_SESSION_INFO
    if not info_path.exists():
        raise FileNotFoundError(
            "Active session info was not found. Please rerun the previous notebook stage first, or set SESSION_PATH manually."
        )
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Active session info is not valid JSON: {info_path}") from exc


def _validate_active_session(session: Any, payload: dict[str, Any]) -> None:
    comparisons = {
        "session_name": getattr(session, "name", None),
        "input_path": getattr(session, "input_path", None),
        "dataset_index": getattr(session, "dataset_index", None),
        "current_stage": getattr(session, "current_stage", "loaded"),
    }
    mismatches: list[str] = []
    for key, actual in comparisons.items():
        expected = payload.get(key)
        if expected != actual:
            mismatches.append(f"{key}={expected!r} (info) != {actual!r} (pickle)")
    if mismatches:
        raise ValueError(
            "Active session metadata does not match the active pickle payload: "
            + "; ".join(mismatches)
        )


def save_active_session(session: Any, results_root: str | Path) -> Path:
    results_root = ensure_directory(results_root)
    active_path = session.save_pickle(results_root / ACTIVE_SESSION_PICKLE)
    payload = {
        "session_name": session.name,
        "input_path": session.input_path,
        "dataset_index": getattr(session, "dataset_index", None),
        "current_stage": getattr(session, "current_stage", "loaded"),
        "active_session_path": str(active_path),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(results_root / ACTIVE_SESSION_INFO, payload)
    return active_path


def load_active_session(results_root: str | Path, required_stage: str | None = None) -> Any:
    from .session import AnalysisSession

    results_root = Path(results_root)
    active_path = results_root / ACTIVE_SESSION_PICKLE
    payload = _load_active_session_info(results_root)
    if not active_path.exists():
        raise FileNotFoundError(
            "No active session was found. Please run the previous notebook stage first, or set SESSION_PATH manually."
        )
    session = AnalysisSession.load_pickle(active_path)
    _validate_active_session(session, payload)
    _ensure_required_stage(session, required_stage, "Active session")
    return session


def load_or_connect_session(
    results_root: str | Path,
    required_stage: str | None = None,
    session_path: str | Path | None = None,
) -> Any:
    from .session import AnalysisSession

    if session_path:
        path = Path(session_path)
        if not path.exists():
            raise FileNotFoundError(f"Session path does not exist: {path}")
        session = AnalysisSession.load_pickle(path)
        _ensure_required_stage(session, required_stage, "Loaded session")
        return session
    return load_active_session(results_root, required_stage=required_stage)


def save_checkpoint(session: Any, results_root: str | Path, filename: str) -> Path:
    results_root = ensure_directory(results_root)
    snapshot_dir = ensure_directory(results_root / session.name / "checkpoints")
    checkpoint_path = session.save_pickle(snapshot_dir / filename)
    return checkpoint_path


def save_session_snapshot(session: Any, results_root: str | Path, filename: str) -> Path:
    results_root = ensure_directory(results_root)
    snapshot_dir = ensure_directory(results_root / session.name / "snapshots")
    snapshot_path = session.save_pickle(snapshot_dir / filename)

    pointer_path = results_root / LATEST_SESSION_POINTER
    pointer_payload: dict[str, Any] = {}
    if pointer_path.exists():
        try:
            pointer_payload = json.loads(pointer_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pointer_payload = {}

    snapshots = dict(pointer_payload.get("snapshots", {}))
    snapshots[filename] = str(snapshot_path)
    payload = {
        "session_name": session.name,
        "latest_snapshot": str(snapshot_path),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "snapshots": snapshots,
    }
    write_json(pointer_path, payload)
    return snapshot_path


def resolve_session_snapshot(results_root: str | Path, stage_filename: str | list[str] | tuple[str, ...]) -> Path | None:
    results_root = Path(results_root)
    stage_names = [stage_filename] if isinstance(stage_filename, str) else list(stage_filename)

    pointer_path = results_root / LATEST_SESSION_POINTER
    if pointer_path.exists():
        try:
            pointer_payload = json.loads(pointer_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pointer_payload = {}
        snapshots = pointer_payload.get("snapshots", {})
        for name in stage_names:
            candidate = snapshots.get(name)
            if candidate and Path(candidate).exists():
                return Path(candidate)

    matches: list[Path] = []
    for name in stage_names:
        matches.extend(results_root.glob(f"*/snapshots/{name}"))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def synthetic_gaussian_image(
    shape: tuple[int, int],
    peaks: list[dict[str, float]],
    background: float = 0.1,
    noise_sigma: float = 0.0,
    rng_seed: int = 0,
) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=float)
    image = np.full(shape, background, dtype=float)
    for peak in peaks:
        x0 = peak["x"]
        y0 = peak["y"]
        amplitude = peak.get("amplitude", 1.0)
        sigma_x = peak.get("sigma_x", 1.2)
        sigma_y = peak.get("sigma_y", 1.2)
        theta = peak.get("theta", 0.0)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        x_shift = xx - x0
        y_shift = yy - y0
        x_rot = cos_t * x_shift + sin_t * y_shift
        y_rot = -sin_t * x_shift + cos_t * y_shift
        image += amplitude * np.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2))
    if noise_sigma > 0:
        rng = np.random.default_rng(rng_seed)
        image += rng.normal(0.0, noise_sigma, size=shape)
    return image


def synthetic_lattice_image(
    shape: tuple[int, int] = (96, 96),
    spacing: float = 12.0,
    amplitude: float = 1.0,
    sigma: float = 1.2,
    jitter: float = 0.0,
    background: float = 0.1,
    noise_sigma: float = 0.0,
    rng_seed: int = 0,
) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(rng_seed)
    peaks: list[dict[str, float]] = []
    coords: list[dict[str, float]] = []
    for y in np.arange(spacing, shape[0] - spacing / 2, spacing):
        for x in np.arange(spacing, shape[1] - spacing / 2, spacing):
            x_j = float(x + rng.normal(0.0, jitter))
            y_j = float(y + rng.normal(0.0, jitter))
            peaks.append({"x": x_j, "y": y_j, "amplitude": amplitude, "sigma_x": sigma, "sigma_y": sigma})
            coords.append({"x_px": x_j, "y_px": y_j})
    image = synthetic_gaussian_image(
        shape=shape,
        peaks=peaks,
        background=background,
        noise_sigma=noise_sigma,
        rng_seed=rng_seed,
    )
    return image, pd.DataFrame(coords)


def synthetic_hfo2_multichannel_bundle(
    shape: tuple[int, int] = (128, 128),
    spacing: float = 18.0,
    heavy_sigma: float = 1.4,
    light_sigma: float = 1.0,
    heavy_amplitude_haadf: float = 1.6,
    heavy_amplitude_idpc: float = 1.0,
    light_amplitude_idpc: float = 0.33,
    heavy_amplitude_abf: float = 0.18,
    light_amplitude_abf: float = 0.55,
    close_pair_separation_px: float = 2.4,
    noise_sigma_haadf: float = 0.02,
    noise_sigma_idpc: float = 0.015,
    noise_sigma_abf: float = 0.01,
    blur_sigma_haadf: float = 0.35,
    blur_sigma_idpc: float = 0.45,
    blur_sigma_abf: float = 0.35,
    include_abf: bool = True,
    invert_abf: bool = True,
    rng_seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(rng_seed)
    heavy_coords: list[dict[str, float | str]] = []
    light_coords: list[dict[str, float | str]] = []

    ys = np.arange(spacing, shape[0] - spacing, spacing)
    xs = np.arange(spacing, shape[1] - spacing, spacing)

    for y in ys:
        for x in xs:
            heavy_coords.append({"x_px": float(x), "y_px": float(y), "column_role": "heavy_atom"})

    for y in ys:
        for x_left, x_right in zip(xs[:-1], xs[1:], strict=True):
            light_coords.append(
                {
                    "x_px": float(0.5 * (x_left + x_right)),
                    "y_px": float(y),
                    "column_role": "light_atom",
                }
            )

    for y_top, y_bottom in zip(ys[:-1], ys[1:], strict=True):
        for x_left, x_right in zip(xs[:-1], xs[1:], strict=True):
            center_x = float(0.5 * (x_left + x_right))
            center_y = float(0.5 * (y_top + y_bottom))
            if x_left == xs[1] and y_top == ys[1]:
                light_coords.append(
                    {
                        "x_px": center_x - 0.5 * close_pair_separation_px,
                        "y_px": center_y,
                        "column_role": "light_atom",
                    }
                )
                light_coords.append(
                    {
                        "x_px": center_x + 0.5 * close_pair_separation_px,
                        "y_px": center_y,
                        "column_role": "light_atom",
                    }
                )
            else:
                light_coords.append({"x_px": center_x, "y_px": center_y, "column_role": "light_atom"})

    heavy_df = pd.DataFrame(heavy_coords)
    light_df = pd.DataFrame(light_coords)

    haadf_peaks = [
        {"x": row["x_px"], "y": row["y_px"], "amplitude": heavy_amplitude_haadf, "sigma_x": heavy_sigma, "sigma_y": heavy_sigma}
        for _, row in heavy_df.iterrows()
    ]
    idpc_heavy_peaks = [
        {"x": row["x_px"], "y": row["y_px"], "amplitude": heavy_amplitude_idpc, "sigma_x": heavy_sigma, "sigma_y": heavy_sigma}
        for _, row in heavy_df.iterrows()
    ]
    idpc_light_peaks = [
        {
            "x": row["x_px"] + rng.normal(0.0, 0.04),
            "y": row["y_px"] + rng.normal(0.0, 0.04),
            "amplitude": light_amplitude_idpc,
            "sigma_x": light_sigma,
            "sigma_y": light_sigma,
        }
        for _, row in light_df.iterrows()
    ]
    abf_peaks = [
        {"x": row["x_px"], "y": row["y_px"], "amplitude": heavy_amplitude_abf, "sigma_x": heavy_sigma, "sigma_y": heavy_sigma}
        for _, row in heavy_df.iterrows()
    ] + [
        {
            "x": row["x_px"],
            "y": row["y_px"],
            "amplitude": light_amplitude_abf,
            "sigma_x": light_sigma,
            "sigma_y": light_sigma,
        }
        for _, row in light_df.iterrows()
    ]

    haadf = synthetic_gaussian_image(
        shape=shape,
        peaks=haadf_peaks,
        background=0.08,
        noise_sigma=noise_sigma_haadf,
        rng_seed=rng_seed,
    )
    idpc = synthetic_gaussian_image(
        shape=shape,
        peaks=idpc_heavy_peaks + idpc_light_peaks,
        background=0.05,
        noise_sigma=noise_sigma_idpc,
        rng_seed=rng_seed + 1,
    )
    images = {
        "haadf": gaussian_filter(haadf, sigma=blur_sigma_haadf) if blur_sigma_haadf > 0 else haadf,
        "idpc": gaussian_filter(idpc, sigma=blur_sigma_idpc) if blur_sigma_idpc > 0 else idpc,
    }

    if include_abf:
        abf = synthetic_gaussian_image(
            shape=shape,
            peaks=abf_peaks,
            background=0.12,
            noise_sigma=noise_sigma_abf,
            rng_seed=rng_seed + 2,
        )
        if blur_sigma_abf > 0:
            abf = gaussian_filter(abf, sigma=blur_sigma_abf)
        if invert_abf:
            abf = float(np.max(abf) + np.min(abf)) - abf
        images["abf"] = abf

    truth = {
        "heavy": heavy_df.reset_index(drop=True),
        "light": light_df.reset_index(drop=True),
        "all": pd.concat([heavy_df, light_df], ignore_index=True),
    }
    return images, truth


def attach_physical_coordinates(
    table: pd.DataFrame,
    pixel_size: float | None,
    unit: str,
    x_col: str = "x_px",
    y_col: str = "y_px",
) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    result = table.copy()
    result["x_phys"] = calibration_to_physical(result[x_col], pixel_size)
    result["y_phys"] = calibration_to_physical(result[y_col], pixel_size)
    result["unit"] = unit
    return result
