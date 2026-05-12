from __future__ import annotations

from dataclasses import fields, replace
import math

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

from .session import AnalysisSession, RefinementConfig
from .utils import attach_physical_coordinates, border_values, extract_patch


def _positive_patch_for_com(patch: np.ndarray) -> tuple[np.ndarray, float]:
    background = float(np.median(border_values(patch)))
    patch_pos = np.clip(patch - background, 0.0, None)
    return patch_pos, background


def _center_of_mass(patch: np.ndarray) -> tuple[float, float]:
    patch_pos, _ = _positive_patch_for_com(patch)
    total = patch_pos.sum()
    if total <= 0:
        return (patch.shape[1] - 1) / 2.0, (patch.shape[0] - 1) / 2.0
    yy, xx = np.indices(patch.shape, dtype=float)
    x_com = float((xx * patch_pos).sum() / total)
    y_com = float((yy * patch_pos).sum() / total)
    return x_com, y_com


def _gaussian2d(
    coords: tuple[np.ndarray, np.ndarray],
    amplitude: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    background: float,
) -> np.ndarray:
    xx, yy = coords
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_shift = xx - x0
    y_shift = yy - y0
    x_rot = cos_t * x_shift + sin_t * y_shift
    y_rot = -sin_t * x_shift + cos_t * y_shift
    exponent = -0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2)
    return background + amplitude * np.exp(exponent)


def _double_gaussian_shared_shape(
    coords: tuple[np.ndarray, np.ndarray],
    amplitude_a: float,
    x_a: float,
    y_a: float,
    amplitude_b: float,
    x_b: float,
    y_b: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    background: float,
) -> np.ndarray:
    return (
        _gaussian2d(coords, amplitude_a, x_a, y_a, sigma_x, sigma_y, theta, 0.0)
        + _gaussian2d(coords, amplitude_b, x_b, y_b, sigma_x, sigma_y, theta, 0.0)
        + background
    )


def _quadratic_peak(patch: np.ndarray) -> tuple[float, float]:
    y0, x0 = np.unravel_index(np.argmax(patch), patch.shape)
    if not (0 < x0 < patch.shape[1] - 1 and 0 < y0 < patch.shape[0] - 1):
        return float(x0), float(y0)

    neighborhood = patch[y0 - 1 : y0 + 2, x0 - 1 : x0 + 2]
    center = neighborhood[1, 1]
    dx_num = neighborhood[1, 2] - neighborhood[1, 0]
    dx_den = 2 * (2 * center - neighborhood[1, 2] - neighborhood[1, 0])
    dy_num = neighborhood[2, 1] - neighborhood[0, 1]
    dy_den = 2 * (2 * center - neighborhood[2, 1] - neighborhood[0, 1])
    dx = float(dx_num / dx_den) if abs(dx_den) > 1e-8 else 0.0
    dy = float(dy_num / dy_den) if abs(dy_den) > 1e-8 else 0.0
    return float(x0) + np.clip(dx, -0.5, 0.5), float(y0) + np.clip(dy, -0.5, 0.5)


def _coerce_half_window(value: float | int) -> int:
    return max(int(round(float(value))), 1)


def _shared_fallback_half_window(config: RefinementConfig) -> int:
    return max(_coerce_half_window(config.fit_half_window), _coerce_half_window(config.com_half_window))


def _resolve_nn_distances(points: pd.DataFrame, context_points: pd.DataFrame | None = None) -> np.ndarray:
    context = points if context_points is None else context_points
    if len(points) == 0 or len(context) < 2:
        return np.full(len(points), np.nan, dtype=float)

    coords = points[["x_px", "y_px"]].to_numpy(dtype=float)
    context_coords = context[["x_px", "y_px"]].to_numpy(dtype=float)
    tree = cKDTree(context_coords)
    query_k = min(len(context), 8)
    distances, _ = tree.query(coords, k=query_k)
    distances = np.asarray(distances, dtype=float)
    if distances.ndim == 1:
        distances = distances[:, None]

    nn_distances = np.full(len(points), np.nan, dtype=float)
    for row_idx, row_distances in enumerate(distances):
        valid = row_distances[np.isfinite(row_distances) & (row_distances > 0)]
        if valid.size:
            nn_distances[row_idx] = float(valid[0])
    return nn_distances


def _normalize_nn_context_mode(nn_context_mode: str) -> str:
    mode = str(nn_context_mode).strip().lower()
    if mode not in {"all", "same_class"}:
        raise ValueError("nn_context_mode must be 'all' or 'same_class'.")
    return mode


def _resolve_nn_distances_by_mode(points: pd.DataFrame, nn_context_mode: str) -> np.ndarray:
    mode = _normalize_nn_context_mode(nn_context_mode)
    if mode == "all":
        return _resolve_nn_distances(points, points)
    if "class_id" not in points.columns:
        raise ValueError("class_id is required when nn_context_mode='same_class'.")

    nn_distances = np.full(len(points), np.nan, dtype=float)
    class_ids = pd.Series(points["class_id"]).fillna(-1).astype(int)
    for class_id in sorted(class_ids.unique()):
        positions = np.flatnonzero(class_ids.to_numpy(dtype=int) == int(class_id))
        if len(positions) < 2:
            continue
        class_points = points.iloc[positions].reset_index(drop=True)
        nn_distances[positions] = _resolve_nn_distances(class_points, class_points)
    return nn_distances


def _resolve_adaptive_half_window(nn_distance_px: float, config: RefinementConfig) -> int:
    if np.isfinite(nn_distance_px) and nn_distance_px > 0:
        raw_radius = float(nn_distance_px) * float(config.nn_radius_fraction)
        clamped = float(np.clip(raw_radius, config.min_patch_radius_px, config.max_patch_radius_px))
        return _coerce_half_window(clamped)
    return _shared_fallback_half_window(config)


def _resolve_image_and_origin(
    session: AnalysisSession,
    image_source: str,
    channel_name: str | None = None,
) -> tuple[np.ndarray, tuple[int, int], str]:
    if image_source == "processed":
        return session.get_processed_image(channel_name), session.get_processed_origin(channel_name), "processed"
    if image_source == "raw":
        channel = session.get_channel_state(channel_name)
        if channel.raw_image is not None:
            return channel.raw_image, (0, 0), "raw"
        return session.get_processed_image(channel_name), session.get_processed_origin(channel_name), "processed"
    raise ValueError(f"Unsupported image_source: {image_source}")


def _refine_center_with_com(
    image: np.ndarray,
    center_xy_global: tuple[float, float],
    half_window: int,
    origin_xy: tuple[int, int],
) -> tuple[float, float]:
    patch, bounds = extract_patch(
        image=image,
        center_xy_global=center_xy_global,
        half_window=_coerce_half_window(half_window),
        origin_xy=origin_xy,
    )
    if patch.size == 0:
        return float(center_xy_global[0]), float(center_xy_global[1])

    x_patch, y_patch = _center_of_mass(np.asarray(patch, dtype=float))
    x0_local, _, y0_local, _ = bounds
    return (
        float(origin_xy[0] + x0_local + x_patch),
        float(origin_xy[1] + y0_local + y_patch),
    )


def _fit_gaussian_once(patch: np.ndarray, config: RefinementConfig) -> dict[str, float | bool | str]:
    yy, xx = np.indices(patch.shape, dtype=float)
    x_com, y_com = _center_of_mass(patch)
    border = border_values(patch)
    background0 = float(np.median(border))
    amplitude0 = float(max(patch.max() - background0, 1e-6))

    p0 = (
        amplitude0,
        x_com,
        y_com,
        config.initial_sigma_px,
        config.initial_sigma_px,
        0.0,
        background0,
    )
    bounds = (
        [0.0, 0.0, 0.0, config.min_sigma_px, config.min_sigma_px, -np.pi / 2, patch.min() - amplitude0],
        [np.inf, patch.shape[1] - 1, patch.shape[0] - 1, config.max_sigma_px, config.max_sigma_px, np.pi / 2, patch.max()],
    )

    params, _ = curve_fit(
        _gaussian2d,
        (xx.ravel(), yy.ravel()),
        patch.ravel(),
        p0=p0,
        bounds=bounds,
        maxfev=config.max_nfev,
    )
    fitted = _gaussian2d((xx, yy), *params).reshape(patch.shape)
    residual = float(np.sqrt(np.mean((patch - fitted) ** 2)) / (np.std(patch) + 1e-8))
    return {
        "fit_success": True,
        "refinement_method": "gaussian",
        "x_patch": float(params[1]),
        "y_patch": float(params[2]),
        "amplitude": float(params[0]),
        "sigma_x": float(params[3]),
        "sigma_y": float(params[4]),
        "theta": float(params[5]),
        "local_background": float(params[6]),
        "fit_residual": residual,
    }


def _gaussian_fit_is_reasonable(
    fitted: dict[str, float | bool | str],
    patch_shape: tuple[int, int],
    config: RefinementConfig,
) -> bool:
    x_patch = float(fitted["x_patch"])
    y_patch = float(fitted["y_patch"])
    sigma_x = float(fitted["sigma_x"])
    sigma_y = float(fitted["sigma_y"])

    if not np.isfinite(x_patch) or not np.isfinite(y_patch):
        return False
    if not np.isfinite(sigma_x) or not np.isfinite(sigma_y):
        return False
    if sigma_x <= 0 or sigma_y <= 0:
        return False

    sigma_ratio = max(sigma_x, sigma_y) / max(min(sigma_x, sigma_y), 1e-12)
    if not np.isfinite(sigma_ratio) or sigma_ratio > float(config.sigma_ratio_limit):
        return False

    margin = max(float(config.fit_edge_margin_px), 0.0)
    x_max = float(patch_shape[1] - 1) - margin
    y_max = float(patch_shape[0] - 1) - margin
    if x_patch < margin or y_patch < margin or x_patch > x_max or y_patch > y_max:
        return False
    return True


def _quadratic_fallback(patch: np.ndarray) -> dict[str, float | bool | str]:
    x_q, y_q = _quadratic_peak(patch)
    _, background = _positive_patch_for_com(patch)
    amplitude = float(patch.max() - background)
    residual = float(np.std(patch - patch.mean()) / (np.std(patch) + 1e-8))
    return {
        "fit_success": False,
        "refinement_method": "quadratic",
        "x_patch": x_q,
        "y_patch": y_q,
        "amplitude": amplitude,
        "sigma_x": np.nan,
        "sigma_y": np.nan,
        "theta": 0.0,
        "local_background": background,
        "fit_residual": residual,
    }


def _com_fallback(patch: np.ndarray) -> dict[str, float | bool | str]:
    x_c, y_c = _center_of_mass(patch)
    _, background = _positive_patch_for_com(patch)
    amplitude = float(patch.max() - background)
    return {
        "fit_success": False,
        "refinement_method": "com",
        "x_patch": x_c,
        "y_patch": y_c,
        "amplitude": amplitude,
        "sigma_x": np.nan,
        "sigma_y": np.nan,
        "theta": 0.0,
        "local_background": background,
        "fit_residual": np.nan,
    }


def _fallback_from_patch(patch: np.ndarray, config: RefinementConfig) -> dict[str, float | bool | str]:
    if config.fallback_to_quadratic:
        return _quadratic_fallback(patch)
    if config.fallback_to_com:
        return _com_fallback(patch)
    raise RuntimeError("Gaussian fitting failed and no fallback refinement method is enabled.")


def _fit_patch_legacy(
    image: np.ndarray,
    center_xy_global: tuple[float, float],
    half_window: int,
    origin_xy: tuple[int, int],
    config: RefinementConfig,
) -> tuple[dict[str, float | bool | str], tuple[int, int, int, int], int]:
    patch, bounds = extract_patch(
        image=image,
        center_xy_global=center_xy_global,
        half_window=_coerce_half_window(half_window),
        origin_xy=origin_xy,
    )
    if patch.size == 0:
        raise RuntimeError("Patch extraction returned an empty patch for legacy refinement.")

    patch = np.asarray(patch, dtype=float)
    try:
        fitted = _fit_gaussian_once(patch, config)
    except Exception:
        fitted = _fallback_from_patch(patch, config)
    return fitted, bounds, 1


def _fit_patch_adaptive(
    image: np.ndarray,
    center_xy_global: tuple[float, float],
    half_window: int,
    origin_xy: tuple[int, int],
    config: RefinementConfig,
) -> tuple[dict[str, float | bool | str], tuple[int, int, int, int], int, int]:
    current_half_window = _coerce_half_window(half_window)
    min_half_window = _coerce_half_window(config.min_patch_radius_px)
    max_attempts = 1 + max(int(config.gaussian_retry_count), 0)

    last_patch: np.ndarray | None = None
    last_bounds: tuple[int, int, int, int] | None = None
    last_attempt_half_window = current_half_window
    last_exception: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        patch, bounds = extract_patch(
            image=image,
            center_xy_global=center_xy_global,
            half_window=current_half_window,
            origin_xy=origin_xy,
        )
        if patch.size == 0:
            raise RuntimeError("Patch extraction returned an empty patch for adaptive refinement.")

        patch = np.asarray(patch, dtype=float)
        last_patch = patch
        last_bounds = bounds
        last_attempt_half_window = current_half_window

        try:
            fitted = _fit_gaussian_once(patch, config)
            if _gaussian_fit_is_reasonable(fitted, patch.shape, config):
                return fitted, bounds, attempt, current_half_window
        except Exception as exc:  # pragma: no cover - exercised via monkeypatched tests
            last_exception = exc

        if attempt >= max_attempts:
            break

        next_half_window = int(math.floor(current_half_window * float(config.gaussian_retry_shrink_factor)))
        if next_half_window >= current_half_window:
            next_half_window = current_half_window - 1
        if next_half_window < min_half_window:
            break
        current_half_window = next_half_window

    if last_patch is None or last_bounds is None:
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Adaptive refinement failed before a valid patch could be evaluated.")

    fallback = _fallback_from_patch(last_patch, config)
    return fallback, last_bounds, min(max_attempts, max(1, attempt)), last_attempt_half_window


def _fit_patch_overlap_shared(
    image: np.ndarray,
    center_a_xy_global: tuple[float, float],
    center_b_xy_global: tuple[float, float],
    half_window: int,
    origin_xy: tuple[int, int],
    config: RefinementConfig,
) -> tuple[
    dict[str, float | bool | str],
    dict[str, float | bool | str],
    tuple[int, int, int, int],
    int,
    int,
]:
    pair_center = (
        0.5 * (float(center_a_xy_global[0]) + float(center_b_xy_global[0])),
        0.5 * (float(center_a_xy_global[1]) + float(center_b_xy_global[1])),
    )
    pair_half_window = max(
        _coerce_half_window(half_window),
        _coerce_half_window(0.5 * math.hypot(center_a_xy_global[0] - center_b_xy_global[0], center_a_xy_global[1] - center_b_xy_global[1]) + config.min_sigma_px + 1.0),
    )
    patch, bounds = extract_patch(
        image=image,
        center_xy_global=pair_center,
        half_window=pair_half_window,
        origin_xy=origin_xy,
    )
    if patch.size == 0:
        raise RuntimeError("Patch extraction returned an empty patch for overlap refinement.")

    patch = np.asarray(patch, dtype=float)
    yy, xx = np.indices(patch.shape, dtype=float)
    border = border_values(patch)
    background0 = float(np.median(border))

    x0_local, _, y0_local, _ = bounds
    seed_a = (
        float(center_a_xy_global[0]) - origin_xy[0] - x0_local,
        float(center_a_xy_global[1]) - origin_xy[1] - y0_local,
    )
    seed_b = (
        float(center_b_xy_global[0]) - origin_xy[0] - x0_local,
        float(center_b_xy_global[1]) - origin_xy[1] - y0_local,
    )
    amp_a = float(max(patch[int(np.clip(round(seed_a[1]), 0, patch.shape[0] - 1)), int(np.clip(round(seed_a[0]), 0, patch.shape[1] - 1))] - background0, 1e-6))
    amp_b = float(max(patch[int(np.clip(round(seed_b[1]), 0, patch.shape[0] - 1)), int(np.clip(round(seed_b[0]), 0, patch.shape[1] - 1))] - background0, 1e-6))

    p0 = (
        amp_a,
        seed_a[0],
        seed_a[1],
        amp_b,
        seed_b[0],
        seed_b[1],
        config.initial_sigma_px,
        config.initial_sigma_px,
        0.0,
        background0,
    )
    lower_bounds = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        config.min_sigma_px,
        config.min_sigma_px,
        -np.pi / 2,
        patch.min() - max(amp_a, amp_b),
    ]
    upper_bounds = [
        np.inf,
        patch.shape[1] - 1,
        patch.shape[0] - 1,
        np.inf,
        patch.shape[1] - 1,
        patch.shape[0] - 1,
        config.max_sigma_px,
        config.max_sigma_px,
        np.pi / 2,
        patch.max(),
    ]
    params, _ = curve_fit(
        _double_gaussian_shared_shape,
        (xx.ravel(), yy.ravel()),
        patch.ravel(),
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=config.max_nfev,
    )
    fitted_patch = _double_gaussian_shared_shape((xx, yy), *params).reshape(patch.shape)
    residual = float(np.sqrt(np.mean((patch - fitted_patch) ** 2)) / (np.std(patch) + 1e-8))
    first = {
        "fit_success": True,
        "refinement_method": "overlap_gaussian",
        "x_patch": float(params[1]),
        "y_patch": float(params[2]),
        "amplitude": float(params[0]),
        "sigma_x": float(params[6]),
        "sigma_y": float(params[7]),
        "theta": float(params[8]),
        "local_background": float(params[9]),
        "fit_residual": residual,
    }
    second = {
        "fit_success": True,
        "refinement_method": "overlap_gaussian",
        "x_patch": float(params[4]),
        "y_patch": float(params[5]),
        "amplitude": float(params[3]),
        "sigma_x": float(params[6]),
        "sigma_y": float(params[7]),
        "theta": float(params[8]),
        "local_background": float(params[9]),
        "fit_residual": residual,
    }
    same_order_cost = math.hypot(first["x_patch"] - seed_a[0], first["y_patch"] - seed_a[1]) + math.hypot(second["x_patch"] - seed_b[0], second["y_patch"] - seed_b[1])
    swapped_order_cost = math.hypot(first["x_patch"] - seed_b[0], first["y_patch"] - seed_b[1]) + math.hypot(second["x_patch"] - seed_a[0], second["y_patch"] - seed_a[1])
    if swapped_order_cost < same_order_cost:
        first, second = second, first

    if not _gaussian_fit_is_reasonable(first, patch.shape, config) or not _gaussian_fit_is_reasonable(second, patch.shape, config):
        raise RuntimeError("Overlap Gaussian fit did not converge to a reasonable solution.")
    return first, second, bounds, 1, pair_half_window


def _resolve_overlap_pairs(points: pd.DataFrame, trigger_px: float | None) -> dict[int, int]:
    if trigger_px is None or len(points) < 2:
        return {}
    coords = points[["x_px", "y_px"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=2)
    pair_map: dict[int, int] = {}
    for idx, (distance, neighbor) in enumerate(zip(distances[:, 1], indices[:, 1], strict=True)):
        if not np.isfinite(distance) or distance > float(trigger_px):
            continue
        neighbor = int(neighbor)
        if neighbor == idx or neighbor < 0 or neighbor >= len(points):
            continue
        reverse_neighbor = int(indices[neighbor, 1])
        reverse_distance = float(distances[neighbor, 1])
        if reverse_neighbor != idx or not np.isfinite(reverse_distance) or reverse_distance > float(trigger_px):
            continue
        if idx in pair_map or neighbor in pair_map:
            continue
        left, right = sorted((idx, neighbor))
        pair_map[left] = right
        pair_map[right] = left
    return pair_map


def _resolve_overlap_pairs_by_mode(
    points: pd.DataFrame,
    trigger_px: float | None,
    nn_context_mode: str,
) -> dict[int, int]:
    mode = _normalize_nn_context_mode(nn_context_mode)
    if mode == "all":
        return _resolve_overlap_pairs(points, trigger_px)
    if trigger_px is None or len(points) < 2:
        return {}
    if "class_id" not in points.columns:
        raise ValueError("class_id is required when nn_context_mode='same_class'.")

    pair_map: dict[int, int] = {}
    class_ids = pd.Series(points["class_id"]).fillna(-1).astype(int)
    for class_id in sorted(class_ids.unique()):
        positions = np.flatnonzero(class_ids.to_numpy(dtype=int) == int(class_id))
        if len(positions) < 2:
            continue
        class_pairs = _resolve_overlap_pairs(points.iloc[positions].reset_index(drop=True), trigger_px)
        for local_idx, local_neighbor in class_pairs.items():
            pair_map[int(positions[local_idx])] = int(positions[local_neighbor])
    return pair_map


def _patch_to_global(
    bounds: tuple[int, int, int, int],
    origin_xy: tuple[int, int],
    x_patch: float,
    y_patch: float,
) -> tuple[float, float]:
    x0_local, _, y0_local, _ = bounds
    return (
        float(origin_xy[0] + x0_local + x_patch),
        float(origin_xy[1] + y0_local + y_patch),
    )


def _resolve_refinement_path(config: RefinementConfig, fitted: dict[str, float | bool | str]) -> str:
    if bool(fitted["fit_success"]):
        if fitted["refinement_method"] == "overlap_gaussian":
            return "overlap_shared_gaussian"
        return "adaptive_atomap" if config.mode == "adaptive_atomap" else "legacy_gaussian"
    if fitted["refinement_method"] == "quadratic":
        return "quadratic_fallback"
    if fitted["refinement_method"] == "com":
        return "com_fallback"
    return "legacy_gaussian"


def _build_result_record(
    row: pd.Series,
    fitted: dict[str, float | bool | str],
    bounds: tuple[int, int, int, int],
    origin_xy: tuple[int, int],
    nn_distance_px: float,
    adaptive_half_window_px: int,
    gaussian_attempt_count: int,
    gaussian_image_source: str,
    config: RefinementConfig,
    refinement_class_id: int | None = None,
    refinement_config_source: str | None = None,
    nn_context_mode: str | None = None,
) -> dict[str, float | bool | str]:
    x_fit, y_fit = _patch_to_global(
        bounds,
        origin_xy,
        float(fitted["x_patch"]),
        float(fitted["y_patch"]),
    )
    x_input = float(row["x_px"])
    y_input = float(row["y_px"])
    attempted_shift = math.hypot(x_fit - x_input, y_fit - y_input)
    max_shift = float(config.max_center_shift_px)
    center_shift_rejected = (
        not np.isfinite(x_fit)
        or not np.isfinite(y_fit)
        or not np.isfinite(attempted_shift)
        or attempted_shift > max_shift
    )
    if center_shift_rejected:
        x_global = x_input
        y_global = y_input
        position_source = "candidate_shift_guard"
    else:
        x_global = x_fit
        y_global = y_fit
        position_source = "refined"
    shift = math.hypot(x_global - x_input, y_global - y_input)
    residual = float(fitted["fit_residual"]) if not pd.isna(fitted["fit_residual"]) else 0.5
    quality = max(0.0, 1.0 - residual)
    quality *= 1.0 if np.isfinite(attempted_shift) and attempted_shift <= max_shift else 0.5
    if quality < config.quality_floor:
        quality = config.quality_floor

    record: dict[str, float | bool | str] = {
        "atom_id": int(row.get("atom_id", row.get("candidate_id", row.name))),
        "candidate_id": int(row.get("candidate_id", row.get("atom_id", row.name))),
        "x_px": x_global,
        "y_px": y_global,
        "x_input_px": x_input,
        "y_input_px": y_input,
        "x_fit_px": x_fit,
        "y_fit_px": y_fit,
        "amplitude": fitted["amplitude"],
        "local_background": fitted["local_background"],
        "sigma_x": fitted["sigma_x"],
        "sigma_y": fitted["sigma_y"],
        "theta": fitted["theta"],
        "fit_residual": fitted["fit_residual"],
        "fit_success": fitted["fit_success"],
        "refinement_method": fitted["refinement_method"],
        "quality_score": quality,
        "center_shift_px": shift,
        "attempted_center_shift_px": attempted_shift,
        "center_shift_rejected": bool(center_shift_rejected),
        "position_source": position_source,
        "nn_distance_px": float(nn_distance_px) if np.isfinite(nn_distance_px) else np.nan,
        "adaptive_half_window_px": int(adaptive_half_window_px),
        "gaussian_attempt_count": int(gaussian_attempt_count),
        "gaussian_image_source": gaussian_image_source,
        "refinement_path": _resolve_refinement_path(config, fitted),
    }
    for passthrough_column in (
        "column_role",
        "seed_channel",
        "detected_from_channels",
        "confirm_channel",
        "parent_heavy_id",
        "contrast_mode_used",
        "class_id",
        "class_name",
        "class_color",
        "class_confidence",
        "class_source",
        "class_reviewed",
    ):
        if passthrough_column in row.index:
            record[passthrough_column] = row[passthrough_column]
    if refinement_config_source is not None:
        record["refinement_class_id"] = int(refinement_class_id if refinement_class_id is not None else -1)
        record["refinement_config_source"] = str(refinement_config_source)
    if nn_context_mode is not None:
        record["nn_context_mode"] = str(nn_context_mode)
    return record


def _refinement_source_points(session: AnalysisSession, source_table: str) -> tuple[str, pd.DataFrame]:
    source = str(source_table).lower()
    if source == "candidate":
        table = session.candidate_points
    elif source == "refined":
        table = session.refined_points
    elif source == "curated":
        table = session.curated_points
    else:
        raise ValueError("source_table must be 'candidate', 'refined', or 'curated'.")
    if table.empty:
        raise ValueError(f"{source_table} points are required before refinement.")
    points = table.copy().reset_index(drop=True)
    if "atom_id" not in points.columns:
        points.insert(0, "atom_id", np.arange(len(points), dtype=int))
    if "candidate_id" not in points.columns:
        points.insert(1, "candidate_id", points["atom_id"].to_numpy(dtype=int))
    return source, points


def _class_id_from_row(row: pd.Series) -> int:
    if "class_id" not in row.index or pd.isna(row["class_id"]):
        return -1
    return int(row["class_id"])


def _refinement_config_with_override(
    default_config: RefinementConfig,
    override: RefinementConfig | dict[str, object] | None,
) -> RefinementConfig:
    if override is None:
        return default_config
    if isinstance(override, RefinementConfig):
        return override
    if not isinstance(override, dict):
        raise TypeError("class_refinement_overrides values must be dicts or RefinementConfig instances.")
    allowed = {field.name for field in fields(RefinementConfig)}
    unknown = sorted(str(key) for key in override if str(key) not in allowed)
    if unknown:
        raise ValueError(f"Unknown RefinementConfig override field(s): {unknown}")
    return replace(default_config, **{str(key): value for key, value in override.items()})


def _resolve_class_refinement_configs(
    points: pd.DataFrame,
    default_config: RefinementConfig,
    class_refinement_overrides: dict[int | str, RefinementConfig | dict[str, object]] | None,
) -> tuple[list[RefinementConfig], list[int], list[str]]:
    overrides = dict(class_refinement_overrides or {})
    configs: list[RefinementConfig] = []
    class_ids: list[int] = []
    sources: list[str] = []
    for _, row in points.iterrows():
        class_id = _class_id_from_row(row)
        override = None
        source = "default"
        for key in (class_id, str(class_id)):
            if key in overrides:
                override = overrides[key]
                source = f"class_{class_id}"
                break
        configs.append(_refinement_config_with_override(default_config, override))
        class_ids.append(class_id)
        sources.append(source)
    return configs, class_ids, sources


def _refine_point_table_with_configs(
    session: AnalysisSession,
    points: pd.DataFrame,
    configs: list[RefinementConfig],
    *,
    nn_context_points: pd.DataFrame | None = None,
    refinement_class_ids: list[int] | None = None,
    refinement_config_sources: list[str] | None = None,
    nn_context_mode: str = "all",
) -> pd.DataFrame:
    pixel_size = session.pixel_calibration.size
    unit = session.pixel_calibration.unit
    if len(configs) != len(points):
        raise ValueError("configs must contain one RefinementConfig per point.")
    mode = _normalize_nn_context_mode(nn_context_mode)
    nn_distances = (
        _resolve_nn_distances(points, nn_context_points)
        if nn_context_points is not None
        else _resolve_nn_distances_by_mode(points, mode)
    )
    trigger_values = [
        float(config.overlap_trigger_px)
        for config in configs
        if config.overlap_trigger_px is not None
    ]
    overlap_pairs = _resolve_overlap_pairs_by_mode(points, max(trigger_values) if trigger_values else None, mode)

    records: list[dict[str, float | bool | str]] = []
    consumed_pairs: set[int] = set()
    for pos, (_, row) in enumerate(points.iterrows()):
        if pos in consumed_pairs:
            continue
        config = configs[pos]
        refinement_class_id = refinement_class_ids[pos] if refinement_class_ids is not None else None
        refinement_config_source = (
            refinement_config_sources[pos] if refinement_config_sources is not None else None
        )
        nn_distance_px = float(nn_distances[pos]) if pos < len(nn_distances) else np.nan
        seed_channel = row.get("seed_channel")
        channel_name = str(seed_channel) if isinstance(seed_channel, str) and seed_channel in session.list_channels() else None
        processed_image, processed_origin, _ = _resolve_image_and_origin(session, "processed", channel_name)
        raw_image, raw_origin, _ = _resolve_image_and_origin(session, "raw", channel_name)
        gaussian_image, gaussian_origin, gaussian_image_source = _resolve_image_and_origin(
            session,
            config.gaussian_image_source,
            channel_name,
        )

        pair_index = overlap_pairs.get(pos)
        if (
            config.mode == "adaptive_atomap"
            and config.overlap_trigger_px is not None
            and pair_index is not None
            and pair_index not in consumed_pairs
            and pos < pair_index
            and configs[pair_index] == config
        ):
            paired_row = points.iloc[pair_index]
            paired_channel = paired_row.get("seed_channel")
            paired_channel_name = (
                str(paired_channel)
                if isinstance(paired_channel, str) and paired_channel in session.list_channels()
                else None
            )
            if paired_channel_name == channel_name:
                paired_nn_distance_px = float(nn_distances[pair_index]) if pair_index < len(nn_distances) else np.nan
                overlap_half_window = max(
                    _resolve_adaptive_half_window(nn_distance_px, config),
                    _resolve_adaptive_half_window(paired_nn_distance_px, config),
                )
                center_a_xy = (float(row["x_px"]), float(row["y_px"]))
                center_b_xy = (float(paired_row["x_px"]), float(paired_row["y_px"]))
                try:
                    fitted_a, fitted_b, bounds, attempt_count, final_half_window = _fit_patch_overlap_shared(
                        gaussian_image,
                        center_a_xy,
                        center_b_xy,
                        overlap_half_window,
                        gaussian_origin,
                        config,
                    )
                    records.append(
                        _build_result_record(
                            row=row,
                            fitted=fitted_a,
                            bounds=bounds,
                            origin_xy=gaussian_origin,
                            nn_distance_px=nn_distance_px,
                            adaptive_half_window_px=final_half_window,
                            gaussian_attempt_count=attempt_count,
                            gaussian_image_source=gaussian_image_source,
                            config=config,
                            refinement_class_id=refinement_class_id,
                            refinement_config_source=refinement_config_source,
                            nn_context_mode=mode,
                        )
                    )
                    records.append(
                        _build_result_record(
                            row=paired_row,
                            fitted=fitted_b,
                            bounds=bounds,
                            origin_xy=gaussian_origin,
                            nn_distance_px=paired_nn_distance_px,
                            adaptive_half_window_px=final_half_window,
                            gaussian_attempt_count=attempt_count,
                            gaussian_image_source=gaussian_image_source,
                            config=config,
                            refinement_class_id=refinement_class_ids[pair_index] if refinement_class_ids is not None else None,
                            refinement_config_source=(
                                refinement_config_sources[pair_index]
                                if refinement_config_sources is not None
                                else None
                            ),
                            nn_context_mode=mode,
                        )
                    )
                    consumed_pairs.update({pos, pair_index})
                    continue
                except Exception:
                    pass

        if config.mode == "adaptive_atomap":
            adaptive_half_window = _resolve_adaptive_half_window(nn_distance_px, config)
            com_half_window = adaptive_half_window if np.isfinite(nn_distance_px) else _coerce_half_window(config.com_half_window)
            gaussian_half_window = adaptive_half_window if np.isfinite(nn_distance_px) else _coerce_half_window(config.fit_half_window)

            center_xy = (float(row["x_px"]), float(row["y_px"]))
            center_xy = _refine_center_with_com(processed_image, center_xy, com_half_window, processed_origin)
            center_xy = _refine_center_with_com(raw_image, center_xy, com_half_window, raw_origin)
            fitted, bounds, attempt_count, final_half_window = _fit_patch_adaptive(
                gaussian_image,
                center_xy,
                gaussian_half_window,
                gaussian_origin,
                config,
            )
            records.append(
                _build_result_record(
                    row=row,
                    fitted=fitted,
                    bounds=bounds,
                    origin_xy=gaussian_origin,
                    nn_distance_px=nn_distance_px,
                    adaptive_half_window_px=final_half_window,
                    gaussian_attempt_count=attempt_count,
                    gaussian_image_source=gaussian_image_source,
                    config=config,
                    refinement_class_id=refinement_class_id,
                    refinement_config_source=refinement_config_source,
                    nn_context_mode=mode,
                )
            )
            continue

        fitted, bounds, attempt_count = _fit_patch_legacy(
            processed_image,
            (float(row["x_px"]), float(row["y_px"])),
            config.fit_half_window,
            processed_origin,
            config,
        )
        records.append(
            _build_result_record(
                row=row,
                fitted=fitted,
                bounds=bounds,
                origin_xy=processed_origin,
                nn_distance_px=nn_distance_px,
                adaptive_half_window_px=_coerce_half_window(config.fit_half_window),
                gaussian_attempt_count=attempt_count,
                gaussian_image_source="processed",
                config=config,
                refinement_class_id=refinement_class_id,
                refinement_config_source=refinement_config_source,
                nn_context_mode=mode,
            )
        )

    refined = pd.DataFrame(records)
    refined = attach_physical_coordinates(refined, pixel_size=pixel_size, unit=unit)
    return refined


def refine_point_table(
    session: AnalysisSession,
    points: pd.DataFrame,
    config: RefinementConfig,
    *,
    nn_context_points: pd.DataFrame | None = None,
    nn_context_mode: str = "all",
) -> pd.DataFrame:
    return _refine_point_table_with_configs(
        session,
        points,
        [config] * len(points),
        nn_context_points=nn_context_points,
        nn_context_mode=nn_context_mode,
    )


def refine_points(session: AnalysisSession, config: RefinementConfig) -> AnalysisSession:
    if session.candidate_points.empty:
        raise ValueError("Candidate points are required before refinement.")

    points = session.candidate_points.copy()
    if "atom_id" not in points.columns:
        points["atom_id"] = np.arange(len(points), dtype=int)

    refined = refine_point_table(session, points, config)
    session.refined_points = refined
    session.record_step(
        "refine_points",
        parameters=config,
        notes={"refined_count": len(refined), "mode": config.mode},
    )
    return session


def refine_points_by_class(
    session: AnalysisSession,
    default_config: RefinementConfig,
    class_refinement_overrides: dict[int | str, RefinementConfig | dict[str, object]] | None = None,
    *,
    source_table: str = "candidate",
    nn_context_mode: str = "all",
) -> AnalysisSession:
    _, points = _refinement_source_points(session, source_table)
    if "class_id" not in points.columns:
        raise ValueError("class_id is required for class-aware refinement; run classification first.")
    mode = _normalize_nn_context_mode(nn_context_mode)

    configs, class_ids, config_sources = _resolve_class_refinement_configs(
        points,
        default_config,
        class_refinement_overrides,
    )
    refined = _refine_point_table_with_configs(
        session,
        points,
        configs,
        refinement_class_ids=class_ids,
        refinement_config_sources=config_sources,
        nn_context_mode=mode,
    )
    session.refined_points = refined
    session.record_step(
        "refine_points_by_class",
        parameters={
            "default_config": default_config,
            "class_refinement_overrides": class_refinement_overrides or {},
            "source_table": source_table,
            "nn_context_mode": mode,
        },
        notes={
            "refined_count": len(refined),
            "class_ids": sorted(set(class_ids)),
            "config_sources": sorted(set(config_sources)),
            "nn_context_mode": mode,
        },
    )
    return session
