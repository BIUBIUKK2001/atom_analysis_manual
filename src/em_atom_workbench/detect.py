from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay, QhullError, cKDTree
from skimage.feature import peak_local_max

from .refine import refine_point_table
from .session import AnalysisSession, DetectionConfig, HfO2MultichannelDetectionConfig
from .utils import border_values, extract_patch


def _candidate_columns() -> list[str]:
    return [
        "candidate_id",
        "x_px",
        "y_px",
        "x_local_px",
        "y_local_px",
        "center_intensity",
        "local_background",
        "prominence",
        "local_snr",
        "score",
        "contrast_mode_used",
        "column_role",
        "seed_channel",
        "detected_from_channels",
        "confirm_channel",
        "parent_heavy_id",
        "support_score",
        "confirm_score",
        "class_id",
        "class_name",
        "class_color",
        "class_confidence",
        "class_source",
        "class_reviewed",
    ]


def _empty_candidate_table() -> pd.DataFrame:
    return pd.DataFrame(columns=_candidate_columns())


def _resolve_detection_modes(contrast_mode: str) -> list[str]:
    if contrast_mode == "mixed_contrast":
        return ["bright_peak", "dark_dip"]
    return [contrast_mode]


def _prepare_detection_image(
    image: np.ndarray,
    *,
    contrast_mode: str,
    gaussian_sigma: float,
    image_already_inverted: bool = False,
) -> np.ndarray:
    working = np.asarray(image, dtype=float).copy()
    if contrast_mode == "dark_dip" and not image_already_inverted:
        working = -working
    if gaussian_sigma > 0:
        working = gaussian_filter(working, sigma=gaussian_sigma)
    return working


def _candidate_metrics(image: np.ndarray, coords_local: np.ndarray, config: DetectionConfig) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for y_local, x_local in coords_local:
        patch, _ = extract_patch(
            image=image,
            center_xy_global=(float(x_local), float(y_local)),
            half_window=config.patch_radius,
            origin_xy=(0, 0),
        )
        border = border_values(patch)
        if border.size == 0:
            continue
        center_value = float(image[int(round(y_local)), int(round(x_local))])
        background = float(np.median(border))
        noise = float(np.std(border) + 1e-8)
        prominence = center_value - background
        local_snr = prominence / noise
        records.append(
            {
                "x_local_px": float(x_local),
                "y_local_px": float(y_local),
                "center_intensity": center_value,
                "local_background": background,
                "prominence": prominence,
                "local_snr": local_snr,
            }
        )
    return pd.DataFrame(records)


def _detect_single_mode(
    working_image: np.ndarray,
    mode: str,
    config: DetectionConfig,
) -> pd.DataFrame:
    coords_local = peak_local_max(
        working_image,
        min_distance=config.min_distance,
        threshold_abs=config.threshold_abs,
        threshold_rel=config.threshold_rel,
        exclude_border=config.edge_margin,
        num_peaks=config.max_candidates if config.max_candidates is not None else np.inf,
    )
    if coords_local.size == 0:
        return pd.DataFrame(columns=["x_local_px", "y_local_px", "score"])

    candidates = _candidate_metrics(working_image, coords_local, config)
    if candidates.empty:
        return candidates

    candidates["score"] = candidates["prominence"] * np.maximum(candidates["local_snr"], 0.0)
    candidates["contrast_mode_used"] = mode
    candidates = candidates[
        (candidates["prominence"] >= config.min_prominence)
        & (candidates["local_snr"] >= config.min_snr)
    ].reset_index(drop=True)
    return candidates


def _deduplicate_candidates(candidates: pd.DataFrame, radius: float) -> pd.DataFrame:
    if candidates.empty or len(candidates) == 1:
        return candidates.reset_index(drop=True)

    coords = candidates[["x_local_px", "y_local_px"]].to_numpy()
    tree = cKDTree(coords)
    visited: set[int] = set()
    keep_indices: list[int] = []

    for idx in np.argsort(-candidates["score"].to_numpy()):
        idx = int(idx)
        if idx in visited:
            continue
        neighbors = tree.query_ball_point(coords[idx], r=radius)
        visited.update(int(neighbor) for neighbor in neighbors)
        keep_indices.append(idx)

    return candidates.iloc[sorted(keep_indices)].reset_index(drop=True)


def _align_local_coordinates(candidates: pd.DataFrame, origin_xy: tuple[int, int]) -> pd.DataFrame:
    aligned = candidates.copy()
    origin_x, origin_y = origin_xy
    aligned["x_local_px"] = aligned["x_px"] - origin_x
    aligned["y_local_px"] = aligned["y_px"] - origin_y
    return aligned


def _finalize_candidate_table(
    detected_tables: list[pd.DataFrame],
    *,
    origin_xy: tuple[int, int],
    config: DetectionConfig,
    column_role: str,
    seed_channel: str,
    confirm_channel: Any = pd.NA,
    parent_heavy_id: Any = pd.NA,
) -> pd.DataFrame:
    if not detected_tables:
        return _empty_candidate_table()

    merged = pd.concat(detected_tables, ignore_index=True)
    dedupe_radius = config.dedupe_radius_px or max(config.min_distance * 0.5, 1.0)
    merged = _deduplicate_candidates(merged, radius=dedupe_radius)

    origin_x, origin_y = origin_xy
    merged["x_px"] = merged["x_local_px"] + origin_x
    merged["y_px"] = merged["y_local_px"] + origin_y
    merged.insert(0, "candidate_id", np.arange(len(merged), dtype=int))
    merged["column_role"] = column_role
    merged["seed_channel"] = seed_channel
    merged["detected_from_channels"] = [[seed_channel] for _ in range(len(merged))]
    merged["confirm_channel"] = confirm_channel
    merged["parent_heavy_id"] = parent_heavy_id
    merged["support_score"] = merged["score"]
    merged["confirm_score"] = np.nan
    merged["class_id"] = pd.NA
    merged["class_name"] = pd.NA
    merged["class_color"] = pd.NA
    merged["class_confidence"] = np.nan
    merged["class_source"] = pd.NA
    merged["class_reviewed"] = False

    for column in _candidate_columns():
        if column not in merged.columns:
            merged[column] = pd.NA
    return merged[_candidate_columns()]


def _detect_candidates_on_channel(
    session: AnalysisSession,
    config: DetectionConfig,
    *,
    channel_name: str,
    column_role: str = "atomic_column",
) -> pd.DataFrame:
    channel_state = session.get_channel_state(channel_name)
    image = session.get_processed_image(channel_name)
    origin_xy = session.get_processed_origin(channel_name)

    already_inverted = bool(channel_state.preprocess_result) and channel_state.contrast_mode == "dark_dip"
    candidate_tables: list[pd.DataFrame] = []
    for mode in _resolve_detection_modes(config.contrast_mode):
        working = _prepare_detection_image(
            image,
            contrast_mode=mode,
            gaussian_sigma=config.gaussian_sigma,
            image_already_inverted=already_inverted and mode == "dark_dip",
        )
        detected = _detect_single_mode(working, mode, config)
        if not detected.empty:
            candidate_tables.append(detected)

    return _finalize_candidate_table(
        candidate_tables,
        origin_xy=origin_xy,
        config=config,
        column_role=column_role,
        seed_channel=channel_name,
    )


def _build_single_channel_session(session: AnalysisSession, channel_name: str) -> AnalysisSession:
    channel_state = session.get_channel_state(channel_name)
    temp = AnalysisSession(
        name=f"{session.name}_{channel_name}",
        input_path=channel_state.input_path,
        dataset_index=channel_state.dataset_index,
        raw_image=channel_state.raw_image,
        raw_metadata=channel_state.raw_metadata,
        pixel_calibration=session.pixel_calibration,
        contrast_mode=channel_state.contrast_mode,
        primary_channel=channel_name,
    )
    temp.set_channel_state(
        channel_name,
        input_path=channel_state.input_path,
        dataset_index=channel_state.dataset_index,
        raw_image=channel_state.raw_image,
        raw_metadata=channel_state.raw_metadata,
        preprocess_result=channel_state.preprocess_result,
        contrast_mode=channel_state.contrast_mode,
    )
    temp.set_primary_channel(channel_name)
    return temp


def _refine_heavy_candidates(
    session: AnalysisSession,
    heavy_candidates: pd.DataFrame,
    config: HfO2MultichannelDetectionConfig,
) -> pd.DataFrame:
    if heavy_candidates.empty:
        return heavy_candidates.copy()
    temp_session = _build_single_channel_session(session, config.heavy_channel)
    refined = refine_point_table(temp_session, heavy_candidates.copy(), config.heavy_refinement)
    merged = heavy_candidates.merge(
        refined[["candidate_id", "x_px", "y_px"]],
        on="candidate_id",
        how="left",
        suffixes=("", "_refined"),
    )
    merged["x_px"] = merged["x_px_refined"].fillna(merged["x_px"])
    merged["y_px"] = merged["y_px_refined"].fillna(merged["y_px"])
    merged = merged.drop(columns=["x_px_refined", "y_px_refined"])
    return _align_local_coordinates(merged, session.get_processed_origin(config.heavy_channel))


def _normalize_hfo2_heavy_candidates(
    session: AnalysisSession,
    heavy_candidates: pd.DataFrame,
    config: HfO2MultichannelDetectionConfig,
) -> pd.DataFrame:
    heavy = heavy_candidates.copy().reset_index(drop=True)
    if heavy.empty:
        return _empty_candidate_table()
    if "x_local_px" not in heavy.columns or "y_local_px" not in heavy.columns:
        heavy = _align_local_coordinates(heavy, session.get_processed_origin(config.heavy_channel))
    if "score" not in heavy.columns:
        heavy["score"] = np.nan
    if "support_score" not in heavy.columns:
        heavy["support_score"] = heavy["score"]
    if "confirm_score" not in heavy.columns:
        heavy["confirm_score"] = np.nan
    if "contrast_mode_used" not in heavy.columns:
        heavy["contrast_mode_used"] = "manual_edit"
    heavy["column_role"] = "heavy_atom"
    heavy["seed_channel"] = config.heavy_channel
    heavy["confirm_channel"] = pd.NA
    heavy["parent_heavy_id"] = pd.NA
    for column in _candidate_columns():
        if column not in heavy.columns:
            heavy[column] = pd.NA
    heavy = heavy[_candidate_columns()]
    heavy["candidate_id"] = np.arange(len(heavy), dtype=int)
    return heavy


def _detect_hfo2_heavy_candidate_table(
    session: AnalysisSession,
    config: HfO2MultichannelDetectionConfig,
) -> pd.DataFrame:
    heavy_candidates = _detect_candidates_on_channel(
        session,
        config.heavy_detection,
        channel_name=config.heavy_channel,
        column_role="heavy_atom",
    )
    heavy_candidates = _refine_heavy_candidates(session, heavy_candidates, config)
    return _normalize_hfo2_heavy_candidates(session, heavy_candidates, config)


def _median_spacing(points: pd.DataFrame) -> float:
    if len(points) < 2:
        return 0.0
    coords = points[["x_px", "y_px"]].to_numpy(dtype=float)
    distances, _ = cKDTree(coords).query(coords, k=2)
    nn = np.asarray(distances[:, 1], dtype=float)
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if nn.size == 0:
        return 0.0
    return float(np.median(nn))


def _heavy_edges(points: pd.DataFrame, *, neighbor_count: int, max_distance: float) -> list[tuple[int, int]]:
    if len(points) < 2:
        return []
    coords = points[["x_px", "y_px"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=min(neighbor_count + 1, len(points)))
    neighbor_sets: dict[int, set[int]] = {}
    for idx, (dist_row, ind_row) in enumerate(zip(np.atleast_2d(distances), np.atleast_2d(indices), strict=True)):
        neighbor_sets[idx] = {
            int(ind)
            for dist, ind in zip(dist_row[1:], ind_row[1:], strict=True)
            if np.isfinite(dist) and 0 < dist <= max_distance
        }

    edges: set[tuple[int, int]] = set()
    for idx, neighbors in neighbor_sets.items():
        for neighbor in neighbors:
            if idx in neighbor_sets.get(neighbor, set()):
                edges.add(tuple(sorted((int(idx), int(neighbor)))))
    return sorted(edges)


def _void_centers(points: pd.DataFrame, *, max_edge_length: float) -> np.ndarray:
    if len(points) < 3:
        return np.empty((0, 2), dtype=float)
    coords = points[["x_px", "y_px"]].to_numpy(dtype=float)
    try:
        simplices = Delaunay(coords).simplices
    except QhullError:
        return np.empty((0, 2), dtype=float)

    centers: list[np.ndarray] = []
    for simplex in simplices:
        triangle = coords[np.asarray(simplex, dtype=int)]
        edge_lengths = [
            np.linalg.norm(triangle[0] - triangle[1]),
            np.linalg.norm(triangle[1] - triangle[2]),
            np.linalg.norm(triangle[0] - triangle[2]),
        ]
        if max(edge_lengths) <= max_edge_length:
            centers.append(np.mean(triangle, axis=0))
    if not centers:
        return np.empty((0, 2), dtype=float)
    return np.asarray(centers, dtype=float)


def _diagonal_void_midpoints(points: pd.DataFrame, *, min_distance: float, max_distance: float) -> np.ndarray:
    if len(points) < 2:
        return np.empty((0, 2), dtype=float)
    coords = points[["x_px", "y_px"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    centers: list[np.ndarray] = []
    for idx, coord in enumerate(coords):
        for neighbor in tree.query_ball_point(coord, r=max_distance):
            neighbor = int(neighbor)
            if neighbor <= idx:
                continue
            distance = float(np.linalg.norm(coord - coords[neighbor]))
            if min_distance <= distance <= max_distance:
                centers.append(0.5 * (coord + coords[neighbor]))
    if not centers:
        return np.empty((0, 2), dtype=float)
    return np.asarray(centers, dtype=float)


def _proposal_centers(heavy_points: pd.DataFrame, config: HfO2MultichannelDetectionConfig) -> np.ndarray:
    spacing = _median_spacing(heavy_points)
    if spacing <= 0:
        return np.empty((0, 2), dtype=float)

    max_pair_distance = spacing * float(config.midpoint_max_distance_factor)
    max_void_edge = spacing * float(config.void_max_edge_factor)
    coords = heavy_points[["x_px", "y_px"]].to_numpy(dtype=float)

    centers: list[np.ndarray] = []
    for left_idx, right_idx in _heavy_edges(
        heavy_points,
        neighbor_count=config.heavy_neighbor_count,
        max_distance=max_pair_distance,
    ):
        centers.append((coords[left_idx] + coords[right_idx]) * 0.5)

    void_centers = _void_centers(heavy_points, max_edge_length=max_void_edge)
    if void_centers.size:
        centers.extend(void_centers)
    diagonal_midpoints = _diagonal_void_midpoints(
        heavy_points,
        min_distance=spacing * 1.2,
        max_distance=spacing * float(config.void_max_edge_factor),
    )
    if diagonal_midpoints.size:
        centers.extend(diagonal_midpoints)

    if not centers:
        return np.empty((0, 2), dtype=float)

    proposal_frame = pd.DataFrame(centers, columns=["x_local_px", "y_local_px"])
    proposal_frame["score"] = 1.0
    deduped = _deduplicate_candidates(proposal_frame, radius=float(config.proposal_dedupe_radius_px))
    return deduped[["x_local_px", "y_local_px"]].to_numpy(dtype=float)


def _suppress_heavy_response(
    image: np.ndarray,
    heavy_points: pd.DataFrame,
    *,
    origin_xy: tuple[int, int],
    radius_px: float,
    sigma_px: float,
) -> np.ndarray:
    suppressed = np.asarray(image, dtype=float).copy()
    half_window = max(int(round(radius_px * 2.0)), int(round(sigma_px * 3.0)), 2)
    for _, row in heavy_points.iterrows():
        patch, bounds = extract_patch(
            image=suppressed,
            center_xy_global=(float(row["x_px"]), float(row["y_px"])),
            half_window=half_window,
            origin_xy=origin_xy,
        )
        if patch.size == 0:
            continue
        border = border_values(patch)
        background = float(np.median(border)) if border.size else float(np.median(suppressed))
        amplitude = float(np.max(patch) - background)
        if amplitude <= 0:
            continue

        x0_local, x1_local, y0_local, y1_local = bounds
        yy, xx = np.indices((y1_local - y0_local, x1_local - x0_local), dtype=float)
        center_x = float(row["x_px"]) - origin_xy[0] - x0_local
        center_y = float(row["y_px"]) - origin_xy[1] - y0_local
        gaussian = amplitude * np.exp(-0.5 * (((xx - center_x) / sigma_px) ** 2 + ((yy - center_y) / sigma_px) ** 2))
        suppressed[y0_local:y1_local, x0_local:x1_local] = np.maximum(
            suppressed[y0_local:y1_local, x0_local:x1_local] - gaussian,
            background,
        )
    return suppressed


def _metric_at_global_xy(
    image: np.ndarray,
    origin_xy: tuple[int, int],
    xy_global: tuple[float, float],
    config: DetectionConfig,
) -> dict[str, float]:
    x_local = float(xy_global[0]) - origin_xy[0]
    y_local = float(xy_global[1]) - origin_xy[1]
    metrics = _candidate_metrics(
        image,
        np.asarray([[y_local, x_local]], dtype=float),
        config,
    )
    if metrics.empty:
        return {
            "x_local_px": x_local,
            "y_local_px": y_local,
            "center_intensity": np.nan,
            "local_background": np.nan,
            "prominence": np.nan,
            "local_snr": np.nan,
        }
    return metrics.iloc[0].to_dict()


def _prepared_channel_image(
    session: AnalysisSession,
    *,
    channel_name: str,
    config: DetectionConfig,
) -> tuple[np.ndarray, tuple[int, int]]:
    channel_state = session.get_channel_state(channel_name)
    image = session.get_processed_image(channel_name)
    already_inverted = bool(channel_state.preprocess_result) and channel_state.contrast_mode == "dark_dip"
    working = _prepare_detection_image(
        image,
        contrast_mode=config.contrast_mode,
        gaussian_sigma=config.gaussian_sigma,
        image_already_inverted=already_inverted and config.contrast_mode == "dark_dip",
    )
    return working, session.get_processed_origin(channel_name)


def _detect_light_candidates(
    session: AnalysisSession,
    heavy_points: pd.DataFrame,
    config: HfO2MultichannelDetectionConfig,
) -> pd.DataFrame:
    proposals = _proposal_centers(heavy_points, config)
    if proposals.size == 0:
        return _empty_candidate_table()

    light_image, light_origin = _prepared_channel_image(
        session,
        channel_name=config.light_channel,
        config=config.light_detection,
    )
    suppressed = _suppress_heavy_response(
        light_image,
        heavy_points,
        origin_xy=light_origin,
        radius_px=float(config.heavy_suppression_radius_px),
        sigma_px=max(float(config.heavy_suppression_sigma_px), 1e-3),
    )

    confirm_image: np.ndarray | None = None
    confirm_origin = (0, 0)
    confirm_channel_name = config.confirm_channel if config.confirm_channel in session.list_channels() else None
    if confirm_channel_name is not None:
        confirm_image, confirm_origin = _prepared_channel_image(
            session,
            channel_name=confirm_channel_name,
            config=config.light_detection,
        )

    heavy_coords = heavy_points[["x_px", "y_px"]].to_numpy(dtype=float)
    heavy_ids = heavy_points["candidate_id"].to_numpy(dtype=int)
    heavy_tree = cKDTree(heavy_coords) if len(heavy_coords) else None

    records: list[dict[str, Any]] = []
    for proposal in proposals:
        proposal_xy = (float(proposal[0]), float(proposal[1]))
        patch, bounds = extract_patch(
            image=suppressed,
            center_xy_global=proposal_xy,
            half_window=max(int(config.proposal_window_radius_px), 1),
            origin_xy=light_origin,
        )
        if patch.size == 0:
            continue
        y_patch, x_patch = np.unravel_index(np.argmax(patch), patch.shape)
        candidate_xy = (
            float(light_origin[0] + bounds[0] + x_patch),
            float(light_origin[1] + bounds[2] + y_patch),
        )
        if np.hypot(candidate_xy[0] - proposal_xy[0], candidate_xy[1] - proposal_xy[1]) > float(config.proposal_max_offset_px):
            continue

        parent_heavy_id = pd.NA
        heavy_distance = np.inf
        if heavy_tree is not None:
            heavy_distance, heavy_idx = heavy_tree.query(np.asarray(candidate_xy, dtype=float), k=1)
            if np.isfinite(heavy_distance):
                parent_heavy_id = int(heavy_ids[int(heavy_idx)])
        if heavy_distance < float(config.min_light_heavy_separation_px):
            continue

        metrics = _metric_at_global_xy(suppressed, light_origin, candidate_xy, config.light_detection)
        prominence = float(metrics["prominence"])
        local_snr = float(metrics["local_snr"])
        score = float(prominence * max(local_snr, 0.0))
        strong_support = prominence >= float(config.light_detection.min_prominence) and local_snr >= float(config.light_detection.min_snr)
        weak_support = prominence >= float(config.weak_min_prominence) and local_snr >= float(config.weak_min_snr)

        confirm_support = False
        confirm_score = np.nan
        confirm_channel_value: Any = pd.NA
        if confirm_image is not None:
            confirm_metrics = _metric_at_global_xy(confirm_image, confirm_origin, candidate_xy, config.light_detection)
            confirm_prominence = float(confirm_metrics["prominence"])
            confirm_snr = float(confirm_metrics["local_snr"])
            confirm_score = float(confirm_prominence * max(confirm_snr, 0.0))
            confirm_support = (
                confirm_prominence >= float(config.confirm_min_prominence)
                and confirm_snr >= float(config.confirm_min_snr)
            )
            if confirm_support:
                confirm_channel_value = confirm_channel_name

        if not strong_support and not (weak_support and (confirm_image is None or confirm_support)):
            continue

        record = dict(metrics)
        record.update(
            {
                "x_px": candidate_xy[0],
                "y_px": candidate_xy[1],
                "score": score,
                "contrast_mode_used": config.light_detection.contrast_mode,
                "column_role": "light_atom",
                "seed_channel": config.light_channel,
                "confirm_channel": confirm_channel_value,
                "parent_heavy_id": parent_heavy_id,
                "support_score": score,
                "confirm_score": confirm_score,
            }
        )
        records.append(record)

    if not records:
        return _empty_candidate_table()

    light_candidates = pd.DataFrame(records)
    light_candidates = _align_local_coordinates(light_candidates, light_origin)
    dedupe_radius = config.light_detection.dedupe_radius_px or max(config.light_detection.min_distance * 0.5, 1.0)
    light_candidates = _deduplicate_candidates(light_candidates, radius=dedupe_radius)
    light_candidates.insert(0, "candidate_id", np.arange(len(light_candidates), dtype=int))
    for column in _candidate_columns():
        if column not in light_candidates.columns:
            light_candidates[column] = pd.NA
    return light_candidates[_candidate_columns()]


def detect_candidates(
    session: AnalysisSession,
    config: DetectionConfig,
    channel_name: str | None = None,
) -> AnalysisSession:
    channel_key = str(channel_name or session.primary_channel)
    session.clear_downstream_results("detect")
    merged = _detect_candidates_on_channel(session, config, channel_name=channel_key)
    session.candidate_points = merged
    session.set_stage("detected")
    origin_x, origin_y = session.get_processed_origin(channel_key)
    session.record_step(
        "detect_candidates",
        parameters={"channel_name": channel_key, "config": config},
        notes={"candidate_count": len(merged), "origin_x": origin_x, "origin_y": origin_y},
    )
    return session


def _merge_multichannel_candidate_tables(
    candidate_tables: list[pd.DataFrame],
    *,
    dedupe_radius_px: float,
) -> pd.DataFrame:
    if not candidate_tables:
        return _empty_candidate_table()
    merged = pd.concat(candidate_tables, ignore_index=True)
    if merged.empty:
        return _empty_candidate_table()

    coords = merged[["x_px", "y_px"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    visited: set[int] = set()
    keep_rows: list[pd.Series] = []
    scores = pd.to_numeric(merged.get("score", pd.Series(np.nan, index=merged.index)), errors="coerce").fillna(-np.inf)
    for idx in np.argsort(-scores.to_numpy(dtype=float)):
        idx = int(idx)
        if idx in visited:
            continue
        neighbors = [int(neighbor) for neighbor in tree.query_ball_point(coords[idx], r=float(dedupe_radius_px))]
        visited.update(neighbors)
        cluster = merged.iloc[neighbors].copy()
        cluster_scores = pd.to_numeric(cluster.get("score", pd.Series(np.nan, index=cluster.index)), errors="coerce")
        best_label = int(cluster_scores.fillna(-np.inf).idxmax())
        representative = merged.loc[best_label].copy()
        channels: list[str] = []
        for value in cluster.get("detected_from_channels", pd.Series(dtype=object)):
            if isinstance(value, (list, tuple, set)):
                channels.extend(str(item) for item in value)
            elif pd.notna(value):
                channels.append(str(value))
        if "seed_channel" in cluster.columns:
            channels.extend(str(value) for value in cluster["seed_channel"].dropna().tolist())
        representative["detected_from_channels"] = sorted(set(channels))
        representative["support_score"] = float(np.nanmax(cluster_scores.to_numpy(dtype=float)))
        keep_rows.append(representative)

    if not keep_rows:
        return _empty_candidate_table()
    result = pd.DataFrame(keep_rows).reset_index(drop=True)
    result["candidate_id"] = np.arange(len(result), dtype=int)
    for column in _candidate_columns():
        if column not in result.columns:
            result[column] = pd.NA
    return result[_candidate_columns()]


def detect_multichannel_candidates(
    session: AnalysisSession,
    configs_by_channel: dict[str, DetectionConfig],
    *,
    dedupe_radius_px: float | None = None,
) -> AnalysisSession:
    """Detect atom-column candidates from one or more named channels.

    This generic entry point does not assign chemistry-specific roles. All
    detected points use ``column_role='atom_column'`` and remain unclassified
    until the classification stage.
    """
    if not configs_by_channel:
        raise ValueError("configs_by_channel must contain at least one channel.")

    session.clear_downstream_results("detect")
    candidate_tables: list[pd.DataFrame] = []
    radii: list[float] = []
    for channel_name, config in configs_by_channel.items():
        session.get_channel_state(channel_name)
        table = _detect_candidates_on_channel(
            session,
            config,
            channel_name=str(channel_name),
            column_role="atom_column",
        )
        if not table.empty:
            candidate_tables.append(table)
        radii.append(float(config.dedupe_radius_px or max(config.min_distance * 0.5, 1.0)))

    merge_radius = float(dedupe_radius_px if dedupe_radius_px is not None else max(radii or [1.0]))
    merged = _merge_multichannel_candidate_tables(candidate_tables, dedupe_radius_px=merge_radius)
    session.candidate_points = merged
    session.set_stage("detected")
    session.record_step(
        "detect_multichannel_candidates",
        parameters={
            "channel_names": list(configs_by_channel.keys()),
            "configs_by_channel": configs_by_channel,
            "dedupe_radius_px": merge_radius,
        },
        notes={"candidate_count": len(merged)},
    )
    return session


def detect_hfo2_heavy_candidates(
    session: AnalysisSession,
    config: HfO2MultichannelDetectionConfig,
) -> AnalysisSession:
    if config.candidate_mode != "hfo2_multichannel":
        raise ValueError(f"Unsupported candidate_mode: {config.candidate_mode}")
    session.get_channel_state(config.heavy_channel)

    session.clear_downstream_results("detect")
    heavy_candidates = _detect_hfo2_heavy_candidate_table(session, config)
    session.candidate_points = heavy_candidates
    session.set_stage("heavy_reviewed")
    session.record_step(
        "detect_hfo2_heavy_candidates",
        parameters=config,
        notes={
            "candidate_count": len(session.candidate_points),
            "heavy_count": int(len(session.candidate_points)),
            "channel_used": config.heavy_channel,
        },
    )
    return session


def detect_hfo2_light_candidates(
    session: AnalysisSession,
    config: HfO2MultichannelDetectionConfig,
    heavy_points: pd.DataFrame | None = None,
) -> AnalysisSession:
    if config.candidate_mode != "hfo2_multichannel":
        raise ValueError(f"Unsupported candidate_mode: {config.candidate_mode}")
    session.get_channel_state(config.light_channel)

    if heavy_points is None:
        if "column_role" in session.candidate_points.columns:
            heavy_source = session.candidate_points.loc[session.candidate_points["column_role"] == "heavy_atom"].copy()
        else:
            heavy_source = pd.DataFrame(columns=session.candidate_points.columns)
    else:
        heavy_source = heavy_points.copy()
    if heavy_source.empty:
        raise ValueError("Heavy candidates are required before running HfO2 light-column detection.")

    session.clear_downstream_results("detect")
    heavy_candidates = _normalize_hfo2_heavy_candidates(session, heavy_source, config)
    light_candidates = _detect_light_candidates(session, heavy_candidates, config)
    if not light_candidates.empty:
        light_candidates["candidate_id"] = np.arange(len(light_candidates), dtype=int) + len(heavy_candidates)

    merged = (
        pd.concat([heavy_candidates, light_candidates], ignore_index=True)
        if not light_candidates.empty
        else heavy_candidates
    )
    session.candidate_points = merged.reset_index(drop=True)
    session.set_stage("detected")
    session.record_step(
        "detect_hfo2_light_candidates",
        parameters=config,
        notes={
            "candidate_count": len(session.candidate_points),
            "heavy_count": int((session.candidate_points.get("column_role") == "heavy_atom").sum()) if not session.candidate_points.empty else 0,
            "light_count": int((session.candidate_points.get("column_role") == "light_atom").sum()) if not session.candidate_points.empty else 0,
            "channels_used": [config.heavy_channel, config.light_channel, config.confirm_channel],
        },
    )
    return session


def detect_hfo2_multichannel_candidates(
    session: AnalysisSession,
    config: HfO2MultichannelDetectionConfig,
) -> AnalysisSession:
    if config.candidate_mode != "hfo2_multichannel":
        raise ValueError(f"Unsupported candidate_mode: {config.candidate_mode}")
    for required_channel in (config.heavy_channel, config.light_channel):
        session.get_channel_state(required_channel)

    session.clear_downstream_results("detect")
    heavy_candidates = _detect_hfo2_heavy_candidate_table(session, config)
    light_candidates = _detect_light_candidates(session, heavy_candidates, config)
    if not light_candidates.empty:
        light_candidates["candidate_id"] = np.arange(len(light_candidates), dtype=int) + len(heavy_candidates)

    merged = pd.concat([heavy_candidates, light_candidates], ignore_index=True) if not light_candidates.empty else heavy_candidates
    session.candidate_points = merged.reset_index(drop=True)
    session.set_stage("detected")
    session.record_step(
        "detect_hfo2_multichannel_candidates",
        parameters=config,
        notes={
            "candidate_count": len(session.candidate_points),
            "heavy_count": int((session.candidate_points.get("column_role") == "heavy_atom").sum()) if not session.candidate_points.empty else 0,
            "light_count": int((session.candidate_points.get("column_role") == "light_atom").sum()) if not session.candidate_points.empty else 0,
            "channels_used": [config.heavy_channel, config.light_channel, config.confirm_channel],
        },
    )
    return session
