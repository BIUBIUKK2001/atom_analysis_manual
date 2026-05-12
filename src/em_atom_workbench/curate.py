from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .refine import refine_point_table
from .session import AnalysisSession, CurationConfig, RefinementConfig

_CANDIDATE_TABLE_COLUMNS = [
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


def _flag_duplicates(points: pd.DataFrame, radius: float) -> pd.Series:
    if points.empty:
        return pd.Series(dtype=bool)
    coords = points[["x_px", "y_px"]].to_numpy()
    tree = cKDTree(coords)
    duplicate = np.zeros(len(points), dtype=bool)
    for idx in range(len(points)):
        neighbors = tree.query_ball_point(coords[idx], r=radius)
        if any(neighbor < idx for neighbor in neighbors if neighbor != idx):
            duplicate[idx] = True
    return pd.Series(duplicate, index=points.index)


def _flag_edge_points(points: pd.DataFrame, shape: tuple[int, int], margin: int) -> pd.Series:
    if points.empty:
        return pd.Series(dtype=bool)
    return (
        (points["x_px"] < margin)
        | (points["y_px"] < margin)
        | (points["x_px"] > shape[1] - margin - 1)
        | (points["y_px"] > shape[0] - margin - 1)
    )


def _flag_spacing_violations(points: pd.DataFrame, min_spacing_px: float | None) -> pd.Series:
    if points.empty or min_spacing_px is None or len(points) < 2:
        return pd.Series(False, index=points.index)
    coords = points[["x_px", "y_px"]].to_numpy()
    tree = cKDTree(coords)
    violations = np.zeros(len(points), dtype=bool)
    for idx, coord in enumerate(coords):
        distances, _ = tree.query(coord, k=2)
        nearest = distances[1] if np.ndim(distances) > 0 else np.inf
        if nearest < min_spacing_px:
            violations[idx] = True
    return pd.Series(violations, index=points.index)


def curate_points(session: AnalysisSession, config: CurationConfig) -> AnalysisSession:
    source = session.refined_points if not session.refined_points.empty else session.candidate_points
    if source.empty:
        raise ValueError("No points are available for curation.")

    session.clear_downstream_results("curate")
    curated = source.copy()
    if "atom_id" not in curated.columns:
        curated.insert(0, "atom_id", np.arange(len(curated), dtype=int))

    image_shape = session.raw_image.shape if session.raw_image is not None else session.get_processed_image().shape
    curated["flag_duplicate"] = _flag_duplicates(curated, config.duplicate_radius_px)
    curated["flag_edge"] = _flag_edge_points(curated, image_shape, config.edge_margin)
    curated["flag_low_quality"] = curated.get("quality_score", pd.Series(np.nan, index=curated.index)).fillna(1.0) < config.min_quality_score
    curated["flag_poor_fit"] = curated.get("fit_residual", pd.Series(np.nan, index=curated.index)).fillna(0.0) > config.max_fit_residual
    curated["flag_spacing_violation"] = _flag_spacing_violations(curated, config.min_spacing_px)
    curated["keep"] = True

    if config.auto_drop_duplicates:
        curated.loc[curated["flag_duplicate"], "keep"] = False
    if config.auto_drop_edge_points:
        curated.loc[curated["flag_edge"], "keep"] = False
    if config.auto_drop_poor_fits:
        curated.loc[curated["flag_poor_fit"] | curated["flag_low_quality"], "keep"] = False

    session.curated_points = curated.reset_index(drop=True)
    session.set_stage("curated")
    session.record_step("curate_points", parameters=config, notes={"curated_count": len(curated)})
    return session


def _require_napari() -> Any:
    try:
        import napari
    except ImportError as exc:
        raise ImportError("napari is required for interactive curation.") from exc
    return napari


def _points_to_local(points: pd.DataFrame, origin_x: int, origin_y: int) -> np.ndarray:
    if points.empty:
        return np.empty((0, 2), dtype=float)
    return np.column_stack((points["y_px"] - origin_y, points["x_px"] - origin_x))


def _set_points_layer_add_defaults(layer: Any, *, point_size: float, face_color: str, border_color: str) -> None:
    """Keep newly added napari points visually consistent with existing points."""
    for attr, value in (
        ("current_size", float(point_size)),
        ("current_face_color", face_color),
        ("current_border_color", border_color),
        ("current_border_width", 0.0),
        ("current_symbol", "disc"),
    ):
        try:
            setattr(layer, attr, value)
        except Exception:
            continue


def _points_from_viewer(viewer_handle: Any, layer_name: str = "atom_points") -> tuple[pd.DataFrame, dict[str, int]]:
    origin = viewer_handle.layers[layer_name].metadata.get("origin", {"x": 0, "y": 0})
    data = np.asarray(viewer_handle.layers[layer_name].data, dtype=float)
    points = pd.DataFrame(
        {
            "x_px": data[:, 1] + origin["x"] if len(data) else np.array([], dtype=float),
            "y_px": data[:, 0] + origin["y"] if len(data) else np.array([], dtype=float),
        }
    )
    return points, {"x": int(origin["x"]), "y": int(origin["y"])}


def _show_viewer_blocking(viewer: Any) -> None:
    show = getattr(viewer, "show", None)
    if callable(show):
        try:
            show(block=True)
            return
        except TypeError:
            show()

    napari = _require_napari()
    runner = getattr(napari, "run", None)
    if not callable(runner):
        raise RuntimeError("The installed napari build does not expose a blocking viewer API.")
    runner()


def _resolve_editor_image(
    session: AnalysisSession,
    image_key: str = "processed",
    channel_name: str | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    if image_key == "processed":
        image = session.get_processed_image(channel_name)
        origin = session.get_processed_origin(channel_name)
    else:
        if channel_name is None:
            image = session.raw_image
        else:
            image = session.get_channel_state(channel_name).raw_image
        origin = (0, 0)
    if image is None:
        raise ValueError("No image is available for napari candidate editing.")
    return image, origin


def _broadcast_candidate_value(value: Any, length: int) -> list[Any]:
    if isinstance(value, (pd.Series, np.ndarray, list, tuple)):
        if len(value) != length:
            raise ValueError("Candidate metadata length does not match the edited point count.")
        return list(value)
    return [value] * length


def _build_candidate_points_table(
    points: pd.DataFrame,
    origin: dict[str, int] | tuple[int, int],
    *,
    candidate_id_offset: int = 0,
    column_role: str,
    seed_channel: str,
    confirm_channel: Any = pd.NA,
    parent_heavy_id: Any = pd.NA,
    contrast_mode_used: Any = "manual_edit",
) -> pd.DataFrame:
    origin_x = int(origin["x"]) if isinstance(origin, dict) else int(origin[0])
    origin_y = int(origin["y"]) if isinstance(origin, dict) else int(origin[1])
    point_count = int(len(points))
    table = pd.DataFrame(
        {
            "candidate_id": np.arange(point_count, dtype=int) + int(candidate_id_offset),
            "x_px": points["x_px"] if not points.empty else np.array([], dtype=float),
            "y_px": points["y_px"] if not points.empty else np.array([], dtype=float),
            "x_local_px": points["x_px"] - origin_x if not points.empty else np.array([], dtype=float),
            "y_local_px": points["y_px"] - origin_y if not points.empty else np.array([], dtype=float),
            "center_intensity": np.full(point_count, np.nan, dtype=float),
            "local_background": np.full(point_count, np.nan, dtype=float),
            "prominence": np.full(point_count, np.nan, dtype=float),
            "local_snr": np.full(point_count, np.nan, dtype=float),
            "score": np.full(point_count, np.nan, dtype=float),
            "contrast_mode_used": _broadcast_candidate_value(contrast_mode_used, point_count),
            "column_role": _broadcast_candidate_value(column_role, point_count),
            "seed_channel": _broadcast_candidate_value(seed_channel, point_count),
            "detected_from_channels": [[seed_channel] for _ in range(point_count)],
            "confirm_channel": _broadcast_candidate_value(confirm_channel, point_count),
            "parent_heavy_id": _broadcast_candidate_value(parent_heavy_id, point_count),
            "support_score": np.full(point_count, np.nan, dtype=float),
            "confirm_score": np.full(point_count, np.nan, dtype=float),
            "class_id": [pd.NA] * point_count,
            "class_name": [pd.NA] * point_count,
            "class_color": [pd.NA] * point_count,
            "class_confidence": np.full(point_count, np.nan, dtype=float),
            "class_source": [pd.NA] * point_count,
            "class_reviewed": np.full(point_count, False, dtype=bool),
        }
    )
    return table[_CANDIDATE_TABLE_COLUMNS]


def _ensure_candidate_table_schema(points: pd.DataFrame) -> pd.DataFrame:
    normalized = points.copy().reset_index(drop=True)
    for column in _CANDIDATE_TABLE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    if normalized.empty:
        return normalized[_CANDIDATE_TABLE_COLUMNS]
    normalized = normalized[_CANDIDATE_TABLE_COLUMNS]
    normalized["candidate_id"] = np.arange(len(normalized), dtype=int)
    return normalized


def _filter_candidates_by_role(points: pd.DataFrame, role: str) -> pd.DataFrame:
    if points.empty or "column_role" not in points.columns:
        return pd.DataFrame(columns=points.columns)
    return points.loc[points["column_role"] == role].copy()


def _nearest_parent_heavy_ids(light_points: pd.DataFrame, heavy_points: pd.DataFrame) -> list[Any]:
    if light_points.empty:
        return []
    if heavy_points.empty:
        return [pd.NA] * len(light_points)
    heavy_coords = heavy_points[["x_px", "y_px"]].to_numpy(dtype=float)
    heavy_ids = heavy_points["candidate_id"].to_numpy(dtype=int)
    light_coords = light_points[["x_px", "y_px"]].to_numpy(dtype=float)
    _, indices = cKDTree(heavy_coords).query(light_coords, k=1)
    return [int(heavy_ids[int(index)]) for index in np.atleast_1d(indices)]


def launch_napari_candidate_editor(
    session: AnalysisSession,
    image_key: str = "processed",
    *,
    channel_name: str | None = None,
    editable_points: pd.DataFrame | None = None,
    reference_points: pd.DataFrame | None = None,
    title: str | None = None,
    point_size: float = 5.0,
) -> Any:
    napari = _require_napari()

    image, (origin_x, origin_y) = _resolve_editor_image(session, image_key=image_key, channel_name=channel_name)
    viewer = napari.Viewer(title=title or f"EM Atom Workbench - Candidate Review - {session.name}")
    viewer.add_image(image, name="image")
    if reference_points is not None and not reference_points.empty:
        reference_layer = viewer.add_points(
            _points_to_local(reference_points, origin_x, origin_y),
            name="reference_points",
            size=point_size,
            canvas_size_limits=(4, 8),
            face_color="#ff8c00",
            border_color="#ffb347",
            border_width=0.0,
            border_width_is_relative=False,
            opacity=0.85,
            symbol="disc",
        )
        reference_layer.editable = False
        _set_points_layer_add_defaults(
            reference_layer,
            point_size=point_size,
            face_color="#ff8c00",
            border_color="#ffb347",
        )
    points_layer = viewer.add_points(
        _points_to_local(session.candidate_points if editable_points is None else editable_points, origin_x, origin_y),
        name="atom_points",
        size=point_size,
        canvas_size_limits=(4, 8),
        face_color="#00d7ff",
        border_color="cyan",
        border_width=0.0,
        border_width_is_relative=False,
        opacity=0.95,
        symbol="disc",
    )
    points_layer.editable = True
    _set_points_layer_add_defaults(
        points_layer,
        point_size=point_size,
        face_color="#00d7ff",
        border_color="cyan",
    )
    points_layer.metadata["origin"] = {"x": origin_x, "y": origin_y}
    points_layer.metadata["session_name"] = session.name
    points_layer.metadata["point_target"] = "candidate_points"
    points_layer.metadata["channel_name"] = channel_name or session.primary_channel
    points_layer.metadata["image_key"] = image_key
    points_layer.metadata["point_size"] = float(point_size)
    return viewer


def apply_candidate_edits_from_viewer(session: AnalysisSession, viewer_handle: Any) -> AnalysisSession:
    points, origin = _points_from_viewer(viewer_handle)
    previous_count = int(len(session.candidate_points))
    layer_metadata = dict(getattr(viewer_handle.layers["atom_points"], "metadata", {}))
    seed_channel = str(layer_metadata.get("channel_name") or session.primary_channel)
    image_key = str(layer_metadata.get("image_key") or "processed")
    session.clear_downstream_results("detect")
    candidate_points = _build_candidate_points_table(
        points,
        origin,
        column_role="atom_column",
        seed_channel=seed_channel,
        confirm_channel=pd.NA,
        parent_heavy_id=pd.NA,
        contrast_mode_used="manual_edit",
    )

    session.candidate_points = candidate_points
    session.set_stage("candidate_reviewed")
    session.record_step(
        "apply_candidate_edits_from_viewer",
        parameters={
            "image_key": image_key,
            "channel_name": seed_channel,
            "point_size": layer_metadata.get("point_size"),
        },
        notes={
            "previous_candidate_count": previous_count,
            "candidate_count": len(candidate_points),
        },
    )
    return session


def edit_candidates_with_napari(
    session: AnalysisSession,
    image_key: str = "processed",
    *,
    channel_name: str | None = None,
    point_size: float = 5.0,
) -> AnalysisSession:
    viewer = launch_napari_candidate_editor(
        session,
        image_key=image_key,
        channel_name=channel_name,
        point_size=point_size,
    )
    _show_viewer_blocking(viewer)
    return apply_candidate_edits_from_viewer(session, viewer)


def apply_hfo2_heavy_candidate_edits_from_viewer(
    session: AnalysisSession,
    viewer_handle: Any,
    *,
    heavy_channel: str,
) -> AnalysisSession:
    points, origin = _points_from_viewer(viewer_handle)
    session.clear_downstream_results("detect")
    session.candidate_points = _build_candidate_points_table(
        points,
        origin,
        column_role="heavy_atom",
        seed_channel=heavy_channel,
        confirm_channel=pd.NA,
        parent_heavy_id=pd.NA,
        contrast_mode_used="manual_edit",
    )
    session.set_stage("heavy_reviewed")
    session.record_step(
        "apply_hfo2_heavy_candidate_edits_from_viewer",
        parameters={"channel_name": heavy_channel, "image_key": "processed"},
        notes={"heavy_count": len(session.candidate_points)},
    )
    return session


def edit_hfo2_heavy_candidates_with_napari(
    session: AnalysisSession,
    heavy_channel: str,
    image_key: str = "processed",
) -> AnalysisSession:
    heavy_points = _filter_candidates_by_role(session.candidate_points, "heavy_atom")
    if heavy_points.empty:
        heavy_points = session.candidate_points.copy()
    viewer = launch_napari_candidate_editor(
        session,
        image_key=image_key,
        channel_name=heavy_channel,
        editable_points=heavy_points,
        title=f"EM Atom Workbench - Heavy Candidate Review - {session.name}",
    )
    _show_viewer_blocking(viewer)
    return apply_hfo2_heavy_candidate_edits_from_viewer(session, viewer, heavy_channel=heavy_channel)


def apply_hfo2_light_candidate_edits_from_viewer(
    session: AnalysisSession,
    viewer_handle: Any,
    *,
    heavy_points: pd.DataFrame,
    light_channel: str,
) -> AnalysisSession:
    points, origin = _points_from_viewer(viewer_handle)
    heavy_candidates = _ensure_candidate_table_schema(heavy_points)
    parent_heavy_ids = _nearest_parent_heavy_ids(points, heavy_candidates)
    light_candidates = _build_candidate_points_table(
        points,
        origin,
        candidate_id_offset=len(heavy_candidates),
        column_role="light_atom",
        seed_channel=light_channel,
        confirm_channel=pd.NA,
        parent_heavy_id=parent_heavy_ids,
        contrast_mode_used="manual_edit",
    )
    session.clear_downstream_results("detect")
    session.candidate_points = pd.concat([heavy_candidates, light_candidates], ignore_index=True)
    session.set_stage("detected")
    session.record_step(
        "apply_hfo2_light_candidate_edits_from_viewer",
        parameters={"channel_name": light_channel, "image_key": "processed"},
        notes={
            "heavy_count": int((session.candidate_points["column_role"] == "heavy_atom").sum()),
            "light_count": int((session.candidate_points["column_role"] == "light_atom").sum()),
        },
    )
    return session


def edit_hfo2_light_candidates_with_napari(
    session: AnalysisSession,
    heavy_channel: str,
    light_channel: str,
    image_key: str = "processed",
) -> AnalysisSession:
    heavy_points = _filter_candidates_by_role(session.candidate_points, "heavy_atom")
    if heavy_points.empty:
        raise ValueError("Heavy candidates are required before opening the HfO2 light-candidate editor.")
    light_points = _filter_candidates_by_role(session.candidate_points, "light_atom")
    viewer = launch_napari_candidate_editor(
        session,
        image_key=image_key,
        channel_name=light_channel,
        editable_points=light_points,
        reference_points=heavy_points,
        title=f"EM Atom Workbench - Light Candidate Review - {session.name}",
    )
    _show_viewer_blocking(viewer)
    return apply_hfo2_light_candidate_edits_from_viewer(
        session,
        viewer,
        heavy_points=heavy_points,
        light_channel=light_channel,
    )


def launch_napari_curation(session: AnalysisSession, image_key: str = "processed") -> Any:
    napari = _require_napari()

    image = session.get_processed_image() if image_key == "processed" else session.raw_image
    if image is None:
        raise ValueError("No image is available for napari curation.")

    origin_x, origin_y = session.get_processed_origin() if image_key == "processed" else (0, 0)
    points = session.get_atom_table(preferred="curated")

    viewer = napari.Viewer(title=f"EM Atom Workbench - {session.name}")
    viewer.add_image(image, name="image")
    points_layer = viewer.add_points(
        _points_to_local(points, origin_x, origin_y),
        name="atom_points",
        size=5,
        canvas_size_limits=(4, 8),
        face_color="#00d7ff",
        border_color="cyan",
        border_width=0.0,
        border_width_is_relative=False,
        opacity=0.95,
        symbol="disc",
    )
    points_layer.editable = True
    _set_points_layer_add_defaults(
        points_layer,
        point_size=5.0,
        face_color="#00d7ff",
        border_color="cyan",
    )
    points_layer.metadata["origin"] = {"x": origin_x, "y": origin_y}
    points_layer.metadata["session_name"] = session.name
    return viewer


def apply_curation_from_viewer(
    session: AnalysisSession,
    viewer_handle: Any,
    rerefine: bool = True,
    refinement_config: RefinementConfig | None = None,
    curation_config: CurationConfig | None = None,
) -> AnalysisSession:
    points, _ = _points_from_viewer(viewer_handle)
    points = pd.DataFrame(
        {
            "atom_id": np.arange(len(points), dtype=int),
            "x_px": points["x_px"] if not points.empty else np.array([], dtype=float),
            "y_px": points["y_px"] if not points.empty else np.array([], dtype=float),
        }
    )

    session.clear_downstream_results("curate")
    if rerefine and not points.empty:
        refined = refine_point_table(session, points, refinement_config or RefinementConfig())
        session.curated_points = refined
    else:
        session.curated_points = points

    if curation_config is not None:
        curate_points(session, curation_config)
    else:
        session.set_stage("curated")

    session.record_step(
        "apply_curation_from_viewer",
        parameters={"rerefine": rerefine},
        notes={"curated_count": len(session.curated_points)},
    )
    return session


def curate_with_napari(
    session: AnalysisSession,
    refinement_config: RefinementConfig | None = None,
    curation_config: CurationConfig | None = None,
    image_key: str = "processed",
    rerefine: bool = True,
) -> AnalysisSession:
    viewer = launch_napari_curation(session, image_key=image_key)
    _show_viewer_blocking(viewer)
    return apply_curation_from_viewer(
        session,
        viewer,
        rerefine=rerefine,
        refinement_config=refinement_config,
        curation_config=curation_config,
    )
