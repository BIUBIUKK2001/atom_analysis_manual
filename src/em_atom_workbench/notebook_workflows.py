"""Small notebook orchestration helpers.

These functions keep the notebooks focused on parameters and stage calls while
leaving repetitive validation, display, plotting, and active-session updates in
one place.
"""

from __future__ import annotations

import math
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .classification import (
    AtomColumnClassificationConfig,
    apply_class_name_map,
    apply_class_review_from_viewer,
    classification_summary_table,
    classify_atom_columns,
    launch_class_review_napari,
)
from .curate import (
    curate_points,
    edit_candidates_with_napari,
    edit_hfo2_heavy_candidates_with_napari,
    edit_hfo2_light_candidates_with_napari,
)
from .detect import detect_hfo2_heavy_candidates, detect_hfo2_light_candidates, detect_multichannel_candidates
from .io import load_image_bundle
from .plotting import (
    launch_detection_napari_viewer,
    launch_refinement_napari_viewer,
    plot_class_feature_scatter_matrix,
    plot_class_overlay,
    plot_atom_overlay,
    plot_histogram_or_distribution,
    plot_raw_image,
)
from .refine import refine_points, refine_points_by_class
from .session import (
    AnalysisSession,
    CurationConfig,
    DetectionConfig,
    HfO2MultichannelDetectionConfig,
    PixelCalibration,
    RefinementConfig,
)
from .simple_quant import (
    AnalysisROI,
    BasisVectorSpec,
    DirectionalSpacingTask,
    LineGroupingTask,
    LineGuideTask,
    NearestForwardTask,
    PairDistanceTask,
    PairSegmentTask,
    PeriodicVectorTask,
    assign_lines_by_projection,
    combine_analysis_points,
    compute_directional_spacing,
    compute_line_spacing,
    compute_pair_distances,
    make_pair_center_points,
    prepare_analysis_points,
    prepare_quant_points,
    resolve_basis_vector_specs,
    run_measurement_tasks,
    summarize_simple_quant_table,
)
from .utils import (
    load_or_connect_session,
    save_active_session,
    save_checkpoint,
    stage_rank,
    synthetic_gaussian_image,
    synthetic_hfo2_multichannel_bundle,
    write_json,
)


@dataclass(frozen=True)
class HfO2Channels:
    primary: str
    heavy: str
    light: str
    confirm: str | None = None


@dataclass
class NotebookResult:
    session: AnalysisSession | None = None
    active_path: Path | None = None
    tables: list[pd.DataFrame] = field(default_factory=list)
    figures: list[Any] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


def display_notebook_result(result: NotebookResult) -> None:
    """Display tables and figures from a notebook stage result."""
    try:
        from IPython.display import display
    except Exception:  # pragma: no cover - notebooks normally have IPython.
        display = print

    for table in result.tables:
        display(table)
    for figure in result.figures:
        display(figure)
        plt.close(figure)
    for message in result.messages:
        print(message)


def filter_points_by_role(points: pd.DataFrame, role: str) -> pd.DataFrame:
    if points is None or len(points) == 0:
        return pd.DataFrame()
    if "column_role" not in points.columns:
        return points.copy() if role == "light_atom" else points.iloc[0:0].copy()
    return points.loc[points["column_role"] == role].copy()


def workflow_channels(session: AnalysisSession) -> HfO2Channels:
    settings = dict(session.workflow_settings or {})
    primary = str(settings.get("primary_channel") or session.primary_channel)
    heavy = settings.get("heavy_channel")
    light = settings.get("light_channel")
    confirm = settings.get("confirm_channel")

    if primary != "idpc":
        raise ValueError("hfo2_multichannel requires the primary channel to be 'idpc'.")
    missing = [
        name
        for name in (heavy, light)
        if not isinstance(name, str) or name not in session.list_channels()
    ]
    if missing:
        raise ValueError(f"Missing required HfO2 channels: {missing}")
    if confirm is not None and confirm not in session.list_channels():
        raise ValueError(f"Unknown confirm channel: {confirm!r}")
    return HfO2Channels(primary=primary, heavy=str(heavy), light=str(light), confirm=confirm)


def hfo2_stage_summary(session: AnalysisSession, active_path: Path | None) -> pd.DataFrame:
    heavy = filter_points_by_role(session.candidate_points, "heavy_atom")
    light = filter_points_by_role(session.candidate_points, "light_atom")
    rows = [
        {"field": "session_name", "value": session.name},
        {"field": "input_path", "value": session.input_path or "synthetic_multichannel_demo"},
        {"field": "dataset_index", "value": session.dataset_index},
        {"field": "pixel_size", "value": session.pixel_calibration.size},
        {"field": "unit", "value": session.pixel_calibration.unit},
        {"field": "calibration_source", "value": session.pixel_calibration.source},
        {"field": "workflow_mode", "value": session.workflow_mode},
        {"field": "current_stage", "value": session.current_stage},
        {"field": "heavy_count", "value": len(heavy)},
        {"field": "light_count", "value": len(light)},
        {"field": "candidate_count", "value": len(session.candidate_points)},
        {"field": "refined_count", "value": len(session.refined_points)},
        {"field": "curated_count", "value": len(session.curated_points)},
        {"field": "active_session", "value": "" if active_path is None else str(active_path)},
    ]
    return pd.DataFrame(rows)


def channel_summary(session: AnalysisSession) -> pd.DataFrame:
    rows = []
    for channel_name in session.list_channels():
        state = session.get_channel_state(channel_name)
        rows.append(
            {
                "channel": channel_name,
                "is_primary": channel_name == session.primary_channel,
                "input_path": state.input_path,
                "dataset_index": state.dataset_index,
                "contrast_mode": state.contrast_mode,
                "raw_shape": None if state.raw_image is None else tuple(state.raw_image.shape),
            }
        )
    return pd.DataFrame(rows)


def hfo2_channel_summary(session: AnalysisSession) -> pd.DataFrame:
    return channel_summary(session)


def _plot_calibration_kwargs(session: AnalysisSession) -> dict[str, Any]:
    calibration = session.pixel_calibration
    return {"pixel_size": calibration.size, "unit": calibration.unit, "target_unit": "nm"}


def metadata_preview(session: AnalysisSession, max_rows: int = 12) -> pd.DataFrame:
    if not session.raw_metadata:
        return pd.DataFrame(columns=["key", "value"])
    keys = list(session.raw_metadata.keys())
    return pd.DataFrame(
        {
            "key": keys[:max_rows],
            "value": [str(session.raw_metadata[key])[:160] for key in keys[:max_rows]],
        }
    )


def _simple_quant_stage_summary(
    session: AnalysisSession,
    quant_points: pd.DataFrame,
    *,
    source_table: str,
    output_dir: Path,
) -> pd.DataFrame:
    rows = [
        {"field": "session_name", "value": session.name},
        {"field": "current_stage", "value": session.current_stage},
        {"field": "primary_channel", "value": session.primary_channel},
        {"field": "source_table", "value": source_table},
        {"field": "quant_point_count", "value": len(quant_points)},
        {"field": "class_count", "value": int(quant_points["class_id"].nunique(dropna=True)) if "class_id" in quant_points else 0},
        {"field": "pixel_size", "value": session.pixel_calibration.size},
        {"field": "unit", "value": session.pixel_calibration.unit},
        {"field": "output_dir", "value": str(output_dir)},
    ]
    return pd.DataFrame(rows)


def _simple_quant_output_dirs(result_root: str | Path) -> dict[str, Path]:
    output_dir = Path(result_root) / "02_simple_quant"
    dirs = {
        "output": output_dir,
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
        "configs": output_dir / "configs",
        "session": output_dir / "session",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _resolve_simple_quant_image(
    session: AnalysisSession,
    *,
    image_channel: str | None,
    image_key: str,
) -> tuple[np.ndarray, str, str]:
    channel_name = image_channel or session.primary_channel
    key = str(image_key).lower()
    if key == "raw":
        image = session.get_channel_state(channel_name).raw_image
        if image is None:
            raise ValueError(f"Raw image is not available for channel {channel_name!r}.")
        return image, channel_name, key
    if key == "processed":
        return session.get_processed_image(channel_name), channel_name, key
    raise ValueError("image_key must be 'raw' or 'processed'.")


def initialize_simple_quant_analysis(
    *,
    session_path: str | Path | None,
    result_root: str | Path,
    source_table: str,
    use_keep_only: bool,
    class_filter: tuple[str, ...] | list[str] | set[str] | None,
    class_id_filter: tuple[int, ...] | list[int] | set[int] | None,
    roi: tuple[float, float, float, float] | None,
    image_channel: str | None,
    image_key: str,
) -> dict[str, Any]:
    session = load_or_connect_session(result_root, session_path=session_path)
    quant_points = prepare_quant_points(
        session,
        source_table=source_table,
        use_keep_only=use_keep_only,
        class_filter=class_filter,
        class_id_filter=class_id_filter,
        roi=roi,
    )
    image, resolved_channel, resolved_image_key = _resolve_simple_quant_image(
        session,
        image_channel=image_channel,
        image_key=image_key,
    )
    output_dirs = _simple_quant_output_dirs(result_root)
    summary = _simple_quant_stage_summary(
        session,
        quant_points,
        source_table=source_table,
        output_dir=output_dirs["output"],
    )
    return {
        "session": session,
        "quant_points": quant_points,
        "image": image,
        "image_channel": resolved_channel,
        "image_key": resolved_image_key,
        "output_dirs": output_dirs,
        "summary_tables": {
            "simple_quant_summary": summary,
            "quant_points_preview": quant_points.head(),
        },
    }


def run_directional_spacing_analysis(
    quant_points: pd.DataFrame,
    direction_table: pd.DataFrame,
    tasks: tuple[DirectionalSpacingTask, ...] | list[DirectionalSpacingTask],
) -> dict[str, pd.DataFrame]:
    table = compute_directional_spacing(quant_points, direction_table, tasks)
    value_column = "distance_pm" if not table.empty and table["distance_pm"].notna().any() else "distance_px"
    summary = summarize_simple_quant_table(table, ["measurement_name", "direction_name"], value_column)
    return {"directional_spacing_table": table, "directional_spacing_summary": summary}


def run_pair_distance_analysis(
    quant_points: pd.DataFrame,
    direction_table: pd.DataFrame,
    tasks: tuple[PairDistanceTask, ...] | list[PairDistanceTask],
) -> dict[str, pd.DataFrame]:
    table = compute_pair_distances(quant_points, tasks, direction_table=direction_table)
    value_column = "distance_pm" if not table.empty and table["distance_pm"].notna().any() else "distance_px"
    summary = summarize_simple_quant_table(table, ["pair_name"], value_column)
    return {"pair_distance_table": table, "pair_distance_summary": summary}


def run_line_spacing_analysis(
    quant_points: pd.DataFrame,
    direction_table: pd.DataFrame,
    tasks: tuple[LineGroupingTask, ...] | list[LineGroupingTask],
) -> dict[str, pd.DataFrame]:
    assignment_tables: list[pd.DataFrame] = []
    spacing_tables: list[pd.DataFrame] = []
    for task in tasks:
        assignments = assign_lines_by_projection(quant_points, direction_table, task)
        assignment_tables.append(assignments)
        spacing_tables.append(compute_line_spacing(quant_points, assignments, direction_table, task))
    line_assignments = pd.concat(assignment_tables, ignore_index=True) if assignment_tables else pd.DataFrame()
    line_spacing = pd.concat(spacing_tables, ignore_index=True) if spacing_tables else pd.DataFrame()
    if line_spacing.empty:
        line_summary = pd.DataFrame()
    else:
        line_summary = (
            line_spacing.groupby(["line_task_name", "direction_name", "group_axis", "line_id"], dropna=False)
            .agg(
                line_atom_count=("atom_id", "count"),
                line_center_px=("line_center_px", "first"),
                line_width_px=("line_width_px", "first"),
                line_width_nm=("line_width_nm", "first"),
                line_width_pm=("line_width_pm", "first"),
                line_mean_spacing_px=("line_mean_spacing_px", "first"),
                line_mean_spacing_pm=("line_mean_spacing_pm", "first"),
                line_std_spacing_px=("line_std_spacing_px", "first"),
                line_std_spacing_pm=("line_std_spacing_pm", "first"),
            )
            .reset_index()
        )
    return {
        "line_assignments": line_assignments,
        "line_spacing_table": line_spacing,
        "line_summary": line_summary,
    }


def _save_simple_quant_figures(figures: dict[str, Any], figures_dir: Path) -> dict[str, list[str]]:
    saved: dict[str, list[str]] = {}
    for stem, figure in figures.items():
        if figure is None:
            continue
        paths: list[str] = []
        for suffix in ("pdf", "png", "svg"):
            target = figures_dir / f"{stem}.{suffix}"
            figure.savefig(target, bbox_inches="tight", dpi=300)
            paths.append(str(target))
        saved[str(stem)] = paths
    return saved


def export_simple_quant_analysis(
    *,
    session: AnalysisSession,
    output_dirs: dict[str, Path] | None = None,
    result_root: str | Path | None = None,
    tables: dict[str, pd.DataFrame],
    figures: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    direction_table: pd.DataFrame | None = None,
) -> dict[str, Any]:
    dirs = output_dirs or _simple_quant_output_dirs(result_root or "results")
    table_paths: dict[str, str] = {}
    for name, table in tables.items():
        if table is None:
            continue
        target = dirs["tables"] / f"{name}.csv"
        table.to_csv(target, index=False)
        table_paths[str(name)] = str(target)

    figure_paths = _save_simple_quant_figures(figures or {}, dirs["figures"])
    config_path = write_json(dirs["configs"] / "simple_quant_config.json", dict(config or {}))
    direction_path = write_json(
        dirs["configs"] / "directions.json",
        {
            "direction_table": []
            if direction_table is None
            else direction_table.to_dict(orient="records"),
        },
    )
    manifest = {
        "workflow": "simple_quant",
        "output_dir": str(dirs["output"]),
        "tables": table_paths,
        "figures": figure_paths,
        "configs": {
            "simple_quant_config": str(config_path),
            "directions": str(direction_path),
        },
    }
    manifest_path = write_json(dirs["output"] / "manifest.json", manifest)

    session.annotations["simple_quant"] = {
        "output_dir": str(dirs["output"]),
        "source_table": (config or {}).get("source_table"),
        "use_keep_only": (config or {}).get("use_keep_only"),
        "class_filter": (config or {}).get("class_filter"),
        "class_id_filter": (config or {}).get("class_id_filter"),
        "roi": (config or {}).get("roi"),
        "tables": table_paths,
    }
    session.set_stage("simple_quant")
    checkpoint_path = session.save_pickle(dirs["session"] / "02_simple_quant_session.pkl")
    manifest["manifest"] = str(manifest_path)
    manifest["session_checkpoint"] = str(checkpoint_path)
    return manifest


def _roi_table_from_points(points: pd.DataFrame) -> pd.DataFrame:
    if points is None or points.empty or "roi_id" not in points.columns:
        return pd.DataFrame(columns=["roi_id", "roi_name", "roi_color", "point_count"])
    columns = [column for column in ("roi_id", "roi_name", "roi_color") if column in points.columns]
    table = points.groupby(columns, dropna=False).size().reset_index(name="point_count")
    return table


def initialize_simple_quant_v2_analysis(
    *,
    session_path: str | Path | None,
    result_root: str | Path,
    source_table: str,
    use_keep_only: bool,
    class_filter: tuple[str, ...] | list[str] | set[str] | None,
    class_id_filter: tuple[int, ...] | list[int] | set[int] | None,
    rois: list[AnalysisROI] | tuple[AnalysisROI, ...] | None,
    image_channel: str | None,
    image_key: str,
) -> dict[str, Any]:
    session = load_or_connect_session(result_root, session_path=session_path)
    analysis_points = prepare_analysis_points(
        session,
        source_table=source_table,
        use_keep_only=use_keep_only,
        class_filter=class_filter,
        class_id_filter=class_id_filter,
        rois=rois,
    )
    image, resolved_channel, resolved_image_key = _resolve_simple_quant_image(
        session,
        image_channel=image_channel,
        image_key=image_key,
    )
    output_dirs = _simple_quant_output_dirs(result_root)
    roi_table = _roi_table_from_points(analysis_points)
    summary = pd.DataFrame(
        [
            {"field": "session_name", "value": session.name},
            {"field": "current_stage", "value": session.current_stage},
            {"field": "primary_channel", "value": session.primary_channel},
            {"field": "source_table", "value": source_table},
            {"field": "analysis_point_rows", "value": len(analysis_points)},
            {"field": "unique_points", "value": analysis_points["point_id"].nunique() if "point_id" in analysis_points else len(analysis_points)},
            {"field": "roi_count", "value": roi_table["roi_id"].nunique() if "roi_id" in roi_table else 0},
            {"field": "output_dir", "value": str(output_dirs["output"])},
        ]
    )
    return {
        "session": session,
        "analysis_points": analysis_points,
        "roi_table": roi_table,
        "image": image,
        "image_channel": resolved_channel,
        "image_key": resolved_image_key,
        "output_dirs": output_dirs,
        "summary_tables": {
            "simple_quant_v2_summary": summary,
            "roi_table": roi_table,
            "analysis_points_preview": analysis_points.head(),
        },
    }


def run_simple_quant_measurements(
    analysis_points: pd.DataFrame,
    basis_vector_table: pd.DataFrame,
    tasks: tuple[Any, ...] | list[Any],
) -> dict[str, pd.DataFrame]:
    result = run_measurement_tasks(analysis_points, basis_vector_table, tasks)
    measurement_segments = result["measurement_segments"]
    value_column = "distance_pm" if not measurement_segments.empty and measurement_segments["distance_pm"].notna().any() else "distance_px"
    result["summaries"] = summarize_simple_quant_table(
        measurement_segments,
        ["task_name", "task_type", "roi_id"],
        value_column,
    )
    return result


def export_simple_quant_v2_analysis(
    *,
    session: AnalysisSession,
    output_dirs: dict[str, Path] | None = None,
    result_root: str | Path | None = None,
    analysis_points: pd.DataFrame | None = None,
    roi_table: pd.DataFrame | None = None,
    basis_vector_table: pd.DataFrame | None = None,
    roi_basis_table: pd.DataFrame | None = None,
    measurement_segments: pd.DataFrame | None = None,
    pair_center_points: pd.DataFrame | None = None,
    line_guides: pd.DataFrame | None = None,
    summaries: pd.DataFrame | None = None,
    figures: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dirs = output_dirs or _simple_quant_output_dirs(result_root or "results")
    tables = {
        "analysis_points": analysis_points,
        "roi_table": roi_table,
        "basis_vector_table": basis_vector_table,
        "roi_basis_table": roi_basis_table,
        "measurement_segments": measurement_segments,
        "pair_center_points": pair_center_points,
        "line_guides": line_guides,
        "summaries": summaries,
    }
    table_paths: dict[str, str] = {}
    for name, table in tables.items():
        if table is None:
            continue
        target = dirs["tables"] / f"{name}.csv"
        table.to_csv(target, index=False)
        table_paths[name] = str(target)
    figure_paths = _save_simple_quant_figures(figures or {}, dirs["figures"])
    config_path = write_json(dirs["configs"] / "simple_quant_v2_config.json", dict(config or {}))
    manifest = {
        "workflow": "simple_quant_v2",
        "output_dir": str(dirs["output"]),
        "tables": table_paths,
        "figures": figure_paths,
        "configs": {"simple_quant_v2_config": str(config_path)},
    }
    manifest_path = write_json(dirs["output"] / "manifest.json", manifest)
    session.annotations["simple_quant_v2"] = {
        "output_dir": str(dirs["output"]),
        "tables": table_paths,
        "config": str(config_path),
    }
    session.set_stage("simple_quant_v2")
    checkpoint_path = session.save_pickle(dirs["session"] / "02_simple_quant_session.pkl")
    manifest["manifest"] = str(manifest_path)
    manifest["session_checkpoint"] = str(checkpoint_path)
    return manifest


def generic_stage_summary(session: AnalysisSession, active_path: Path | None) -> pd.DataFrame:
    table = session.get_atom_table(preferred="curated")
    class_count = int(table["class_id"].nunique(dropna=True)) if "class_id" in table.columns else 0
    rows = [
        {"field": "session_name", "value": session.name},
        {"field": "input_path", "value": session.input_path or "synthetic_generic_demo"},
        {"field": "dataset_index", "value": session.dataset_index},
        {"field": "pixel_size", "value": session.pixel_calibration.size},
        {"field": "unit", "value": session.pixel_calibration.unit},
        {"field": "workflow_mode", "value": session.workflow_mode},
        {"field": "current_stage", "value": session.current_stage},
        {"field": "channel_count", "value": len(session.list_channels())},
        {"field": "candidate_count", "value": len(session.candidate_points)},
        {"field": "refined_count", "value": len(session.refined_points)},
        {"field": "classified_count", "value": len(table) if "class_id" in table.columns else 0},
        {"field": "class_count", "value": class_count},
        {"field": "curated_count", "value": len(session.curated_points)},
        {"field": "active_session", "value": "" if active_path is None else str(active_path)},
    ]
    return pd.DataFrame(rows)


def build_synthetic_generic_class_session(*, rng_seed: int = 11) -> AnalysisSession:
    rng = np.random.default_rng(rng_seed)
    shape = (128, 128)
    spacing = 16.0
    peaks_a: list[dict[str, float]] = []
    peaks_b: list[dict[str, float]] = []
    coords: list[dict[str, float | int]] = []
    class_index = 0
    for row_idx, y in enumerate(np.arange(spacing, shape[0] - spacing / 2, spacing)):
        for col_idx, x in enumerate(np.arange(spacing, shape[1] - spacing / 2, spacing)):
            class_index = int((row_idx + col_idx) % 3)
            x_j = float(x + rng.normal(0.0, 0.2))
            y_j = float(y + rng.normal(0.0, 0.2))
            amp0 = (0.65, 1.05, 1.45)[class_index]
            amp1 = (1.35, 0.95, 0.55)[class_index]
            sigma = (1.0, 1.25, 1.55)[class_index]
            peaks_a.append({"x": x_j, "y": y_j, "amplitude": amp0, "sigma_x": sigma, "sigma_y": sigma})
            peaks_b.append({"x": x_j, "y": y_j, "amplitude": amp1, "sigma_x": sigma, "sigma_y": sigma})
            coords.append({"x_px": x_j, "y_px": y_j, "truth_class_id": class_index})
    image_a = synthetic_gaussian_image(shape, peaks_a, background=0.06, noise_sigma=0.012, rng_seed=rng_seed)
    image_b = synthetic_gaussian_image(shape, peaks_b, background=0.05, noise_sigma=0.012, rng_seed=rng_seed + 1)
    session = AnalysisSession(
        name="synthetic_generic_atom_columns",
        raw_image=image_a,
        raw_metadata={"source": "synthetic_generic_atom_columns", "truth": pd.DataFrame(coords).to_dict("records")},
        pixel_calibration=PixelCalibration(size=0.2, unit="A", source="synthetic_demo"),
        contrast_mode="bright_peak",
        primary_channel="channel_0",
    )
    session.set_channel_state(
        "channel_0",
        input_path="synthetic://channel_0",
        raw_image=image_a,
        raw_metadata={"source": "synthetic_generic_atom_columns", "channel": "channel_0"},
        contrast_mode="bright_peak",
    )
    session.set_channel_state(
        "channel_1",
        input_path="synthetic://channel_1",
        raw_image=image_b,
        raw_metadata={"source": "synthetic_generic_atom_columns", "channel": "channel_1"},
        contrast_mode="bright_peak",
    )
    session.set_primary_channel("channel_0")
    session.set_stage("loaded")
    session.record_step(
        "load_synthetic_generic_atom_columns",
        notes={"truth_class_count": 3, "truth_count": len(coords)},
    )
    return session


def initialize_generic_classification_session(
    *,
    result_root: str | Path,
    channels: dict[str, str | Path | None],
    primary_channel: str,
    channel_contrast_modes: dict[str, str] | None = None,
    channel_dataset_indices: dict[str, int | None] | None = None,
    manual_calibration: PixelCalibration | dict[str, Any] | float | None = None,
    synthetic_rng_seed: int = 11,
) -> NotebookResult:
    result_root = Path(result_root)
    path_map = {str(name): path for name, path in dict(channels or {}).items() if path}
    if not path_map:
        session = build_synthetic_generic_class_session(rng_seed=synthetic_rng_seed)
        primary_channel = session.primary_channel
    else:
        if primary_channel not in path_map:
            raise KeyError(f"PRIMARY_CHANNEL {primary_channel!r} was not found in CHANNELS.")
        missing = [str(path) for path in path_map.values() if not Path(path).exists()]
        if missing:
            raise FileNotFoundError("Missing channel input files:\n" + "\n".join(missing))
        session = load_image_bundle(
            path_map,
            primary_channel=primary_channel,
            manual_calibration=manual_calibration,
            contrast_modes=dict(channel_contrast_modes or {}),
            dataset_indices=dict(channel_dataset_indices or {}),
        )
    session.set_workflow(
        "atom_column_classification",
        {
            "primary_channel": primary_channel,
            "channel_names": session.list_channels(),
            "channel_contrast_modes": {
                name: session.get_channel_state(name).contrast_mode for name in session.list_channels()
            },
        },
    )
    active_path = save_active_session(session, result_root)
    figures = []
    channel_names = session.list_channels()
    fig, axes = plt.subplots(1, len(channel_names), figsize=(5.0 * len(channel_names), 4.5))
    axes = [axes] if len(channel_names) == 1 else list(axes)
    for axis, channel_name in zip(axes, channel_names, strict=True):
        plot_raw_image(
            session.get_channel_state(channel_name).raw_image,
            title=f"{channel_name} raw",
            ax=axis,
            **_plot_calibration_kwargs(session),
        )
    fig.tight_layout()
    figures.append(fig)
    return NotebookResult(
        session=session,
        active_path=active_path,
        tables=[generic_stage_summary(session, active_path), metadata_preview(session), channel_summary(session)],
        figures=figures,
        messages=[
            f"session: {session.name}",
            f"workflow_mode: {session.workflow_mode}",
            f"current_stage: {session.current_stage}",
            f"primary_channel: {session.primary_channel}",
            f"active_session: {active_path}",
        ],
    )


def run_generic_candidate_detection(
    session: AnalysisSession,
    *,
    result_root: str | Path,
    detection_configs_by_channel: dict[str, DetectionConfig],
    merge_dedupe_radius_px: float | None = None,
    open_viewer: bool = False,
) -> NotebookResult:
    _require_generic_classification_session(session)
    _clear_preprocess_results(session, list(detection_configs_by_channel.keys()))
    session = detect_multichannel_candidates(
        session,
        detection_configs_by_channel,
        dedupe_radius_px=merge_dedupe_radius_px,
    )
    fig, _ = plot_atom_overlay(
        session.get_processed_image(session.primary_channel),
        session.candidate_points,
        title="候选原子柱总览",
        origin_xy=session.get_processed_origin(session.primary_channel),
        point_size=12.0,
        **_plot_calibration_kwargs(session),
    )
    messages = []
    if open_viewer:
        try:
            viewer = launch_detection_napari_viewer(session, show_raw_layer=False)
            viewer.show(block=True)
            messages.append("候选点只读 napari 总览已关闭。")
        except Exception as exc:
            messages.append(f"候选点 napari 总览失败: {type(exc).__name__}: {exc}")
    active_path = save_active_session(session, result_root)
    messages.append(f"candidate_count: {len(session.candidate_points)}")
    return NotebookResult(
        session=session,
        active_path=active_path,
        tables=[generic_stage_summary(session, active_path)],
        figures=[fig],
        messages=messages,
    )


def review_generic_candidates(
    session: AnalysisSession,
    *,
    result_root: str | Path,
    open_viewer: bool = False,
    image_channel: str | None = None,
    image_key: str = "processed",
    point_size: float = 6.0,
) -> NotebookResult:
    _require_generic_classification_session(session)
    if session.candidate_points.empty:
        raise RuntimeError("candidate_points are missing; run candidate detection before candidate review.")
    if not open_viewer:
        return NotebookResult(
            session=session,
            messages=[
                "Set OPEN_CANDIDATE_REVIEW_VIEWER = True, run the napari candidate review cell, "
                "then continue to classification."
            ],
        )
    channel_name = image_channel or session.primary_channel
    try:
        previous_count = len(session.candidate_points)
        session = edit_candidates_with_napari(
            session,
            image_key=image_key,
            channel_name=channel_name,
            point_size=point_size,
        )
        if image_key == "processed":
            display_image = session.get_processed_image(channel_name)
            origin_xy = session.get_processed_origin(channel_name)
        else:
            display_image = session.get_channel_state(channel_name).raw_image
            origin_xy = (0, 0)
        fig, _ = plot_atom_overlay(
            display_image,
            session.candidate_points,
            title="人工复核后的候选原子柱",
            origin_xy=origin_xy,
            point_size=12.0,
            point_color="#00d7ff",
            **_plot_calibration_kwargs(session),
        )
        active_path = save_active_session(session, result_root)
        return NotebookResult(
            session=session,
            active_path=active_path,
            tables=[generic_stage_summary(session, active_path), session.candidate_points.head()],
            figures=[fig],
            messages=[
                "Candidate review viewer closed and edits were applied.",
                f"candidate_count: {previous_count} -> {len(session.candidate_points)}",
            ],
        )
    except Exception as exc:
        return NotebookResult(session=session, messages=[f"Candidate review failed: {type(exc).__name__}: {exc}"])


def run_generic_refinement(
    session: AnalysisSession,
    *,
    result_root: str | Path,
    refinement_config: RefinementConfig,
    class_refinement_overrides: dict[int | str, RefinementConfig | dict[str, object]] | None = None,
    source_table: str = "candidate",
    nn_context_mode: str = "all",
) -> NotebookResult:
    _require_generic_classification_session(session)
    if session.candidate_points.empty:
        raise RuntimeError("candidate_points are missing; run candidate detection first.")
    if stage_rank(session.current_stage) < stage_rank("candidate_reviewed"):
        raise RuntimeError(
            "candidate_points must be reviewed in napari before refinement. "
            "Run the candidate review stage with OPEN_CANDIDATE_REVIEW_VIEWER = True."
        )
    if stage_rank(session.current_stage) < stage_rank("classified"):
        raise RuntimeError(
            "candidate_points must be classified and class-reviewed before refinement. "
            "Run candidate review, automatic classification, and class review first."
        )
    if "class_id" not in session.candidate_points.columns:
        raise RuntimeError("class_id is missing; run atom-column classification before refinement.")
    _clear_preprocess_results(session, session.list_channels())
    session.clear_downstream_results("classified")
    session = refine_points_by_class(
        session,
        refinement_config,
        class_refinement_overrides,
        source_table=source_table,
        nn_context_mode=nn_context_mode,
    )
    fig, _ = plot_atom_overlay(
        session.get_processed_image(session.primary_channel),
        session.refined_points,
        title="按类别精修后的原子柱坐标",
        origin_xy=session.get_processed_origin(session.primary_channel),
        point_size=12.0,
        point_color="#f18f01",
        **_plot_calibration_kwargs(session),
    )
    active_path = save_active_session(session, result_root)
    return NotebookResult(
        session=session,
        active_path=active_path,
        tables=[generic_stage_summary(session, active_path), session.refined_points.head()],
        figures=[fig],
        messages=[
            f"refined_count: {len(session.refined_points)}",
            f"refinement_config_sources: {sorted(session.refined_points['refinement_config_source'].dropna().unique().tolist())}",
            f"nn_context_mode: {nn_context_mode}",
        ],
    )


def run_atom_column_classification(
    session: AnalysisSession,
    *,
    result_root: str | Path,
    classification_config: AtomColumnClassificationConfig,
    class_name_map: dict[int | str, str] | None = None,
    class_color_map: dict[int | str, str] | None = None,
) -> NotebookResult:
    _require_generic_classification_session(session)
    if str(classification_config.source_table).lower() == "candidate":
        session.clear_downstream_results("detect")
    session = classify_atom_columns(session, classification_config)
    session = apply_class_name_map(session, class_name_map, class_color_map)
    points = session.candidate_points if str(classification_config.source_table).lower() == "candidate" else (
        session.refined_points if not session.refined_points.empty else session.candidate_points
    )
    overlay_fig, _ = plot_class_overlay(
        session.get_processed_image(session.primary_channel),
        points,
        title="自动聚类类别叠加图",
        origin_xy=session.get_processed_origin(session.primary_channel),
        point_size=16.0,
        **_plot_calibration_kwargs(session),
    )
    figures = [overlay_fig]
    try:
        scatter_fig, _ = plot_class_feature_scatter_matrix(session.classification_features, points)
        figures.append(scatter_fig)
    except Exception:
        pass
    active_path = save_active_session(session, result_root)
    return NotebookResult(
        session=session,
        active_path=active_path,
        tables=[
            generic_stage_summary(session, active_path),
            classification_summary_table(session),
            session.classification_features.head(),
        ],
        figures=figures,
        messages=[f"classified_count: {len(points)}"],
    )


def run_generic_curation(
    session: AnalysisSession,
    *,
    result_root: str | Path,
    curation_config: CurationConfig,
) -> NotebookResult:
    _require_generic_classification_session(session)
    if "class_id" not in session.get_atom_table(preferred="refined").columns:
        raise RuntimeError("class_id is missing; run atom-column classification before final curation.")
    session = curate_points(session, curation_config)
    display_points = session.curated_points.query("keep == True") if "keep" in session.curated_points.columns else session.curated_points
    fig, _ = plot_class_overlay(
        session.get_processed_image(session.primary_channel),
        display_points if not display_points.empty else session.curated_points,
        title="最终保留原子柱类别叠加图",
        origin_xy=session.get_processed_origin(session.primary_channel),
        point_size=16.0,
        **_plot_calibration_kwargs(session),
    )
    active_path = save_active_session(session, result_root)
    return NotebookResult(
        session=session,
        active_path=active_path,
        tables=[generic_stage_summary(session, active_path), classification_summary_table(session), session.curated_points.head()],
        figures=[fig],
        messages=[f"curated_count: {len(session.curated_points)}", f"auto_keep_count: {len(display_points)}"],
    )


def show_atom_column_class_review(
    session: AnalysisSession,
    *,
    result_root: str | Path | None = None,
    open_viewer: bool = False,
    image_channel: str | None = None,
    point_size: float = 6.0,
    source_table: str = "refined",
) -> NotebookResult:
    _require_generic_classification_session(session)
    if not open_viewer:
        return NotebookResult(session=session, messages=["Set OPEN_CLASS_REVIEW_VIEWER = True to open napari class review."])
    try:
        viewer = launch_class_review_napari(
            session,
            image_channel=image_channel,
            point_size=point_size,
            source_table=source_table,
        )
        viewer.show(block=True)
        session = apply_class_review_from_viewer(session, viewer, source_table=source_table)
        active_path = save_active_session(session, result_root) if result_root is not None else None
        return NotebookResult(session=session, active_path=active_path, messages=["Class review viewer closed and edits were applied."])
    except Exception as exc:
        return NotebookResult(session=session, messages=[f"Class review viewer failed: {type(exc).__name__}: {exc}"])


def initialize_hfo2_multichannel_session(
    *,
    result_root: str | Path,
    idpc_path: str | Path | None,
    haadf_path: str | Path | None,
    abf_path: str | Path | None = None,
    channel_dataset_indices: dict[str, int | None] | None = None,
    bundle_manual_calibration: PixelCalibration | dict[str, Any] | float | None = None,
    primary_channel: str = "idpc",
    synthetic_include_abf: bool = True,
    synthetic_rng_seed: int = 11,
) -> NotebookResult:
    if primary_channel != "idpc":
        raise ValueError("hfo2_multichannel requires PRIMARY_CHANNEL = 'idpc'.")

    result_root = Path(result_root)
    if idpc_path is None and haadf_path is None:
        session = build_synthetic_hfo2_session(
            include_abf=synthetic_include_abf,
            rng_seed=synthetic_rng_seed,
        )
    else:
        session = _load_real_hfo2_session(
            idpc_path=idpc_path,
            haadf_path=haadf_path,
            abf_path=abf_path,
            channel_dataset_indices=channel_dataset_indices,
            bundle_manual_calibration=bundle_manual_calibration,
            primary_channel=primary_channel,
        )

    session.set_primary_channel("idpc")
    session.set_workflow(
        "hfo2_multichannel",
        {
            "primary_channel": "idpc",
            "heavy_channel": "haadf",
            "light_channel": "idpc",
            "confirm_channel": "abf" if "abf" in session.list_channels() else None,
        },
    )
    active_path = save_active_session(session, result_root)

    figures = []
    channel_names = session.list_channels()
    fig, axes = plt.subplots(1, len(channel_names), figsize=(5.2 * len(channel_names), 4.8))
    axes = [axes] if len(channel_names) == 1 else list(axes)
    for axis, channel_name in zip(axes, channel_names, strict=True):
        plot_raw_image(
            session.get_channel_state(channel_name).raw_image,
            title=f"{channel_name.upper()} raw image",
            ax=axis,
            **_plot_calibration_kwargs(session),
        )
    plt.tight_layout()
    figures.append(fig)

    return NotebookResult(
        session=session,
        active_path=active_path,
        tables=[
            hfo2_stage_summary(session, active_path),
            metadata_preview(session),
            hfo2_channel_summary(session),
        ],
        figures=figures,
        messages=[
            f"session: {session.name}",
            f"workflow_mode: {session.workflow_mode}",
            f"current_stage: {session.current_stage}",
            f"primary_channel: {session.primary_channel}",
            f"active_session: {active_path}",
        ],
    )


def build_synthetic_hfo2_session(*, include_abf: bool = True, rng_seed: int = 11) -> AnalysisSession:
    images, truth = synthetic_hfo2_multichannel_bundle(include_abf=include_abf, rng_seed=rng_seed)
    session = AnalysisSession(
        name="synthetic_multichannel_demo",
        raw_image=images["idpc"],
        raw_metadata={
            "source": "synthetic_hfo2_multichannel_demo",
            "heavy_truth_count": len(truth["heavy"]),
            "light_truth_count": len(truth["light"]),
            "rng_seed": rng_seed,
        },
        pixel_calibration=PixelCalibration(size=0.2, unit="A", source="synthetic_demo"),
        contrast_mode="bright_peak",
        primary_channel="idpc",
    )
    for channel_name, image in images.items():
        session.set_channel_state(
            channel_name,
            input_path=f"synthetic://{channel_name}",
            raw_image=image,
            raw_metadata={"source": "synthetic_hfo2_multichannel_demo", "channel": channel_name},
            contrast_mode="dark_dip" if channel_name == "abf" else "bright_peak",
        )
    session.set_primary_channel("idpc")
    session.set_stage("loaded")
    session.record_step(
        "load_synthetic_hfo2_multichannel_demo",
        notes={
            "heavy_truth_count": len(truth["heavy"]),
            "light_truth_count": len(truth["light"]),
            "include_abf": bool(include_abf),
        },
    )
    return session


def run_hfo2_heavy_detection(
    session: AnalysisSession,
    *,
    result_root: str | Path,
    heavy_detection: DetectionConfig,
    open_viewer: bool = False,
) -> NotebookResult:
    _require_hfo2_session(session)
    channels = workflow_channels(session)
    config = HfO2MultichannelDetectionConfig(
        heavy_channel=channels.heavy,
        light_channel=channels.light,
        confirm_channel=channels.confirm,
        heavy_detection=heavy_detection,
    )

    _clear_preprocess_results(session, [channels.heavy])
    session = detect_hfo2_heavy_candidates(session, config)
    heavy_points = filter_points_by_role(session.candidate_points, "heavy_atom")
    if heavy_points.empty:
        heavy_points = session.candidate_points.copy()

    fig, _ = plot_atom_overlay(
        session.get_processed_image(channels.heavy),
        heavy_points,
        title="HAADF heavy-column candidates",
        origin_xy=session.get_processed_origin(channels.heavy),
        point_size=14.0,
        point_color="#ff8c00",
        **_plot_calibration_kwargs(session),
    )

    messages = []
    if open_viewer:
        try:
            session = edit_hfo2_heavy_candidates_with_napari(
                session,
                heavy_channel=channels.heavy,
                image_key="processed",
            )
            messages.append("HAADF heavy-column napari review finished.")
        except Exception as exc:
            messages.append(f"HAADF heavy-column napari review failed: {type(exc).__name__}: {exc}")

    active_path = save_active_session(session, result_root)
    messages.append(f"heavy_count: {len(filter_points_by_role(session.candidate_points, 'heavy_atom'))}")
    return NotebookResult(
        session=session,
        active_path=active_path,
        tables=[hfo2_stage_summary(session, active_path)],
        figures=[fig],
        messages=messages,
    )


def run_hfo2_light_detection(
    session: AnalysisSession,
    *,
    result_root: str | Path,
    light_detection: DetectionConfig,
    light_options: dict[str, Any],
    open_viewer: bool = False,
) -> NotebookResult:
    session, heavy_points = _require_heavy_anchors(session)
    channels = workflow_channels(session)
    config = HfO2MultichannelDetectionConfig(
        heavy_channel=channels.heavy,
        light_channel=channels.light,
        confirm_channel=channels.confirm,
        light_detection=light_detection,
        **dict(light_options),
    )

    channels_to_clear = [channels.light]
    if channels.confirm is not None:
        channels_to_clear.append(channels.confirm)
    _clear_preprocess_results(session, channels_to_clear)
    session = detect_hfo2_light_candidates(session, config, heavy_points=heavy_points)

    heavy_points = filter_points_by_role(session.candidate_points, "heavy_atom")
    light_points = filter_points_by_role(session.candidate_points, "light_atom")
    light_fig, _ = plot_atom_overlay(
        session.get_processed_image(channels.light),
        light_points,
        title="iDPC light-column candidates",
        origin_xy=session.get_processed_origin(channels.light),
        point_size=14.0,
        point_color="#00a5cf",
        **_plot_calibration_kwargs(session),
    )
    heavy_fig, _ = plot_atom_overlay(
        session.get_processed_image(channels.heavy),
        heavy_points,
        title="HAADF heavy-column reference",
        origin_xy=session.get_processed_origin(channels.heavy),
        point_size=14.0,
        point_color="#ff8c00",
        **_plot_calibration_kwargs(session),
    )

    messages = []
    if open_viewer:
        try:
            session = edit_hfo2_light_candidates_with_napari(
                session,
                heavy_channel=channels.heavy,
                light_channel=channels.light,
                image_key="processed",
            )
            messages.append("iDPC light-column napari review finished.")
        except Exception as exc:
            messages.append(f"iDPC light-column napari review failed: {type(exc).__name__}: {exc}")

    active_path = save_active_session(session, result_root)
    messages.extend(
        [
            f"heavy_count: {len(filter_points_by_role(session.candidate_points, 'heavy_atom'))}",
            f"light_count: {len(filter_points_by_role(session.candidate_points, 'light_atom'))}",
        ]
    )
    return NotebookResult(
        session=session,
        active_path=active_path,
        tables=[hfo2_stage_summary(session, active_path)],
        figures=[light_fig, heavy_fig],
        messages=messages,
    )


def show_hfo2_detection_overview(
    session: AnalysisSession,
    *,
    open_viewer: bool = False,
    show_raw_layer: bool = False,
) -> NotebookResult:
    _require_hfo2_session(session)
    channels = workflow_channels(session)
    heavy_points = filter_points_by_role(session.candidate_points, "heavy_atom")
    light_points = filter_points_by_role(session.candidate_points, "light_atom")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    plot_atom_overlay(
        session.get_processed_image(channels.heavy),
        heavy_points,
        ax=axes[0],
        title="HAADF heavy-column overview",
        origin_xy=session.get_processed_origin(channels.heavy),
        point_size=14.0,
        point_color="#ff8c00",
        **_plot_calibration_kwargs(session),
    )
    plot_atom_overlay(
        session.get_processed_image(channels.light),
        light_points,
        ax=axes[1],
        title="iDPC light-column overview",
        origin_xy=session.get_processed_origin(channels.light),
        point_size=14.0,
        point_color="#00a5cf",
        **_plot_calibration_kwargs(session),
    )
    plt.tight_layout()

    messages = []
    if open_viewer:
        try:
            viewer = launch_detection_napari_viewer(session, show_raw_layer=show_raw_layer)
            viewer.show(block=True)
            messages.append("Read-only detection overview closed.")
        except Exception as exc:
            messages.append(f"Read-only detection overview failed: {type(exc).__name__}: {exc}")
    else:
        messages.append("Set OPEN_DETECTION_OVERVIEW_VIEWER = True to open the read-only napari overview.")

    return NotebookResult(session=session, figures=[fig], messages=messages)


def run_hfo2_refine_curate(
    session: AnalysisSession,
    *,
    result_root: str | Path,
    refinement_config: RefinementConfig,
    curation_config: CurationConfig,
) -> NotebookResult:
    _require_detection_ready(session)
    channels = workflow_channels(session)

    _clear_preprocess_results(session, session.list_channels())
    session.clear_downstream_results("detect")
    session = refine_points(session, refinement_config)
    session = curate_points(session, curation_config)

    display_points = (
        session.curated_points.query("keep == True")
        if "keep" in session.curated_points.columns
        else session.curated_points
    )
    overlay_points = display_points if not display_points.empty else session.refined_points
    heavy_overlay = filter_points_by_role(overlay_points, "heavy_atom")
    light_overlay = filter_points_by_role(overlay_points, "light_atom")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    plot_atom_overlay(
        session.get_processed_image(channels.heavy),
        heavy_overlay,
        ax=axes[0],
        title="HAADF refined heavy columns",
        origin_xy=session.get_processed_origin(channels.heavy),
        point_size=14.0,
        point_color="#ff8c00",
        **_plot_calibration_kwargs(session),
    )
    plot_atom_overlay(
        session.get_processed_image(channels.light),
        light_overlay,
        ax=axes[1],
        title="iDPC refined light columns",
        origin_xy=session.get_processed_origin(channels.light),
        point_size=14.0,
        point_color="#00a5cf",
        **_plot_calibration_kwargs(session),
    )
    plot_histogram_or_distribution(
        session.curated_points.get("quality_score", pd.Series(dtype=float)),
        title="Refinement quality",
        xlabel="quality_score",
        ax=axes[2],
    )
    plt.tight_layout()

    active_path = save_active_session(session, result_root)
    return NotebookResult(
        session=session,
        active_path=active_path,
        tables=[hfo2_stage_summary(session, active_path), session.curated_points.head()],
        figures=[fig],
        messages=[
            f"refined_count: {len(session.refined_points)}",
            f"auto_keep_count: {len(display_points)}",
        ],
    )


def show_hfo2_refinement_viewer(
    session: AnalysisSession,
    *,
    open_viewer: bool = False,
    show_raw_layer: bool = False,
    show_candidate_layer: bool = False,
    point_size: float = 5.0,
) -> NotebookResult:
    _require_hfo2_session(session)
    if session.refined_points.empty:
        return NotebookResult(session=session, messages=["No refined_points yet; run the refinement stage first."])

    if open_viewer:
        try:
            viewer = launch_refinement_napari_viewer(
                session,
                show_raw_layer=show_raw_layer,
                show_candidate_layer=show_candidate_layer,
                point_size=point_size,
            )
            viewer.show(block=True)
            return NotebookResult(session=session, messages=["Read-only refinement viewer closed."])
        except Exception as exc:
            return NotebookResult(
                session=session,
                messages=[f"Read-only refinement viewer failed: {type(exc).__name__}: {exc}"],
            )
    return NotebookResult(
        session=session,
        messages=["Set OPEN_REFINEMENT_VIEWER = True to open the read-only refinement viewer."],
    )


def save_final_checkpoint_if_requested(
    session: AnalysisSession | None,
    *,
    result_root: str | Path,
    filename: str,
    enabled: bool,
) -> NotebookResult:
    if session is None:
        return NotebookResult(messages=["No session yet; run the main workflow stages first."])
    if session.current_stage != "curated":
        return NotebookResult(
            session=session,
            messages=[f"Current stage is {session.current_stage!r}; finish refinement/curation first."],
        )
    if enabled:
        checkpoint_path = save_checkpoint(session, result_root, filename)
        return NotebookResult(session=session, messages=[f"Final checkpoint saved: {checkpoint_path}"])
    return NotebookResult(
        session=session,
        messages=["Final checkpoint not saved; notebook 03 can continue from the active session."],
    )


def _ordered_final_atom_columns(table: pd.DataFrame) -> list[str]:
    preferred = [
        "atom_id",
        "candidate_id",
        "x_px",
        "y_px",
        "x_nm",
        "y_nm",
        "class_id",
        "class_name",
        "class_color",
        "class_confidence",
        "class_source",
        "class_reviewed",
        "keep",
        "quality_score",
        "fit_residual",
        "fit_success",
        "fit_method",
        "fit_amplitude",
        "fit_background",
        "fit_sigma_x",
        "fit_sigma_y",
        "center_intensity",
        "local_background",
        "prominence",
        "local_snr",
        "integrated_intensity",
        "column_role",
        "seed_channel",
        "detected_from_channels",
        "flag_duplicate",
        "flag_edge",
        "flag_low_quality",
        "flag_poor_fit",
        "flag_spacing_violation",
    ]
    return [column for column in preferred if column in table.columns] + [
        column for column in table.columns if column not in preferred
    ]


def _final_atom_table_with_physical_units(session: AnalysisSession) -> pd.DataFrame:
    table = session.curated_points.copy()
    calibration = session.pixel_calibration
    pixel_size = getattr(calibration, "size", None)
    unit = str(getattr(calibration, "unit", "px") or "px").strip().lower()
    unit_to_nm = {
        "nm": 1.0,
        "nanometer": 1.0,
        "nanometers": 1.0,
        "a": 0.1,
        "å": 0.1,
        "angstrom": 0.1,
        "angstroms": 0.1,
        "pm": 0.001,
        "picometer": 0.001,
        "picometers": 0.001,
    }
    if pixel_size is not None and unit in unit_to_nm and "x_px" in table.columns and "y_px" in table.columns:
        scale_nm = float(pixel_size) * unit_to_nm[unit]
        if "x_nm" not in table.columns:
            table["x_nm"] = pd.to_numeric(table["x_px"], errors="coerce") * scale_nm
        if "y_nm" not in table.columns:
            table["y_nm"] = pd.to_numeric(table["y_px"], errors="coerce") * scale_nm
    return table[_ordered_final_atom_columns(table)]


def _final_atom_class_summary(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    group_columns = [column for column in ("class_id", "class_name", "class_color") if column in table.columns]
    if not group_columns:
        kept_rows = int(table["keep"].fillna(False).astype(bool).sum()) if "keep" in table.columns else len(table)
        return pd.DataFrame({"total_rows": [len(table)], "kept_rows": [kept_rows]})
    work = table.copy()
    if "keep" not in work.columns:
        work["keep"] = True
    agg_spec: dict[str, tuple[str, str]] = {
        "total_rows": ("atom_id" if "atom_id" in work.columns else group_columns[0], "count"),
        "kept_rows": ("keep", "sum"),
    }
    if "quality_score" in work.columns:
        agg_spec["mean_quality_score"] = ("quality_score", "mean")
        agg_spec["median_quality_score"] = ("quality_score", "median")
    if "fit_residual" in work.columns:
        agg_spec["mean_fit_residual"] = ("fit_residual", "mean")
        agg_spec["median_fit_residual"] = ("fit_residual", "median")
    return work.groupby(group_columns, dropna=False).agg(**agg_spec).reset_index()


def _final_atom_flag_summary(table: pd.DataFrame) -> pd.DataFrame:
    flag_columns = [column for column in table.columns if column.startswith("flag_")]
    rows = []
    for column in flag_columns:
        values = table[column].fillna(False).astype(bool)
        rows.append({"flag": column, "true_count": int(values.sum()), "false_count": int((~values).sum())})
    return pd.DataFrame(rows)


def _xlsx_column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _xlsx_cell_ref(row_index: int, column_index: int) -> str:
    return f"{_xlsx_column_name(column_index)}{row_index}"


def _xlsx_safe_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)


def _xlsx_safe_sheet_name(name: str, used: set[str]) -> str:
    safe = re.sub(r"[\[\]:*?/\\]", "_", str(name)).strip() or "Sheet"
    safe = safe[:31]
    candidate = safe
    counter = 2
    while candidate in used:
        suffix = f"_{counter}"
        candidate = f"{safe[:31 - len(suffix)]}{suffix}"
        counter += 1
    used.add(candidate)
    return candidate


def _xlsx_cell_xml(value: Any, row_index: int, column_index: int) -> str:
    cell_ref = _xlsx_cell_ref(row_index, column_index)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f'<c r="{cell_ref}"/>'
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, set, np.ndarray)) else False:
        return f'<c r="{cell_ref}"/>'
    if isinstance(value, (bool, np.bool_)):
        return f'<c r="{cell_ref}" t="b"><v>{1 if bool(value) else 0}</v></c>'
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            return f'<c r="{cell_ref}"><v>{numeric:.15g}</v></c>'
    text = escape(_xlsx_safe_text(value))
    return f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _write_minimal_xlsx(output_path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    """Write a dependency-free .xlsx workbook for notebook exports.

    This fallback intentionally keeps formatting simple. It exists so the final
    atom export works in existing notebook kernels that do not have openpyxl.
    """
    used_names: set[str] = set()
    safe_sheets = [(_xlsx_safe_sheet_name(name, used_names), table.copy()) for name, table in sheets.items()]

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for index, _ in enumerate(safe_sheets, start=1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    workbook_sheets = [
        f'<sheet name="{escape(sheet_name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, (sheet_name, _) in enumerate(safe_sheets, start=1)
    ]
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(workbook_sheets)}</sheets>'
        "</workbook>"
    )
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for index, _ in enumerate(safe_sheets, start=1):
        workbook_rels.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
    workbook_rels.append(
        f'<Relationship Id="rId{len(safe_sheets) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )
    workbook_rels.append("</Relationships>")

    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '</styleSheet>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        '</Relationships>'
    )
    app_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        '<Application>em_atom_workbench</Application>'
        '</Properties>'
    )
    core_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:creator>em_atom_workbench</dc:creator>'
        '<dc:title>Final atom table export</dc:title>'
        '</cp:coreProperties>'
    )

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "\n".join(content_types))
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("docProps/app.xml", app_xml)
        archive.writestr("docProps/core.xml", core_xml)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", "\n".join(workbook_rels))
        archive.writestr("xl/styles.xml", styles_xml)
        for sheet_index, (_, table) in enumerate(safe_sheets, start=1):
            columns = [str(column) for column in table.columns]
            row_xml = [
                '<row r="1">'
                + "".join(_xlsx_cell_xml(column, 1, column_index) for column_index, column in enumerate(columns, start=1))
                + "</row>"
            ]
            for row_index, row in enumerate(table.itertuples(index=False, name=None), start=2):
                row_xml.append(
                    f'<row r="{row_index}">'
                    + "".join(_xlsx_cell_xml(value, row_index, column_index) for column_index, value in enumerate(row, start=1))
                    + "</row>"
                )
            max_row = max(len(table) + 1, 1)
            max_col = max(len(columns), 1)
            dimension = f"A1:{_xlsx_cell_ref(max_row, max_col)}"
            worksheet_xml = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                f'<dimension ref="{dimension}"/>'
                '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
                'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
                f'<sheetData>{"".join(row_xml)}</sheetData>'
                f'<autoFilter ref="{dimension}"/>'
                '</worksheet>'
            )
            archive.writestr(f"xl/worksheets/sheet{sheet_index}.xml", worksheet_xml)


def _write_excel_workbook(output_path: Path, sheets: dict[str, pd.DataFrame]) -> str:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        _write_minimal_xlsx(output_path, sheets)
        return "minimal_xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, table in sheets.items():
            table.to_excel(writer, sheet_name=sheet_name, index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_length + 2, 10), 32)
    return "openpyxl"


def export_final_atom_table_excel(
    session: AnalysisSession | None,
    *,
    result_root: str | Path,
    filename: str = "01_final_atom_columns.xlsx",
    kept_only_sheet: bool = True,
) -> NotebookResult:
    if session is None:
        raise RuntimeError("No session is loaded. Load the active session first or run the previous 01 notebook stages.")
    if session.curated_points is None or session.curated_points.empty:
        raise RuntimeError("curated_points is empty. Run the final curation stage before exporting Excel.")

    output_dir = Path(result_root) / "01_findatom" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    all_points = _final_atom_table_with_physical_units(session)
    if "keep" in all_points.columns:
        kept_points = all_points.loc[all_points["keep"] == True].copy()  # noqa: E712
    else:
        kept_points = all_points.copy()
    class_summary = _final_atom_class_summary(all_points)
    flag_summary = _final_atom_flag_summary(all_points)
    calibration = session.pixel_calibration
    metadata = pd.DataFrame(
        [
            {"field": "session_name", "value": session.name},
            {"field": "current_stage", "value": session.current_stage},
            {"field": "primary_channel", "value": session.primary_channel},
            {"field": "curated_rows", "value": len(all_points)},
            {"field": "kept_rows", "value": len(kept_points)},
            {"field": "pixel_size", "value": getattr(calibration, "size", None)},
            {"field": "pixel_unit", "value": getattr(calibration, "unit", None)},
            {"field": "calibration_source", "value": getattr(calibration, "source", None)},
        ]
    )

    sheets = {"all_curated_points": all_points}
    if kept_only_sheet:
        sheets["kept_points"] = kept_points
    sheets.update(
        {
            "class_summary": class_summary,
            "flag_summary": flag_summary,
            "metadata": metadata,
        }
    )
    writer_engine = _write_excel_workbook(output_path, sheets)

    return NotebookResult(
        session=session,
        tables=[metadata, class_summary, flag_summary],
        messages=[f"Final atom Excel exported: {output_path}", f"Excel writer: {writer_engine}"],
    )


def _load_real_hfo2_session(
    *,
    idpc_path: str | Path | None,
    haadf_path: str | Path | None,
    abf_path: str | Path | None,
    channel_dataset_indices: dict[str, int | None] | None,
    bundle_manual_calibration: PixelCalibration | dict[str, Any] | float | None,
    primary_channel: str,
) -> AnalysisSession:
    if not idpc_path or not haadf_path:
        raise ValueError("Real HfO2 mode requires both IDPC_PATH and HAADF_PATH.")

    path_map = {"idpc": Path(idpc_path), "haadf": Path(haadf_path)}
    if abf_path:
        path_map["abf"] = Path(abf_path)

    missing = [str(path) for path in path_map.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing multichannel input files:\n" + "\n".join(missing))

    contrast_modes = {"idpc": "bright_peak", "haadf": "bright_peak"}
    if "abf" in path_map:
        contrast_modes["abf"] = "dark_dip"

    return load_image_bundle(
        path_map,
        primary_channel=primary_channel,
        manual_calibration=bundle_manual_calibration,
        contrast_modes=contrast_modes,
        dataset_indices=channel_dataset_indices,
    )


def _require_hfo2_session(session: AnalysisSession | None) -> AnalysisSession:
    if session is None:
        raise RuntimeError("No session yet; initialize the HfO2 multichannel session first.")
    if session.workflow_mode != "hfo2_multichannel":
        raise RuntimeError(f"Expected hfo2_multichannel, got {session.workflow_mode!r}.")
    return session


def _require_generic_classification_session(session: AnalysisSession | None) -> AnalysisSession:
    if session is None:
        raise RuntimeError("No session yet; initialize the atom-column classification session first.")
    if session.workflow_mode != "atom_column_classification":
        raise RuntimeError(f"Expected atom_column_classification, got {session.workflow_mode!r}.")
    return session


def _require_heavy_anchors(session: AnalysisSession) -> tuple[AnalysisSession, pd.DataFrame]:
    _require_hfo2_session(session)
    heavy_points = filter_points_by_role(session.candidate_points, "heavy_atom")
    if heavy_points.empty:
        raise RuntimeError("Heavy anchors are missing; run the HAADF heavy-column stage first.")
    return session, heavy_points


def _require_detection_ready(session: AnalysisSession) -> AnalysisSession:
    _require_hfo2_session(session)
    if session.current_stage == "heavy_reviewed":
        raise RuntimeError("Only the heavy-column stage is complete; run the iDPC light-column stage next.")
    if session.candidate_points.empty:
        raise RuntimeError("candidate_points are missing; run heavy and light detection first.")
    return session


def _safe_export_sheet(table: pd.DataFrame | None, sheet_name: str) -> pd.DataFrame:
    if table is None:
        return pd.DataFrame({"warning": [f"{sheet_name} table is not available"]})
    if isinstance(table, pd.DataFrame) and table.empty and len(table.columns) == 0:
        return pd.DataFrame({"warning": [f"{sheet_name} table is empty"]})
    return table.copy()


def _export_task_excel(output_path: str | Path, sheets: dict[str, pd.DataFrame | None]) -> dict[str, Any]:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    safe_sheets = {name: _safe_export_sheet(table, name) for name, table in sheets.items()}
    writer_engine = _write_excel_workbook(target, safe_sheets)
    warnings = [
        {"sheet": name, "warning": str(table.iloc[0]["warning"])}
        for name, table in safe_sheets.items()
        if list(table.columns) == ["warning"]
    ]
    return {"path": str(target), "writer_engine": writer_engine, "warnings": warnings}


def export_task1A_excel(
    output_dirs: dict[str, Path],
    *,
    period_segment_table: pd.DataFrame,
    period_summary_table: pd.DataFrame,
    task1A_config: pd.DataFrame,
    roi_class_selection: pd.DataFrame,
    basis_vectors: pd.DataFrame,
    filename: str = "task1A_period_statistics.xlsx",
) -> dict[str, Any]:
    return _export_task_excel(
        output_dirs["tables"] / filename,
        {
            "period_segments": period_segment_table,
            "period_summary": period_summary_table,
            "task1A_config": task1A_config,
            "roi_class_selection": roi_class_selection,
            "basis_vectors": basis_vectors,
        },
    )


def export_task1B_excel(
    output_dirs: dict[str, Path],
    *,
    cell_table: pd.DataFrame,
    strain_reference_table: pd.DataFrame,
    task1B_config: pd.DataFrame,
    anchor_selection: pd.DataFrame,
    qc_summary: pd.DataFrame,
    filename: str = "task1B_polygon_strain_mapping.xlsx",
) -> dict[str, Any]:
    valid_cell_table = cell_table.loc[cell_table.get("valid", False).astype(bool)].copy() if cell_table is not None and not cell_table.empty else pd.DataFrame()
    invalid_cell_table = cell_table.loc[~cell_table.get("valid", False).astype(bool)].copy() if cell_table is not None and not cell_table.empty else pd.DataFrame()
    return _export_task_excel(
        output_dirs["tables"] / filename,
        {
            "cell_table": cell_table,
            "valid_cells": valid_cell_table,
            "invalid_cells": invalid_cell_table,
            "strain_reference": strain_reference_table,
            "task1B_config": task1B_config,
            "anchor_selection": anchor_selection,
            "qc_summary": qc_summary,
        },
    )


def export_task2_excel(
    output_dirs: dict[str, Path],
    *,
    pair_table: pd.DataFrame,
    pair_line_summary_table: pd.DataFrame,
    task2_config: pd.DataFrame,
    projection_axis_table: pd.DataFrame,
    line_grouping_summary: pd.DataFrame,
    filename: str = "task2_pair_line_statistics.xlsx",
) -> dict[str, Any]:
    valid_pair_table = pair_table.loc[pair_table.get("valid", False).astype(bool)].copy() if pair_table is not None and not pair_table.empty else pd.DataFrame()
    invalid_pair_table = pair_table.loc[~pair_table.get("valid", False).astype(bool)].copy() if pair_table is not None and not pair_table.empty else pd.DataFrame()
    return _export_task_excel(
        output_dirs["tables"] / filename,
        {
            "pair_table": pair_table,
            "valid_pairs": valid_pair_table,
            "invalid_pairs": invalid_pair_table,
            "line_summary": pair_line_summary_table,
            "task2_config": task2_config,
            "projection_axis": projection_axis_table,
            "line_grouping": line_grouping_summary,
        },
    )


def export_task3_excel(
    output_dirs: dict[str, Path],
    *,
    group_centroid_table: pd.DataFrame,
    group_displacement_table: pd.DataFrame,
    task3_roi_table: pd.DataFrame,
    task3_group_config: pd.DataFrame,
    task3_summary: pd.DataFrame,
    filename: str = "task3_group_center_displacement.xlsx",
) -> dict[str, Any]:
    return _export_task_excel(
        output_dirs["tables"] / filename,
        {
            "group_centroids": group_centroid_table,
            "group_displacements": group_displacement_table,
            "task3_rois": task3_roi_table,
            "task3_groups": task3_group_config,
            "task3_summary": task3_summary,
        },
    )


def save_notebook02_figures(
    figures: dict[str, Any],
    figures_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 600,
) -> dict[str, list[str]]:
    target_dir = Path(figures_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, list[str]] = {}
    for stem, figure in (figures or {}).items():
        if figure is None:
            continue
        paths: list[str] = []
        for fmt in formats:
            target = target_dir / f"{stem}.{str(fmt).lstrip('.')}"
            figure.savefig(target, bbox_inches="tight", dpi=int(dpi))
            paths.append(str(target))
        saved[str(stem)] = paths
    return saved


def export_notebook02_results(
    *,
    session: AnalysisSession,
    output_dirs: dict[str, Path],
    tables: dict[str, pd.DataFrame | None],
    figures: dict[str, Any] | None = None,
    configs: dict[str, Any] | None = None,
    excel_exports: dict[str, Any] | None = None,
    figure_formats: tuple[str, ...] = ("png", "pdf"),
    figure_dpi: int = 600,
) -> dict[str, Any]:
    table_paths: dict[str, str] = {}
    for name, table in (tables or {}).items():
        if table is None:
            continue
        target = output_dirs["tables"] / f"{name}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(target, index=False)
        table_paths[str(name)] = str(target)
    figure_paths = save_notebook02_figures(
        figures or {},
        output_dirs["figures"],
        formats=tuple(figure_formats),
        dpi=int(figure_dpi),
    )
    config_paths: dict[str, str] = {}
    for name, payload in (configs or {}).items():
        path = write_json(output_dirs["configs"] / f"{name}.json", payload if isinstance(payload, dict) else {"value": payload})
        config_paths[str(name)] = str(path)
    manifest = {
        "workflow": "simple_quant_v2_task_notebook02",
        "output_dir": str(output_dirs["output"]),
        "tables": table_paths,
        "figures": figure_paths,
        "configs": config_paths,
        "excel_exports": excel_exports or {},
    }
    manifest_path = write_json(output_dirs["output"] / "manifest.json", manifest)
    session.annotations["notebook02_task_quant"] = {
        "output_dir": str(output_dirs["output"]),
        "tables": table_paths,
        "figures": figure_paths,
        "configs": config_paths,
        "excel_exports": excel_exports or {},
    }
    session.set_stage("simple_quant_v2")
    checkpoint_path = session.save_pickle(output_dirs["session"] / "02_simple_quant_session.pkl")
    manifest["manifest"] = str(manifest_path)
    manifest["session_checkpoint"] = str(checkpoint_path)
    return manifest


def _clear_preprocess_results(session: AnalysisSession, channel_names: list[str | None]) -> None:
    for channel_name in channel_names:
        if channel_name is None or channel_name not in session.list_channels():
            continue
        session.set_channel_state(channel_name, preprocess_result={})
