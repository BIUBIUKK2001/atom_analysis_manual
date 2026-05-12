from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd

from .styles import apply_publication_style


_STRAIN_COMPONENT_ALIASES = {
    "eps_xx": ("eps_xx",),
    "eps_yy": ("eps_yy",),
    "eps_xy": ("eps_xy",),
    "rotation_deg": ("rotation_deg",),
    "principal_eps_1": ("principal_eps_1",),
    "principal_eps_2": ("principal_eps_2",),
    "dilatation": ("dilatation",),
    "shear_magnitude": ("shear_magnitude",),
    "affine_residual": ("affine_residual",),
    "affine_residual_nm": ("affine_residual_nm", "affine_residual"),
    "local_a_nm": ("local_a_length_nm", "local_a_length"),
    "local_b_nm": ("local_b_length_nm", "local_b_length"),
    "local_gamma_deg": ("local_gamma_deg",),
}

_STRAIN_QC_COLORS = {
    "ok": "#2ca25f",
    "too_few_pairs": "#fdae61",
    "ill_conditioned": "#d7191c",
    "fit_failed": "#756bb1",
}


def _prepare_axes(ax=None, figsize: tuple[float, float] = (5.2, 5.2)):
    apply_publication_style()
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


_UNIT_TO_NM = {
    "nm": 1.0,
    "nanometer": 1.0,
    "nanometers": 1.0,
    "å": 0.1,
    "Å": 0.1,
    "a": 0.1,
    "angstrom": 0.1,
    "angstroms": 0.1,
    "pm": 0.001,
    "picometer": 0.001,
    "picometers": 0.001,
    "um": 1000.0,
    "µm": 1000.0,
    "μm": 1000.0,
    "micrometer": 1000.0,
    "micrometers": 1000.0,
}


def _axis_scale(
    pixel_size: float | None = None,
    unit: str | None = "px",
    target_unit: str = "nm",
) -> tuple[float, str]:
    if pixel_size is None or str(unit or "px").lower() == "px":
        return 1.0, "px"
    source_key = str(unit).strip().lower()
    target_key = str(target_unit).strip().lower()
    source_to_nm = _UNIT_TO_NM.get(source_key)
    target_to_nm = _UNIT_TO_NM.get(target_key)
    if source_to_nm is None or target_to_nm is None:
        return float(pixel_size), str(unit)
    return float(pixel_size) * source_to_nm / target_to_nm, target_unit


def _format_axis_value(value: float, scale: float, *, integer: bool = False) -> str:
    scaled = float(value) * float(scale)
    if abs(scaled) < 1e-12:
        scaled = 0.0
    if integer:
        return f"{int(np.rint(scaled))}"
    return f"{scaled:g}"


def _integer_nm_step(span_nm: float, target_ticks: int = 7) -> int:
    if not np.isfinite(span_nm) or span_nm <= 0:
        return 1
    raw_step = span_nm / max(target_ticks - 1, 1)
    if raw_step <= 1:
        return 1
    exponent = 10 ** np.floor(np.log10(raw_step))
    for multiplier in (1, 2, 5, 10):
        step = multiplier * exponent
        if step >= raw_step:
            return int(step)
    return int(10 * exponent)


def _apply_integer_nm_locator(axis, limits: tuple[float, float], scale: float) -> None:
    if not np.isfinite(scale) or scale <= 0:
        return
    span_nm = abs(float(limits[1] - limits[0]) * float(scale))
    step_nm = _integer_nm_step(span_nm)
    axis.set_major_locator(MultipleLocator(step_nm / float(scale)))


def _apply_calibrated_xy_axes(
    ax,
    *,
    pixel_size: float | None = None,
    unit: str | None = "px",
    target_unit: str = "nm",
) -> None:
    scale, display_unit = _axis_scale(pixel_size, unit, target_unit)
    integer_ticks = str(display_unit).strip().lower() == "nm"
    if integer_ticks:
        _apply_integer_nm_locator(ax.xaxis, ax.get_xlim(), scale)
        _apply_integer_nm_locator(ax.yaxis, ax.get_ylim(), scale)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: _format_axis_value(value, scale, integer=integer_ticks))
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: _format_axis_value(value, scale, integer=integer_ticks))
    )
    ax.set_xlabel(f"x ({display_unit})")
    ax.set_ylabel(f"y ({display_unit})")


def _strain_table(session) -> pd.DataFrame:
    table = getattr(session, "strain_table", pd.DataFrame())
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError("session.strain_table 为空；请先运行 compute_local_affine_strain。")
    return table.copy()


def _strain_plot_table(
    session,
    *,
    atom_role: str | None = None,
    qc_only: bool = False,
) -> pd.DataFrame:
    table = _strain_table(session)
    if qc_only and "qc_flag" in table.columns:
        table = table.loc[table["qc_flag"] == "ok"].copy()
    if atom_role is not None:
        if "role" not in table.columns:
            raise ValueError("已指定 atom_role，但 strain_table 中没有 role 列。")
        table = table.loc[table["role"].astype(str) == str(atom_role)].copy()
    if table.empty:
        raise ValueError("过滤后没有可绘制的 strain 点。")
    return table


def _strain_xy_columns(table: pd.DataFrame) -> tuple[str, str]:
    if {"x_px", "y_px"}.issubset(table.columns):
        return "x_px", "y_px"
    if {"x", "y"}.issubset(table.columns):
        return "x", "y"
    raise ValueError("strain_table 需要包含 x_px/y_px 或 x/y 坐标列。")


def _strain_background_image(session, image_source: str | None):
    if image_source is None:
        return None
    if image_source == "raw":
        return getattr(session, "raw_image", None)
    if image_source == "processed":
        try:
            return session.get_processed_image()
        except Exception:
            return None
    raise ValueError("image_source 必须是 'raw'、'processed' 或 None。")


def _resolve_strain_component(table: pd.DataFrame, component: str) -> tuple[str, str]:
    if component not in _STRAIN_COMPONENT_ALIASES:
        allowed = ", ".join(sorted(_STRAIN_COMPONENT_ALIASES))
        raise ValueError(f"不支持的 strain 组件: {component}。可用组件: {allowed}")
    for column in _STRAIN_COMPONENT_ALIASES[component]:
        if column in table.columns:
            return column, component
    aliases = ", ".join(_STRAIN_COMPONENT_ALIASES[component])
    raise ValueError(f"strain_table 中没有可用于 {component} 的列；已检查: {aliases}")


def _draw_strain_background(ax, image: np.ndarray | None) -> None:
    if image is not None:
        ax.imshow(image, cmap="gray", origin="upper")


def plot_raw_image(
    image: np.ndarray,
    title: str = "Raw image",
    ax=None,
    pixel_size: float | None = None,
    unit: str | None = "px",
    target_unit: str = "nm",
):
    fig, ax = _prepare_axes(ax=ax, figsize=(6.6, 5.4))
    ax.imshow(image, cmap="gray", origin="upper")
    ax.set_title(title)
    _apply_calibrated_xy_axes(ax, pixel_size=pixel_size, unit=unit, target_unit=target_unit)
    return fig, ax


def plot_fft_magnitude(
    spectrum: np.ndarray,
    title: str = "FFT magnitude",
    ax=None,
    cmap: str = "magma",
):
    fig, ax = _prepare_axes(ax=ax)
    ax.imshow(spectrum, cmap=cmap, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    return fig, ax


def plot_fft_aperture_mask(
    mask: np.ndarray,
    title: str = "FFT aperture mask",
    ax=None,
    cmap: str = "viridis",
):
    fig, ax = _prepare_axes(ax=ax)
    ax.imshow(mask, cmap=cmap, origin="upper", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    return fig, ax


def plot_preprocess_diagnostics(
    session,
    figsize: tuple[float, float] = (5.8, 5.8),
    overlay_point_size: float = 8.0,
    separate_figures: bool = True,
):
    calibration = getattr(session, "pixel_calibration", None)
    calibration_kwargs = {
        "pixel_size": getattr(calibration, "size", None),
        "unit": getattr(calibration, "unit", "px"),
        "target_unit": "nm",
    }
    if separate_figures:
        figures = []
        figures.append(plot_raw_image(session.raw_image, title="\u539f\u59cb\u56fe\u50cf", **calibration_kwargs))
        figures.append(plot_raw_image(session.get_processed_image(), title="\u6700\u7ec8\u5de5\u4f5c\u56fe\u50cf", **calibration_kwargs))
        figures.append(
            plot_atom_overlay(
                session.get_processed_image(),
                session.candidate_points,
                title="\u5019\u9009\u70b9\u53e0\u52a0\u7ed3\u679c",
                origin_xy=session.get_processed_origin(),
                point_size=overlay_point_size,
                **calibration_kwargs,
            )
        )
        return figures

    fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 3.0, figsize[1]))
    plot_raw_image(session.raw_image, title="\u539f\u59cb\u56fe\u50cf", ax=axes[0], **calibration_kwargs)
    plot_raw_image(session.get_processed_image(), title="\u6700\u7ec8\u5de5\u4f5c\u56fe\u50cf", ax=axes[1], **calibration_kwargs)
    plot_atom_overlay(
        session.get_processed_image(),
        session.candidate_points,
        ax=axes[2],
        title="\u5019\u9009\u70b9\u53e0\u52a0\u7ed3\u679c",
        origin_xy=session.get_processed_origin(),
        point_size=overlay_point_size,
        **calibration_kwargs,
    )
    fig.tight_layout()
    return fig, axes


def _require_napari():
    try:
        import napari
    except ImportError as exc:
        raise ImportError("napari is required for interactive preprocessing review.") from exc
    return napari


def _napari_point_data(points: pd.DataFrame) -> np.ndarray:
    if points.empty:
        return np.empty((0, 2), dtype=float)
    return np.column_stack(
        (
            points["y_px"].to_numpy(dtype=float),
            points["x_px"].to_numpy(dtype=float),
        )
    )


def _add_napari_points_layer(
    viewer,
    points: pd.DataFrame,
    *,
    name: str,
    point_size: float,
    point_color: str,
    visible: bool,
):
    points_layer = viewer.add_points(
        _napari_point_data(points),
        name=name,
        size=point_size,
        canvas_size_limits=(4, 8),
        face_color=point_color,
        border_width=0.0,
        border_width_is_relative=False,
        symbol="disc",
        visible=visible,
    )
    points_layer.editable = False
    return points_layer


def _points_for_role(points: pd.DataFrame, role: str) -> pd.DataFrame:
    if points.empty:
        return points.copy()
    if "column_role" not in points.columns:
        return points.copy() if role == "light_atom" else points.iloc[0:0].copy()
    return points.loc[points["column_role"] == role].copy()


def launch_preprocess_napari_viewer(
    session,
    point_size: float = 4.0,
    point_color: str = "#00a5cf",
    show_raw_layer: bool = False,
    show_filtered_layer: bool = False,
):
    napari = _require_napari()

    try:
        processed_image = session.get_processed_image()
        origin_x, origin_y = session.get_processed_origin()
        viewer = napari.Viewer(title=f"EM Atom Workbench - Preprocess Review - {session.name}")

        if session.raw_image is not None:
            viewer.add_image(session.raw_image, name="raw_image", visible=show_raw_layer)

        filtered_image = session.preprocess_result.get("filtered_image")
        if filtered_image is not None:
            viewer.add_image(
                filtered_image,
                name="filtered_image",
                visible=show_filtered_layer,
                translate=(origin_y, origin_x),
            )

        viewer.add_image(
            processed_image,
            name="processed_image",
            visible=True,
            translate=(origin_y, origin_x),
        )

        if not session.candidate_points.empty:
            point_data = np.column_stack(
                (
                    session.candidate_points["y_px"].to_numpy(dtype=float),
                    session.candidate_points["x_px"].to_numpy(dtype=float),
                )
            )
            points_layer = viewer.add_points(
                point_data,
                name="candidate_points",
                size=point_size,
                canvas_size_limits=(4, 8),
                face_color=point_color,
                border_width=0.0,
                border_width_is_relative=False,
                symbol="disc",
            )
            points_layer.editable = False
        return viewer
    except Exception as exc:
        raise RuntimeError(
            "Failed to open the napari preprocessing viewer. "
            "This only affects the interactive viewer; static preprocessing previews "
            "and candidate detection results are still available."
        ) from exc


def launch_detection_napari_viewer(
    session,
    show_raw_layer: bool = False,
    point_size: float = 5.0,
):
    napari = _require_napari()

    try:
        viewer = napari.Viewer(title=f"EM Atom Workbench - Detection Review - {session.name}")

        if session.workflow_mode == "hfo2_multichannel":
            workflow_settings = dict(session.workflow_settings or {})
            primary_channel = str(workflow_settings.get("primary_channel") or session.primary_channel)
            heavy_channel = str(workflow_settings.get("heavy_channel") or "haadf")
            light_channel = str(workflow_settings.get("light_channel") or primary_channel)
            confirm_channel = workflow_settings.get("confirm_channel")

            primary_state = session.get_channel_state(primary_channel)
            if primary_state.raw_image is not None:
                viewer.add_image(primary_state.raw_image, name="raw_primary_image", visible=show_raw_layer)

            for channel_name, visible in (
                (light_channel, True),
                (heavy_channel, False),
                (str(confirm_channel), False) if confirm_channel else (None, False),
            ):
                if channel_name is None:
                    continue
                origin_x, origin_y = session.get_processed_origin(channel_name)
                viewer.add_image(
                    session.get_processed_image(channel_name),
                    name=f"processed_{channel_name}",
                    visible=visible,
                    translate=(origin_y, origin_x),
                )

            heavy_points = _points_for_role(session.candidate_points, "heavy_atom")
            light_points = _points_for_role(session.candidate_points, "light_atom")

            _add_napari_points_layer(
                viewer,
                heavy_points,
                name="heavy_atom_points",
                point_size=point_size,
                point_color="#ff8c00",
                visible=True,
            )
            _add_napari_points_layer(
                viewer,
                light_points,
                name="light_atom_points",
                point_size=point_size,
                point_color="#00a5cf",
                visible=True,
            )
            return viewer

        processed_image = session.get_processed_image()
        origin_x, origin_y = session.get_processed_origin()
        if session.raw_image is not None:
            viewer.add_image(session.raw_image, name="raw_image", visible=show_raw_layer)
        viewer.add_image(
            processed_image,
            name="processed_image",
            visible=True,
            translate=(origin_y, origin_x),
        )
        _add_napari_points_layer(
            viewer,
            session.candidate_points,
            name="candidate_points",
            point_size=point_size,
            point_color="#00a5cf",
            visible=True,
        )
        return viewer
    except Exception as exc:
        raise RuntimeError(
            "Failed to open the napari detection viewer. "
            "This only affects the interactive detection overview; "
            "the session state remains unchanged."
        ) from exc


def launch_refinement_napari_viewer(
    session,
    show_raw_layer: bool = False,
    show_candidate_layer: bool = False,
    point_size: float = 5.0,
):
    napari = _require_napari()

    try:
        processed_image = session.get_processed_image()
        origin_x, origin_y = session.get_processed_origin()
        viewer = napari.Viewer(title=f"EM Atom Workbench - Refinement Review - {session.name}")

        if session.raw_image is not None:
            viewer.add_image(session.raw_image, name="raw_image", visible=show_raw_layer)

        viewer.add_image(
            processed_image,
            name="processed_image",
            visible=True,
            translate=(origin_y, origin_x),
        )

        _add_napari_points_layer(
            viewer,
            session.candidate_points,
            name="candidate_points",
            point_size=point_size,
            point_color="#00a5cf",
            visible=show_candidate_layer,
        )
        _add_napari_points_layer(
            viewer,
            session.refined_points,
            name="refined_points",
            point_size=point_size,
            point_color="#ff7a00",
            visible=True,
        )
        return viewer
    except Exception as exc:
        raise RuntimeError(
            "Failed to open the napari refinement viewer. "
            "This only affects the interactive refinement review; "
            "the refinement and curation results in the session remain unchanged."
        ) from exc


def plot_atom_overlay(
    image: np.ndarray,
    points: pd.DataFrame,
    ax=None,
    title: str = "Atom overlay",
    origin_xy: tuple[int, int] = (0, 0),
    point_size: float = 8.0,
    point_color: str = "#00a5cf",
    pixel_size: float | None = None,
    unit: str | None = "px",
    target_unit: str = "nm",
):
    fig, ax = _prepare_axes(ax=ax)
    ax.imshow(image, cmap="gray", origin="upper")
    if not points.empty:
        ax.scatter(
            points["x_px"] - origin_xy[0],
            points["y_px"] - origin_xy[1],
            s=point_size,
            c=point_color,
            marker="o",
            linewidths=0.0,
            alpha=0.95,
        )
    ax.set_title(title)
    _apply_calibrated_xy_axes(ax, pixel_size=pixel_size, unit=unit, target_unit=target_unit)
    return fig, ax


def plot_class_overlay(
    image: np.ndarray,
    points: pd.DataFrame,
    ax=None,
    title: str = "Atom-column classes",
    origin_xy: tuple[int, int] = (0, 0),
    point_size: float = 16.0,
    pixel_size: float | None = None,
    unit: str | None = "px",
    target_unit: str = "nm",
):
    fig, ax = _prepare_axes(ax=ax)
    ax.imshow(image, cmap="gray", origin="upper")
    if not points.empty:
        if "class_id" not in points.columns:
            raise ValueError("points must contain class_id for class overlay plotting.")
        classes = points.groupby("class_id", dropna=False)
        for class_id, class_points in classes:
            class_name = (
                str(class_points["class_name"].iloc[0])
                if "class_name" in class_points.columns and pd.notna(class_points["class_name"].iloc[0])
                else f"class_{class_id}"
            )
            color = (
                str(class_points["class_color"].iloc[0])
                if "class_color" in class_points.columns and pd.notna(class_points["class_color"].iloc[0])
                else "#00a5cf"
            )
            ax.scatter(
                class_points["x_px"] - origin_xy[0],
                class_points["y_px"] - origin_xy[1],
                s=point_size,
                c=color,
                marker="o",
                linewidths=0.35,
                edgecolors="white",
                alpha=0.95,
                label=class_name,
            )
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=False,
            title="class_name",
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    ax.set_title(title)
    _apply_calibrated_xy_axes(ax, pixel_size=pixel_size, unit=unit, target_unit=target_unit)
    return fig, ax


def plot_class_feature_scatter_matrix(
    features: pd.DataFrame,
    points: pd.DataFrame,
    feature_columns: tuple[str, str] | None = None,
    ax=None,
    title: str = "Class feature scatter",
):
    if features.empty:
        raise ValueError("features must not be empty.")
    if points.empty or "class_id" not in points.columns:
        raise ValueError("points must contain classified atom columns.")
    joined = features.merge(
        points[[column for column in ("atom_id", "candidate_id", "class_id", "class_name", "class_color") if column in points.columns]],
        on="atom_id" if "atom_id" in features.columns and "atom_id" in points.columns else "candidate_id",
        how="left",
    )
    numeric_columns = [
        column
        for column in features.columns
        if column not in {"atom_id", "candidate_id", "x_px", "y_px"} and pd.api.types.is_numeric_dtype(features[column])
    ]
    if len(numeric_columns) < 2:
        raise ValueError("At least two numeric feature columns are required.")
    x_column, y_column = feature_columns or (numeric_columns[0], numeric_columns[1])
    fig, ax = _prepare_axes(ax=ax, figsize=(5.4, 4.4))
    for _, class_points in joined.groupby("class_id", dropna=False):
        class_name = (
            str(class_points["class_name"].iloc[0])
            if "class_name" in class_points.columns and pd.notna(class_points["class_name"].iloc[0])
            else "unclassified"
        )
        color = (
            str(class_points["class_color"].iloc[0])
            if "class_color" in class_points.columns and pd.notna(class_points["class_color"].iloc[0])
            else "#737373"
        )
        ax.scatter(
            class_points[x_column],
            class_points[y_column],
            s=22,
            c=color,
            linewidths=0.25,
            edgecolors="white",
            alpha=0.9,
            label=class_name,
        )
    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=False,
        title="class_name",
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    return fig, ax


def plot_metric_map(
    image: np.ndarray,
    points: pd.DataFrame,
    values: Iterable[float],
    title: str,
    label: str,
    ax=None,
    origin_xy: tuple[int, int] = (0, 0),
    cmap: str = "viridis",
):
    fig, ax = _prepare_axes(ax=ax)
    ax.imshow(image, cmap="gray", origin="upper")
    scatter = ax.scatter(
        points["x_px"] - origin_xy[0],
        points["y_px"] - origin_xy[1],
        c=np.asarray(list(values), dtype=float),
        cmap=cmap,
        s=24,
        linewidths=0.0,
    )
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.82)
    colorbar.set_label(label)
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    return fig, ax


def plot_reference_candidate_map(
    session,
    suggestion,
    *,
    image_source: str | None = "raw",
    ax=None,
    title: str | None = None,
    show_candidate_vectors: bool = True,
    selected_candidate_id: int | None = None,
    vector_scale: float = 1.0,
):
    assignments = pd.DataFrame(getattr(suggestion, "assignments", pd.DataFrame())).copy()
    if assignments.empty:
        raise ValueError("reference suggestion 中没有可绘制的 candidate 分配记录。")
    if "candidate_id" not in assignments.columns:
        raise ValueError("reference suggestion assignments 需要包含 candidate_id 列。")
    if {"x_px", "y_px"}.issubset(assignments.columns):
        x_column, y_column = "x_px", "y_px"
    elif {"x", "y"}.issubset(assignments.columns):
        x_column, y_column = "x", "y"
    else:
        raise ValueError("reference suggestion assignments 需要包含 x_px/y_px 或 x/y 坐标列。")

    fig, ax = _prepare_axes(ax=ax)
    image = _strain_background_image(session, image_source)
    _draw_strain_background(ax, image)

    candidate_ids = pd.Series(assignments["candidate_id"]).astype(int)
    selected_candidate_id = None if selected_candidate_id is None else int(selected_candidate_id)
    cmap = plt.get_cmap("tab10")
    plotted = False
    ordered_candidate_ids = sorted(candidate_ids.unique())
    for index, candidate_id in enumerate(ordered_candidate_ids):
        mask = candidate_ids == candidate_id
        if not mask.any():
            continue
        is_selected = selected_candidate_id == int(candidate_id)
        ax.scatter(
            assignments.loc[mask, x_column].to_numpy(dtype=float),
            assignments.loc[mask, y_column].to_numpy(dtype=float),
            s=34 if is_selected else 24,
            c=[cmap(index % 10)],
            marker="o",
            linewidths=0.75 if is_selected else 0.35,
            edgecolors="#111111" if is_selected else "white",
            alpha=1.0 if selected_candidate_id is None or is_selected else 0.45,
            label=f"candidate {candidate_id}",
        )
        plotted = True
    if not plotted:
        raise ValueError("没有可绘制的 reference candidate 点。")

    if show_candidate_vectors:
        candidates = pd.DataFrame(getattr(suggestion, "candidates", pd.DataFrame())).copy()
        vector_columns = {"candidate_id", "basis_a_x", "basis_a_y", "basis_b_x", "basis_b_y"}
        if vector_columns.issubset(candidates.columns):
            candidate_index = {int(row["candidate_id"]): row for _, row in candidates.iterrows()}
            scale = float(vector_scale)
            for index, candidate_id in enumerate(ordered_candidate_ids):
                row = candidate_index.get(int(candidate_id))
                if row is None:
                    continue
                mask = candidate_ids == candidate_id
                anchor_x = float(np.median(assignments.loc[mask, x_column].to_numpy(dtype=float)))
                anchor_y = float(np.median(assignments.loc[mask, y_column].to_numpy(dtype=float)))
                vectors = np.array(
                    [
                        [float(row["basis_a_x"]), float(row["basis_a_y"])],
                        [float(row["basis_b_x"]), float(row["basis_b_y"])],
                    ],
                    dtype=float,
                )
                is_selected = selected_candidate_id == int(candidate_id)
                color = cmap(index % 10)
                ax.quiver(
                    [anchor_x, anchor_x],
                    [anchor_y, anchor_y],
                    vectors[:, 0] * scale,
                    vectors[:, 1] * scale,
                    color=[color, color],
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    width=0.006 if is_selected else 0.0038,
                    alpha=1.0 if selected_candidate_id is None or is_selected else 0.55,
                    zorder=5,
                )
                ax.text(
                    anchor_x + vectors[0, 0] * scale,
                    anchor_y + vectors[0, 1] * scale,
                    f"a{candidate_id}",
                    color=color,
                    fontsize=7,
                    ha="left",
                    va="bottom",
                    zorder=6,
                )
                ax.text(
                    anchor_x + vectors[1, 0] * scale,
                    anchor_y + vectors[1, 1] * scale,
                    f"b{candidate_id}",
                    color=color,
                    fontsize=7,
                    ha="left",
                    va="bottom",
                    zorder=6,
                )

    ax.set_aspect("equal", adjustable="box")
    if image is None:
        ax.invert_yaxis()
    ax.set_title(title or "Reference lattice candidates")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=False,
        title="candidate_id",
    )
    fig.tight_layout()
    return fig, ax


def plot_strain_component_map(
    session,
    component: str = "eps_xx",
    *,
    image_source: str | None = "raw",
    atom_role: str | None = None,
    qc_only: bool = True,
    scale: float | None = None,
    ax=None,
    title: str | None = None,
):
    table = _strain_plot_table(session, atom_role=atom_role, qc_only=qc_only)
    value_column, label = _resolve_strain_component(table, component)
    x_column, y_column = _strain_xy_columns(table)
    values = table[value_column].to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError(f"{component} 没有可绘制的有限数值。")
    table = table.loc[finite].copy()
    values = values[finite]
    if scale is not None:
        values = values * float(scale)

    fig, ax = _prepare_axes(ax=ax)
    image = _strain_background_image(session, image_source)
    _draw_strain_background(ax, image)
    scatter = ax.scatter(
        table[x_column].to_numpy(dtype=float),
        table[y_column].to_numpy(dtype=float),
        c=values,
        cmap="coolwarm" if component.startswith("eps_") or component in {"rotation_deg", "dilatation"} else "viridis",
        s=26,
        linewidths=0.0,
        alpha=0.95,
    )
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.82)
    colorbar.set_label(label if scale is None else f"{label} × {scale:g}")
    ax.set_aspect("equal", adjustable="box")
    if image is None:
        ax.invert_yaxis()
    ax.set_title(title or f"Local affine strain - {component}")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    return fig, ax


def plot_strain_qc_map(
    session,
    *,
    image_source: str | None = "raw",
    ax=None,
    title: str | None = None,
):
    table = _strain_table(session)
    if "qc_flag" not in table.columns:
        raise ValueError("strain_table 中没有 qc_flag 列。")
    x_column, y_column = _strain_xy_columns(table)

    fig, ax = _prepare_axes(ax=ax)
    image = _strain_background_image(session, image_source)
    _draw_strain_background(ax, image)

    flags = pd.Series(table["qc_flag"]).fillna("unknown").astype(str)
    plotted = False
    ordered_flags = [flag for flag in _STRAIN_QC_COLORS if flag in set(flags)]
    ordered_flags.extend(sorted(set(flags) - set(ordered_flags)))
    for flag in ordered_flags:
        mask = flags == flag
        if not mask.any():
            continue
        color = _STRAIN_QC_COLORS.get(flag, "#737373")
        ax.scatter(
            table.loc[mask, x_column].to_numpy(dtype=float),
            table.loc[mask, y_column].to_numpy(dtype=float),
            s=28,
            c=color,
            marker="o",
            linewidths=0.35,
            edgecolors="white",
            alpha=0.95,
            label=flag,
        )
        plotted = True
    if not plotted:
        raise ValueError("没有可绘制的 qc_flag 点。")

    ax.set_aspect("equal", adjustable="box")
    if image is None:
        ax.invert_yaxis()
    ax.set_title(title or "Local affine strain QC")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.legend(loc="best", frameon=False, title="qc_flag")
    return fig, ax


def plot_domain_annotation(
    image: np.ndarray,
    points: pd.DataFrame,
    label_column: str = "annotation_label",
    ax=None,
    title: str = "Domain annotation",
    origin_xy: tuple[int, int] = (0, 0),
):
    fig, ax = _prepare_axes(ax=ax)
    ax.imshow(image, cmap="gray", origin="upper")
    if not points.empty and label_column in points.columns:
        labels = pd.Series(points[label_column]).fillna("unlabeled")
        categories = pd.Categorical(labels)
        scatter = ax.scatter(
            points["x_px"] - origin_xy[0],
            points["y_px"] - origin_xy[1],
            c=categories.codes,
            cmap="tab10",
            s=24,
        )
        handles = []
        for code, category in enumerate(categories.categories):
            handles.append(
                plt.Line2D([], [], linestyle="", marker="o", color=scatter.cmap(scatter.norm(code)), label=str(category))
            )
        ax.legend(handles=handles, loc="upper right", frameon=False)
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    return fig, ax


def plot_vector_field(
    image: np.ndarray,
    vector_field: pd.DataFrame,
    ax=None,
    title: str = "Vector field",
    origin_xy: tuple[int, int] = (0, 0),
    color_by: str = "magnitude_px",
    cmap: str = "magma",
):
    fig, ax = _prepare_axes(ax=ax)
    ax.imshow(image, cmap="gray", origin="upper")
    if not vector_field.empty:
        colors = vector_field[color_by] if color_by in vector_field.columns else "tab:red"
        quiver = ax.quiver(
            vector_field["x_px"] - origin_xy[0],
            vector_field["y_px"] - origin_xy[1],
            vector_field["u_px"],
            vector_field["v_px"],
            colors if isinstance(colors, str) else colors.to_numpy(dtype=float),
            cmap=cmap if not isinstance(colors, str) else None,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
        )
        if not isinstance(colors, str):
            colorbar = fig.colorbar(quiver, ax=ax, shrink=0.82)
            colorbar.set_label(color_by)
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    return fig, ax


def plot_histogram_or_distribution(
    values: Iterable[float],
    bins: int = 30,
    title: str = "Distribution",
    xlabel: str = "Value",
    ax=None,
):
    fig, ax = _prepare_axes(ax=ax, figsize=(5.2, 3.8))
    cleaned = np.asarray(list(values), dtype=float)
    cleaned = cleaned[np.isfinite(cleaned)]
    ax.hist(cleaned, bins=bins, color="#4c6faf", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    return fig, ax


def add_scale_bar(
    ax,
    pixel_size: float | None,
    length_phys: float | None = None,
    unit: str = "px",
    color: str = "white",
    linewidth: float = 2.5,
    pad_px: float = 10.0,
):
    if pixel_size is None:
        return ax
    if length_phys is None:
        length_phys = 5 * pixel_size
    length_px = float(length_phys / pixel_size)
    x0 = ax.get_xlim()[1] - pad_px - length_px
    x1 = ax.get_xlim()[1] - pad_px
    y = ax.get_ylim()[0] + pad_px
    ax.plot([x0, x1], [y, y], color=color, linewidth=linewidth, solid_capstyle="butt")
    ax.text((x0 + x1) / 2, y - pad_px * 0.3, f"{length_phys:g} {unit}", color=color, ha="center", va="top")
    return ax


def plot_neighbor_graph(
    image: np.ndarray,
    points: pd.DataFrame,
    edges: pd.DataFrame,
    ax=None,
    title: str = "Neighbor graph",
    origin_xy: tuple[int, int] = (0, 0),
):
    fig, ax = _prepare_axes(ax=ax)
    ax.imshow(image, cmap="gray", origin="upper")
    if not points.empty:
        ax.scatter(points["x_px"] - origin_xy[0], points["y_px"] - origin_xy[1], s=12, c="#f18f01")
    if not edges.empty:
        id_lookup = points.set_index("atom_id")[["x_px", "y_px"]]
        segments = []
        for _, row in edges.iterrows():
            p0 = id_lookup.loc[row["source_atom_id"]]
            p1 = id_lookup.loc[row["target_atom_id"]]
            segments.append(
                [
                    (p0["x_px"] - origin_xy[0], p0["y_px"] - origin_xy[1]),
                    (p1["x_px"] - origin_xy[0], p1["y_px"] - origin_xy[1]),
                ]
            )
        collection = LineCollection(segments, colors="#2f4858", linewidths=0.7, alpha=0.8)
        ax.add_collection(collection)
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    return fig, ax


def save_figure_multi_format(fig, base_path: str | Path, formats: Iterable[str]) -> list[Path]:
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for fmt in formats:
        target = base.with_suffix(f".{fmt}")
        fig.savefig(target, bbox_inches="tight")
        saved_paths.append(target)
    return saved_paths
