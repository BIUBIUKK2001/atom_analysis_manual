from __future__ import annotations

from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon, Rectangle
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .styles import apply_publication_style


def _prepare_axes(ax=None, figsize: tuple[float, float] = (5.5, 5.5)):
    apply_publication_style()
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def create_overlay_figure_with_side_panel(
    figsize: tuple[float, float] = (7.5, 6.0),
    side_width_ratio: float = 0.28,
):
    apply_publication_style()
    fig, (image_ax, side_ax) = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.0, float(side_width_ratio)]},
        constrained_layout=True,
    )
    side_ax.axis("off")
    side_ax.set_xlim(0, 1)
    side_ax.set_ylim(0, 1)
    side_ax._simple_quant_y = 0.98  # type: ignore[attr-defined]
    return fig, image_ax, side_ax


def _show_image(
    ax,
    image: np.ndarray | None,
    show_axes: bool = False,
    axis_label_mode: str = "none",
) -> None:
    if image is not None:
        ax.imshow(image, cmap="gray", origin="upper")
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
    ax.set_aspect("equal")
    if show_axes:
        if axis_label_mode == "pixel":
            ax.set_xlabel("x (px)")
            ax.set_ylabel("y (px)")
        elif axis_label_mode == "ab_projected":
            ax.set_xlabel("a projection (px)")
            ax.set_ylabel("b projection (px)")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)


def _finite_xy(table: pd.DataFrame, x_column: str, y_column: str) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame()
    data = table.copy()
    data[x_column] = pd.to_numeric(data[x_column], errors="coerce")
    data[y_column] = pd.to_numeric(data[y_column], errors="coerce")
    return data.loc[data[x_column].notna() & data[y_column].notna()].copy()


def _estimate_spacing_px(table: pd.DataFrame, x_column: str, y_column: str) -> float:
    data = _finite_xy(table, x_column, y_column)
    if len(data) < 2:
        return 8.0
    coords = data[[x_column, y_column]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
    if nearest.size == 0:
        return 8.0
    return float(np.median(nearest))


def _add_colorbar(fig, ax, collection, label: str) -> None:
    colorbar = fig.colorbar(collection, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(label)


def _is_color(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and mcolors.is_color_like(value)


def _class_label(row: pd.Series) -> str:
    name = row.get("class_name", pd.NA)
    if pd.notna(name) and str(name):
        return str(name)
    class_id = row.get("class_id", pd.NA)
    return f"class_{int(class_id)}" if pd.notna(class_id) else "unclassified"


def resolve_class_color_map(points: pd.DataFrame, fallback_palette: str = "tab10") -> dict:
    if points is None or points.empty:
        return {}
    data = points.copy()
    labels = [_class_label(row) for _, row in data.iterrows()]
    palette = plt.get_cmap(fallback_palette)
    color_map: dict[str, str] = {}
    for label, (_, row) in zip(labels, data.iterrows(), strict=False):
        value = row.get("class_color", pd.NA)
        if _is_color(value):
            color_map.setdefault(label, str(value))
    missing = [label for label in sorted(set(labels)) if label not in color_map]
    for index, label in enumerate(missing):
        if label == "unclassified":
            color_map[label] = "#8a8a8a"
        else:
            color_map[label] = mcolors.to_hex(palette(index % palette.N))
    return color_map


def plot_class_colored_atoms(
    ax,
    points: pd.DataFrame,
    *,
    point_size: float = 18.0,
    alpha: float = 0.85,
    edgecolor: str = "black",
    linewidth: float = 0.25,
    fallback_palette: str = "tab10",
    zorder: int = 3,
):
    data = _finite_xy(points, "x_px", "y_px")
    if data.empty:
        return None, {}
    color_map = resolve_class_color_map(data, fallback_palette=fallback_palette)
    labels = [_class_label(row) for _, row in data.iterrows()]
    colors = [color_map.get(label, "#8a8a8a") for label in labels]
    scatter = ax.scatter(
        data["x_px"],
        data["y_px"],
        s=point_size,
        c=colors,
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    return scatter, color_map


def _next_side_y(side_ax, amount: float = 0.08) -> float:
    y = float(getattr(side_ax, "_simple_quant_y", 0.98))
    side_ax._simple_quant_y = y - amount  # type: ignore[attr-defined]
    return y


def _draw_side_header(side_ax, title: str) -> None:
    y = _next_side_y(side_ax, 0.055)
    side_ax.text(0.0, y, title, transform=side_ax.transAxes, fontsize=9, weight="bold", va="top")


def draw_class_legend(side_ax, class_color_map: dict, title: str = "Classes") -> None:
    if side_ax is None or not class_color_map:
        return
    _draw_side_header(side_ax, title)
    for label, color in class_color_map.items():
        y = _next_side_y(side_ax, 0.045)
        side_ax.scatter([0.03], [y - 0.012], s=24, color=color, edgecolors="black", linewidths=0.25, transform=side_ax.transAxes)
        side_ax.text(0.10, y, str(label), transform=side_ax.transAxes, fontsize=8, va="top")


def draw_roi_legend(side_ax, roi_table_or_points: pd.DataFrame, title: str = "ROIs") -> None:
    if side_ax is None or roi_table_or_points is None or roi_table_or_points.empty or "roi_id" not in roi_table_or_points:
        return
    data = roi_table_or_points[["roi_id", "roi_name", "roi_color"]].drop_duplicates() if "roi_name" in roi_table_or_points else roi_table_or_points[["roi_id"]].drop_duplicates()
    _draw_side_header(side_ax, title)
    for _, row in data.iterrows():
        color = row.get("roi_color", "#ff9f1c") if "roi_color" in row.index else "#ff9f1c"
        if not _is_color(color):
            color = "#ff9f1c"
        label = row.get("roi_name", row.get("roi_id", "ROI"))
        y = _next_side_y(side_ax, 0.045)
        side_ax.plot([0.0, 0.06], [y - 0.015, y - 0.015], color=color, linewidth=3, transform=side_ax.transAxes)
        side_ax.text(0.10, y, str(label), transform=side_ax.transAxes, fontsize=8, va="top")


def draw_task_legend(side_ax, task_color_map: dict, title: str = "Tasks") -> None:
    if side_ax is None or not task_color_map:
        return
    _draw_side_header(side_ax, title)
    for label, color in task_color_map.items():
        y = _next_side_y(side_ax, 0.045)
        side_ax.plot([0.0, 0.06], [y - 0.015, y - 0.015], color=color, linewidth=3, transform=side_ax.transAxes)
        side_ax.text(0.10, y, str(label), transform=side_ax.transAxes, fontsize=8, va="top")


def draw_text_summary(side_ax, lines: list[str] | tuple[str, ...]) -> None:
    if side_ax is None or not lines:
        return
    _draw_side_header(side_ax, "Summary")
    for line in lines:
        y = _next_side_y(side_ax, 0.040)
        side_ax.text(0.0, y, str(line), transform=side_ax.transAxes, fontsize=8, va="top")


def _categorical_color_map(values: pd.Series, palette_name: str = "tab10") -> dict[str, str]:
    labels = [str(value) for value in values.dropna().astype(str).unique()]
    palette = plt.get_cmap(palette_name)
    return {label: mcolors.to_hex(palette(index % palette.N)) for index, label in enumerate(labels)}


def plot_atom_id_overlay(
    image: np.ndarray | None,
    quant_points: pd.DataFrame,
    label_column: str = "atom_id",
    max_labels: int = 300,
    class_color: bool = True,
    *,
    ax=None,
    title: str | None = None,
    point_size: float = 14.0,
    default_color: str = "#00a5cf",
):
    fig, ax = _prepare_axes(ax)
    _show_image(ax, image)
    points = _finite_xy(quant_points, "x_px", "y_px")
    if not points.empty:
        if class_color:
            plot_class_colored_atoms(ax, points, point_size=point_size)
        else:
            ax.scatter(points["x_px"], points["y_px"], s=point_size, c=default_color, edgecolors="black", linewidths=0.25)
        if label_column in points.columns and len(points) <= int(max_labels):
            for _, row in points.iterrows():
                ax.text(float(row["x_px"]) + 1.0, float(row["y_px"]) - 1.0, str(row[label_column]), fontsize=7)
    if title:
        ax.set_title(title)
    return fig, ax


def plot_direction_overlay(
    image: np.ndarray | None,
    quant_points: pd.DataFrame,
    direction_table: pd.DataFrame,
    *,
    ax=None,
    title: str | None = None,
    point_size: float = 10.0,
    arrow_length_px: float | None = None,
):
    fig, ax = _prepare_axes(ax)
    _show_image(ax, image)
    points = _finite_xy(quant_points, "x_px", "y_px")
    if not points.empty:
        plot_class_colored_atoms(ax, points, point_size=point_size, alpha=0.75)
        center_x = float(points["x_px"].median())
        center_y = float(points["y_px"].median())
    else:
        center_x = center_y = 0.0
    if arrow_length_px is None:
        if image is not None:
            arrow_length_px = 0.18 * float(min(image.shape[:2]))
        else:
            arrow_length_px = max(_estimate_spacing_px(points, "x_px", "y_px") * 3.0, 10.0)
    for _, row in direction_table.iterrows():
        dx = float(row["ux"]) * float(arrow_length_px)
        dy = float(row["uy"]) * float(arrow_length_px)
        ax.arrow(center_x, center_y, dx, dy, width=0.8, head_width=5.0, length_includes_head=True, color="#f18f01")
        ax.text(center_x + dx, center_y + dy, str(row["direction_name"]), color="#f18f01", weight="bold")
    if title:
        ax.set_title(title)
    return fig, ax


def plot_scalar_blocks_on_image(
    image: np.ndarray | None,
    table: pd.DataFrame,
    x_column: str,
    y_column: str,
    value_column: str,
    *,
    ax=None,
    title: str | None = None,
    block_size_px: float | None = None,
    block_fraction: float = 0.55,
    cmap: str = "viridis",
    alpha: float = 0.75,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
):
    fig, ax = _prepare_axes(ax)
    _show_image(ax, image)
    data = _finite_xy(table, x_column, y_column)
    if value_column not in data.columns:
        if title:
            ax.set_title(title)
        return fig, ax
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data = data.loc[data[value_column].notna()].copy()
    if block_size_px is None:
        block_size_px = _estimate_spacing_px(data, x_column, y_column) * float(block_fraction)
    patches = [
        Rectangle((float(row[x_column]) - block_size_px / 2.0, float(row[y_column]) - block_size_px / 2.0), block_size_px, block_size_px)
        for _, row in data.iterrows()
    ]
    if patches:
        collection = PatchCollection(patches, cmap=cmap, alpha=alpha, edgecolor="none")
        collection.set_array(data[value_column].to_numpy(dtype=float))
        if vmin is not None or vmax is not None:
            collection.set_clim(vmin, vmax)
        ax.add_collection(collection)
        _add_colorbar(fig, ax, collection, colorbar_label or value_column)
    if title:
        ax.set_title(title)
    return fig, ax


def _segment_polygon(x1: float, y1: float, x2: float, y2: float, width: float) -> Polygon | None:
    dx = float(x2) - float(x1)
    dy = float(y2) - float(y1)
    length = float(np.hypot(dx, dy))
    if not np.isfinite(length) or length <= 0.0:
        return None
    nx = -dy / length * width / 2.0
    ny = dx / length * width / 2.0
    vertices = np.asarray(
        [
            [x1 + nx, y1 + ny],
            [x2 + nx, y2 + ny],
            [x2 - nx, y2 - ny],
            [x1 - nx, y1 - ny],
        ],
        dtype=float,
    )
    return Polygon(vertices, closed=True)


def plot_pair_blocks_on_image(
    image: np.ndarray | None,
    pair_table: pd.DataFrame,
    value_column: str,
    *,
    ax=None,
    title: str | None = None,
    block_width_px: float | None = None,
    width_fraction: float = 0.35,
    cmap: str = "viridis",
    alpha: float = 0.75,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
):
    fig, ax = _prepare_axes(ax)
    _show_image(ax, image)
    data = pair_table.copy() if pair_table is not None else pd.DataFrame()
    required = ["source_x_px", "source_y_px", "target_x_px", "target_y_px", value_column]
    if data.empty or any(column not in data.columns for column in required):
        if title:
            ax.set_title(title)
        return fig, ax
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=required)
    if block_width_px is None:
        distances = np.hypot(data["target_x_px"] - data["source_x_px"], data["target_y_px"] - data["source_y_px"])
        distances = distances[np.isfinite(distances) & (distances > 0.0)]
        block_width_px = float(np.median(distances)) * float(width_fraction) if len(distances) else 4.0
    patches = []
    for _, row in data.iterrows():
        patch = _segment_polygon(
            float(row["source_x_px"]),
            float(row["source_y_px"]),
            float(row["target_x_px"]),
            float(row["target_y_px"]),
            float(block_width_px),
        )
        if patch is not None:
            patches.append(patch)
    if patches:
        collection = PatchCollection(patches, cmap=cmap, alpha=alpha, edgecolor="none")
        collection.set_array(data[value_column].to_numpy(dtype=float)[: len(patches)])
        if vmin is not None or vmax is not None:
            collection.set_clim(vmin, vmax)
        ax.add_collection(collection)
        _add_colorbar(fig, ax, collection, colorbar_label or value_column)
    if title:
        ax.set_title(title)
    return fig, ax


def plot_line_spacing_blocks_on_image(
    image: np.ndarray | None,
    line_spacing_table: pd.DataFrame,
    value_column: str = "spacing_to_next_pm",
    *,
    ax=None,
    title: str | None = None,
    block_width_px: float | None = None,
    width_fraction: float = 0.35,
    cmap: str = "viridis",
    alpha: float = 0.75,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
):
    table = line_spacing_table.copy() if line_spacing_table is not None else pd.DataFrame()
    if not table.empty:
        if "next_atom_id" not in table.columns:
            table = table.iloc[0:0].copy()
        else:
            table = table.loc[table["next_atom_id"].notna()].copy()
        table = table.rename(
            columns={
                "x_px": "source_x_px",
                "y_px": "source_y_px",
                "next_x_px": "target_x_px",
                "next_y_px": "target_y_px",
            }
        )
    return plot_pair_blocks_on_image(
        image,
        table,
        value_column=value_column,
        ax=ax,
        title=title,
        block_width_px=block_width_px,
        width_fraction=width_fraction,
        cmap=cmap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        colorbar_label=colorbar_label or value_column,
    )


def plot_line_assignment_overlay(
    image: np.ndarray | None,
    line_spacing_table: pd.DataFrame,
    *,
    ax=None,
    title: str | None = None,
    point_size: float = 16.0,
):
    fig, ax = _prepare_axes(ax)
    _show_image(ax, image)
    table = _finite_xy(line_spacing_table, "x_px", "y_px")
    if not table.empty and "line_id" in table.columns:
        plot_class_colored_atoms(ax, table, point_size=point_size, alpha=0.9)
    if title:
        ax.set_title(title)
    return fig, ax


def plot_basis_glyph(
    ax,
    basis_vector_table: pd.DataFrame,
    anchor: str | tuple[float, float] = "outside",
    side_ax=None,
    names: tuple[str, ...] = ("a", "b"),
):
    if basis_vector_table is None or basis_vector_table.empty:
        return None
    if anchor == "outside" and side_ax is not None:
        _draw_side_header(side_ax, "Basis")
        origin = np.asarray([0.20, _next_side_y(side_ax, 0.18) - 0.05])
        scale = 0.14
        for index, (_, row) in enumerate(basis_vector_table.head(len(names)).iterrows()):
            angle = float(row.get("angle_deg", 0.0))
            direction = np.asarray([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
            start = origin + np.asarray([0.0, -0.10 * index])
            end = start + direction * scale
            side_ax.annotate(
                "",
                xy=end,
                xytext=start,
                xycoords=side_ax.transAxes,
                arrowprops={"arrowstyle": "->", "linewidth": 1.6, "color": "#f18f01"},
            )
            side_ax.text(end[0] + 0.02, end[1], str(row.get("basis_name", names[index])), transform=side_ax.transAxes, fontsize=8, va="center")
        return side_ax
    if isinstance(anchor, tuple):
        x0, y0 = (float(anchor[0]), float(anchor[1]))
        for _, row in basis_vector_table.iterrows():
            ax.arrow(
                x0,
                y0,
                float(row["vector_x_px"]),
                float(row["vector_y_px"]),
                width=0.8,
                head_width=5.0,
                length_includes_head=True,
                color="#f18f01",
                clip_on=False,
            )
            ax.text(
                x0 + float(row["vector_x_px"]),
                y0 + float(row["vector_y_px"]),
                str(row["basis_name"]),
                color="#f18f01",
                weight="bold",
                clip_on=False,
            )
        return ax
    return None


def _segment_colors(data: pd.DataFrame, color_by: str, fixed_color: str) -> tuple[list[str] | None, dict[str, str]]:
    if data.empty:
        return [], {}
    if color_by == "fixed":
        return [fixed_color] * len(data), {"segments": fixed_color}
    if color_by == "roi" and "roi_color" in data.columns:
        colors = []
        color_map: dict[str, str] = {}
        fallback = _categorical_color_map(data.get("roi_id", pd.Series(dtype=object)))
        for _, row in data.iterrows():
            label = str(row.get("roi_id", "ROI"))
            color = row.get("roi_color", pd.NA)
            if not _is_color(color):
                color = fallback.get(label, fixed_color)
            color_map.setdefault(label, str(color))
            colors.append(str(color))
        return colors, color_map
    if color_by == "class_pair":
        labels = (
            data.get("source_class_name", pd.Series(["source"] * len(data))).astype(str)
            + " -> "
            + data.get("target_class_name", pd.Series(["target"] * len(data))).astype(str)
        )
        color_map = _categorical_color_map(labels)
        return [color_map[str(label)] for label in labels], color_map
    values = data.get("task_name", pd.Series(["task"] * len(data))).astype(str)
    color_map = _categorical_color_map(values)
    return [color_map[str(label)] for label in values], color_map


def plot_measurement_segments_on_image(
    image: np.ndarray | None,
    points: pd.DataFrame,
    segments: pd.DataFrame,
    *,
    basis_vector_table: pd.DataFrame | None = None,
    line_guides: pd.DataFrame | None = None,
    color_by: str = "task",
    fixed_color: str = "#ff9f1c",
    value_column: str = "distance_pm",
    cmap: str = "viridis",
    show_value_colorbar: bool = False,
    linewidth: float = 3.0,
    alpha: float = 0.90,
    show_atoms: bool = True,
    atom_size: float = 18.0,
    show_side_panel: bool = True,
    show_axes: bool = False,
    title: str | None = None,
):
    if show_side_panel:
        fig, image_ax, side_ax = create_overlay_figure_with_side_panel()
    else:
        fig, image_ax = _prepare_axes(figsize=(6.0, 6.0))
        side_ax = None
    _show_image(image_ax, image, show_axes=show_axes, axis_label_mode="pixel")
    class_color_map = {}
    if show_atoms and points is not None and not points.empty:
        _, class_color_map = plot_class_colored_atoms(image_ax, points, point_size=atom_size, zorder=3)
    data = segments.copy() if segments is not None else pd.DataFrame()
    required = ["source_x_px", "source_y_px", "target_x_px", "target_y_px"]
    if not data.empty and all(column in data.columns for column in required):
        for column in required + ([value_column] if value_column in data.columns else []):
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna(subset=required)
    task_color_map: dict[str, str] = {}
    if not data.empty:
        lines = data[required].to_numpy(dtype=float).reshape(-1, 2, 2)
        if color_by == "value" and value_column in data.columns:
            collection = LineCollection(lines, cmap=cmap, linewidths=linewidth, alpha=alpha, zorder=4)
            collection.set_array(pd.to_numeric(data[value_column], errors="coerce").to_numpy(dtype=float))
            image_ax.add_collection(collection)
            if show_value_colorbar:
                _add_colorbar(fig, image_ax, collection, value_column)
        else:
            colors, task_color_map = _segment_colors(data, color_by=color_by, fixed_color=fixed_color)
            collection = LineCollection(lines, colors=colors, linewidths=linewidth, alpha=alpha, zorder=4)
            try:
                collection.set_capstyle("round")
            except Exception:
                pass
            image_ax.add_collection(collection)
    if line_guides is not None and not line_guides.empty:
        guide_lines = line_guides[["line_start_x_px", "line_start_y_px", "line_end_x_px", "line_end_y_px"]].to_numpy(dtype=float).reshape(-1, 2, 2)
        image_ax.add_collection(LineCollection(guide_lines, colors="#f18f01", linewidths=1.2, alpha=0.55, zorder=2))
    if basis_vector_table is not None:
        plot_basis_glyph(image_ax, basis_vector_table, side_ax=side_ax)
    if side_ax is not None:
        draw_class_legend(side_ax, class_color_map)
        draw_roi_legend(side_ax, points)
        draw_task_legend(side_ax, task_color_map)
        draw_text_summary(
            side_ax,
            [
                f"points: {0 if points is None else len(points)}",
                f"segments: {0 if segments is None else len(segments)}",
                f"color_by: {color_by}",
            ],
        )
    if title:
        image_ax.set_title(title)
    return fig, image_ax, side_ax


def plot_line_guides_on_image(
    image: np.ndarray | None,
    points: pd.DataFrame,
    line_guides: pd.DataFrame,
    *,
    basis_vector_table: pd.DataFrame | None = None,
    show_atoms: bool = True,
    line_color_by: str = "roi",
    line_width: float = 1.5,
    line_alpha: float = 0.7,
    label_mode: str = "all",
    label_every_n: int = 2,
    selected_line_ids: tuple[int, ...] | list[int] | set[int] | None = None,
    label_outside: bool = True,
    show_side_panel: bool = True,
    show_axes: bool = False,
    title: str | None = None,
):
    if show_side_panel:
        fig, image_ax, side_ax = create_overlay_figure_with_side_panel()
    else:
        fig, image_ax = _prepare_axes(figsize=(6.0, 6.0))
        side_ax = None
    _show_image(image_ax, image, show_axes=show_axes, axis_label_mode="pixel")
    class_color_map = {}
    if show_atoms and points is not None and not points.empty:
        _, class_color_map = plot_class_colored_atoms(image_ax, points, point_size=18.0, zorder=3)
    data = line_guides.copy() if line_guides is not None else pd.DataFrame()
    task_color_map: dict[str, str] = {}
    if not data.empty:
        required = ["line_start_x_px", "line_start_y_px", "line_end_x_px", "line_end_y_px"]
        lines = data[required].to_numpy(dtype=float).reshape(-1, 2, 2)
        if line_color_by == "roi" and "roi_color" in data.columns:
            colors, task_color_map = _segment_colors(data.rename(columns={"task_name": "line_task"}), color_by="roi", fixed_color="#f18f01")
        else:
            labels = data.get("task_name", pd.Series(["line"] * len(data))).astype(str)
            task_color_map = _categorical_color_map(labels, "tab10")
            colors = [task_color_map[str(label)] for label in labels]
        image_ax.add_collection(LineCollection(lines, colors=colors, linewidths=line_width, alpha=line_alpha, zorder=2))
        selected = {int(value) for value in selected_line_ids} if selected_line_ids is not None else None
        for position, (_, row) in enumerate(data.iterrows()):
            line_id = int(row["line_id"])
            draw_label = label_mode == "all"
            if label_mode == "every_n":
                draw_label = line_id % max(int(label_every_n), 1) == 0
            if label_mode == "selected":
                draw_label = selected is not None and line_id in selected
            if label_mode == "none":
                draw_label = False
            if draw_label:
                image_ax.text(
                    float(row["line_label_x_px"]),
                    float(row["line_label_y_px"]),
                    str(line_id),
                    fontsize=8,
                    weight="bold",
                    color=colors[position],
                    clip_on=not label_outside,
                    zorder=5,
                )
    if basis_vector_table is not None:
        plot_basis_glyph(image_ax, basis_vector_table, side_ax=side_ax)
    if side_ax is not None:
        draw_class_legend(side_ax, class_color_map)
        draw_roi_legend(side_ax, points)
        draw_task_legend(side_ax, task_color_map, title="Line guides")
        draw_text_summary(side_ax, [f"points: {0 if points is None else len(points)}", f"lines: {0 if line_guides is None else len(line_guides)}"])
    if title:
        image_ax.set_title(title)
    return fig, image_ax, side_ax


def plot_spacing_histogram(
    table: pd.DataFrame,
    value_column: str,
    group_column: str | None = None,
    *,
    ax=None,
    title: str | None = None,
    bins: int = 30,
    alpha: float = 0.65,
):
    fig, ax = _prepare_axes(ax, figsize=(5.8, 4.2))
    data = table.copy() if table is not None else pd.DataFrame()
    if value_column in data.columns:
        data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
        data = data.loc[data[value_column].notna()]
    else:
        data = pd.DataFrame()
    if not data.empty:
        if group_column and group_column in data.columns:
            for label, group in data.groupby(group_column, dropna=False):
                ax.hist(group[value_column], bins=bins, alpha=alpha, label=str(label))
            ax.legend(title=group_column, bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0)
        else:
            ax.hist(data[value_column], bins=bins, alpha=alpha, color="#00a5cf")
        mean_value = float(data[value_column].mean())
        median_value = float(data[value_column].median())
        ax.axvline(mean_value, color="#f18f01", linewidth=1.5, label="mean")
        ax.axvline(median_value, color="#2ca25f", linewidth=1.5, linestyle="--", label="median")
        if not group_column:
            ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0)
    ax.set_xlabel(value_column)
    ax.set_ylabel("count")
    if title:
        ax.set_title(title)
    return fig, ax


def plot_spacing_profile(
    table: pd.DataFrame,
    x_column: str,
    y_column: str,
    group_column: str | None = None,
    *,
    ax=None,
    title: str | None = None,
):
    fig, ax = _prepare_axes(ax, figsize=(6.0, 4.2))
    data = table.copy() if table is not None else pd.DataFrame()
    if x_column in data.columns and y_column in data.columns:
        data[x_column] = pd.to_numeric(data[x_column], errors="coerce")
        data[y_column] = pd.to_numeric(data[y_column], errors="coerce")
        data = data.loc[data[x_column].notna() & data[y_column].notna()]
    else:
        data = pd.DataFrame()
    if not data.empty:
        if group_column and group_column in data.columns:
            for label, group in data.groupby(group_column, dropna=False):
                group = group.sort_values(x_column)
                ax.plot(group[x_column], group[y_column], marker="o", linewidth=1.0, label=str(label))
            ax.legend(title=group_column, bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0)
        else:
            data = data.sort_values(x_column)
            ax.plot(data[x_column], data[y_column], marker="o", linewidth=1.0)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    if title:
        ax.set_title(title)
    return fig, ax


def plot_line_width_summary(
    line_summary: pd.DataFrame,
    *,
    ax=None,
    title: str | None = None,
    value_column: str | None = None,
):
    fig, ax = _prepare_axes(ax, figsize=(6.2, 4.2))
    data = line_summary.copy() if line_summary is not None else pd.DataFrame()
    if value_column is None:
        value_column = "line_width_pm" if "line_width_pm" in data.columns and data["line_width_pm"].notna().any() else "line_width_px"
    if not data.empty and "line_id" in data.columns and value_column in data.columns:
        data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
        data = data.loc[data[value_column].notna()].sort_values("line_id")
        ax.bar(data["line_id"].astype(str), data[value_column], color="#00a5cf", alpha=0.8)
    ax.set_xlabel("line_id")
    ax.set_ylabel(value_column)
    if title:
        ax.set_title(title)
    return fig, ax
