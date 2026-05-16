from __future__ import annotations

from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .styles import FigureStyleConfig, apply_publication_style, coerce_figure_style


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


def _coerce_polygon(value: Any):
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.ndarray):
        polygon = value.astype(float)
    elif isinstance(value, (list, tuple)):
        polygon = np.asarray(value, dtype=float)
    else:
        return None
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        return None
    return polygon


def _roi_outline_records(rois_or_points) -> list[dict[str, Any]]:
    if rois_or_points is None:
        return []
    if isinstance(rois_or_points, pd.DataFrame):
        if rois_or_points.empty or "roi_id" not in rois_or_points.columns:
            return []
        records = []
        columns = [column for column in ("roi_id", "roi_name", "roi_color", "polygon_xy_px") if column in rois_or_points.columns]
        for _, row in rois_or_points[columns].drop_duplicates("roi_id").iterrows():
            records.append(
                {
                    "roi_id": row.get("roi_id", "global"),
                    "roi_name": row.get("roi_name", row.get("roi_id", "global")),
                    "roi_color": row.get("roi_color", "#ff9f1c"),
                    "polygon_xy_px": row.get("polygon_xy_px", None),
                }
            )
        return records
    records = []
    for roi in rois_or_points:
        records.append(
            {
                "roi_id": getattr(roi, "roi_id", "global"),
                "roi_name": getattr(roi, "roi_name", None) or getattr(roi, "roi_id", "global"),
                "roi_color": getattr(roi, "color", "#ff9f1c"),
                "polygon_xy_px": getattr(roi, "polygon_xy_px", None),
                "enabled": getattr(roi, "enabled", True),
            }
        )
    return records


def plot_roi_outlines_on_image(
    ax,
    rois_or_points,
    *,
    show_roi_labels: bool = True,
    label_mode: str = "outside",
    linewidth: float = 1.8,
    alpha: float = 0.95,
    zorder: int = 5,
):
    artists = []
    for record in _roi_outline_records(rois_or_points):
        if not bool(record.get("enabled", True)):
            continue
        if str(record.get("roi_id", "")) == "global":
            continue
        polygon = _coerce_polygon(record.get("polygon_xy_px"))
        if polygon is None:
            continue
        color = record.get("roi_color", "#ff9f1c")
        if not _is_color(color):
            color = "#ff9f1c"
        patch = Polygon(polygon, closed=True, fill=False, edgecolor=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
        ax.add_patch(patch)
        artists.append(patch)
        if not show_roi_labels or label_mode == "none":
            continue
        label = str(record.get("roi_name", record.get("roi_id", "ROI")))
        x_min, y_min = np.nanmin(polygon, axis=0)
        x_max, y_max = np.nanmax(polygon, axis=0)
        if label_mode == "inside":
            x_label = float((x_min + x_max) / 2.0)
            y_label = float((y_min + y_max) / 2.0)
            clip_on = True
        else:
            margin = max(4.0, 0.02 * max(abs(float(x_max - x_min)), abs(float(y_max - y_min)), 1.0))
            x_label = float(x_min)
            y_label = float(y_min - margin)
            clip_on = False
        text = ax.text(
            x_label,
            y_label,
            label,
            color=color,
            fontsize=8,
            weight="bold",
            clip_on=clip_on,
            zorder=zorder + 1,
        )
        artists.append(text)
    return artists


def summarize_rois_and_points(analysis_points: pd.DataFrame, rois=None) -> pd.DataFrame:
    if analysis_points is None or analysis_points.empty or "roi_id" not in analysis_points.columns:
        rows = []
        for record in _roi_outline_records(rois):
            rows.append(
                {
                    "roi_id": record.get("roi_id", "global"),
                    "roi_name": record.get("roi_name", record.get("roi_id", "global")),
                    "point_rows": 0,
                    "unique_point_count": 0,
                    "unique_atom_count": 0,
                    "point_set_count": 0,
                    "class_count": 0,
                }
            )
        return pd.DataFrame(rows)
    rows: list[dict[str, Any]] = []
    for roi_id, group in analysis_points.groupby("roi_id", dropna=False, sort=False):
        rows.append(
            {
                "roi_id": roi_id,
                "roi_name": group["roi_name"].iloc[0] if "roi_name" in group.columns and len(group) else roi_id,
                "point_rows": int(len(group)),
                "unique_point_count": int(group["point_id"].nunique(dropna=True)) if "point_id" in group else int(len(group)),
                "unique_atom_count": int(group["atom_id"].nunique(dropna=True)) if "atom_id" in group else 0,
                "point_set_count": int(group["point_set"].nunique(dropna=True)) if "point_set" in group else 0,
                "class_count": int(group["class_id"].nunique(dropna=True)) if "class_id" in group else 0,
            }
        )
    return pd.DataFrame(rows)


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
        for _, row in basis_vector_table.iterrows():
            length = row.get("length_px", np.nan)
            period = row.get("period_px", np.nan)
            angle = row.get("angle_deg", np.nan)
            line = f"{row.get('basis_name', '')}: L={float(length):.2f}px  P={float(period):.2f}px  angle={float(angle):.1f}deg"
            y = _next_side_y(side_ax, 0.045)
            side_ax.text(0.0, y, line, transform=side_ax.transAxes, fontsize=7.5, va="top")
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


def _finite_value(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def plot_basis_vectors_on_image(
    ax,
    basis_vector_table: pd.DataFrame,
    *,
    show_labels: bool = True,
    show_period: bool = True,
    linewidth: float = 2.2,
    color: str = "#f18f01",
    label_mode: str = "outside",
    zorder: int = 6,
):
    if basis_vector_table is None or basis_vector_table.empty:
        return []
    artists = []
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_center = float(np.mean(xlim))
    y_center = float(np.mean(ylim))
    x_span = abs(float(xlim[1] - xlim[0]))
    y_span = abs(float(ylim[1] - ylim[0]))
    for _, row in basis_vector_table.iterrows():
        if _finite_value(row.get("from_x1_px")) and _finite_value(row.get("from_y1_px")):
            x1 = float(row["from_x1_px"])
            y1 = float(row["from_y1_px"])
            if _finite_value(row.get("from_x2_px")) and _finite_value(row.get("from_y2_px")):
                x2 = float(row["from_x2_px"])
                y2 = float(row["from_y2_px"])
            else:
                x2 = x1 + float(row["vector_x_px"])
                y2 = y1 + float(row["vector_y_px"])
        else:
            x1 = x_center
            y1 = y_center
            x2 = x1 + float(row["vector_x_px"])
            y2 = y1 + float(row["vector_y_px"])
        arrow = ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "linewidth": linewidth, "color": color},
            clip_on=False,
            zorder=zorder,
        )
        artists.append(arrow)
        if not show_labels or label_mode == "none":
            continue
        label = str(row.get("basis_name", "basis"))
        if show_period:
            label += f"  L={float(row.get('length_px', np.nan)):.2f}px  P={float(row.get('period_px', np.nan)):.2f}px"
        if label_mode == "near_arrow":
            x_label = x2 + 0.01 * x_span
            y_label = y2 - 0.01 * y_span
            clip_on = False
        else:
            x_label = xlim[1] + 0.02 * x_span if (x2 - x1) >= 0 else xlim[0] - 0.22 * x_span
            y_label = y2
            clip_on = False
        text = ax.text(x_label, y_label, label, color=color, fontsize=8, weight="bold", clip_on=clip_on, zorder=zorder + 1)
        artists.append(text)
    return artists


def _basis_role_from_row(row: pd.Series) -> str:
    value = row.get("basis_role", pd.NA)
    if pd.notna(value) and str(value):
        return str(value)
    name = str(row.get("basis_name", ""))
    if "_" in name:
        return name.rsplit("_", 1)[-1]
    return name


def _roi_color_lookup(rois, points: pd.DataFrame | None = None) -> dict[str, str]:
    records = _roi_outline_records(rois if rois is not None else points)
    color_map: dict[str, str] = {}
    for record in records:
        roi_id = str(record.get("roi_id", "global"))
        color = record.get("roi_color", "#f18f01")
        color_map[roi_id] = str(color) if _is_color(color) else "#f18f01"
    return color_map


def _basis_arrow_color(row: pd.Series, roi_color_map: dict[str, str], fallback_index: int) -> str:
    roi_id = row.get("roi_id", pd.NA)
    if pd.notna(roi_id) and str(roi_id) in roi_color_map:
        return roi_color_map[str(roi_id)]
    palette = plt.get_cmap("tab10")
    return mcolors.to_hex(palette(fallback_index % palette.N))


def _basis_arrow_endpoints(row: pd.Series, ax) -> tuple[float, float, float, float] | None:
    if _finite_value(row.get("from_x1_px")) and _finite_value(row.get("from_y1_px")):
        x1 = float(row["from_x1_px"])
        y1 = float(row["from_y1_px"])
        if _finite_value(row.get("from_x2_px")) and _finite_value(row.get("from_y2_px")):
            return x1, y1, float(row["from_x2_px"]), float(row["from_y2_px"])
        if _finite_value(row.get("vector_x_px")) and _finite_value(row.get("vector_y_px")):
            return x1, y1, x1 + float(row["vector_x_px"]), y1 + float(row["vector_y_px"])
    if _finite_value(row.get("vector_x_px")) and _finite_value(row.get("vector_y_px")):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x1 = float(np.mean(xlim))
        y1 = float(np.mean(ylim))
        return x1, y1, x1 + float(row["vector_x_px"]), y1 + float(row["vector_y_px"])
    return None


def _draw_compact_basis_check_legend(
    legend_ax,
    class_color_map: dict,
    roi_color_map: dict[str, str],
    *,
    fontsize: float = 7.0,
) -> None:
    if legend_ax is None:
        return
    legend_ax.axis("off")
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    y = 0.98

    def header(text: str) -> None:
        nonlocal y
        legend_ax.text(0.0, y, text, transform=legend_ax.transAxes, fontsize=fontsize + 1.0, weight="bold", va="top")
        y -= 0.070

    def item_marker(label: str, color: str, marker: str = "point") -> None:
        nonlocal y
        if y < 0.04:
            return
        if marker == "line":
            legend_ax.plot([0.0, 0.12], [y - 0.012, y - 0.012], color=color, linewidth=2.4, transform=legend_ax.transAxes)
        else:
            legend_ax.scatter([0.035], [y - 0.016], s=18, color=color, edgecolors="black", linewidths=0.2, transform=legend_ax.transAxes)
        legend_ax.text(0.16, y, str(label), transform=legend_ax.transAxes, fontsize=fontsize, va="top")
        y -= 0.055

    if class_color_map:
        header("Classes")
        for label, color in class_color_map.items():
            item_marker(str(label), color, "point")
        y -= 0.030
    if roi_color_map:
        header("ROIs / basis")
        for roi_id, color in roi_color_map.items():
            if roi_id == "global":
                continue
            item_marker(roi_id, color, "line")


def plot_basis_check_on_image(
    image: np.ndarray | None,
    points: pd.DataFrame,
    basis_vector_table: pd.DataFrame,
    *,
    rois=None,
    show_atoms: bool = True,
    atom_size: float = 18.0,
    atom_alpha: float = 0.85,
    show_roi_outlines: bool = True,
    show_roi_labels: bool = False,
    roi_linewidth: float = 1.8,
    roi_alpha: float = 0.95,
    show_basis_labels: bool = False,
    show_basis_table: bool = False,
    basis_linewidth: float = 2.4,
    basis_alpha: float = 0.95,
    basis_mutation_scale: float = 11.0,
    show_legend: bool = True,
    legend_width_ratio: float = 0.16,
    figsize: tuple[float, float] = (7.6, 5.6),
    show_axes: bool = False,
    title: str | None = "ROI and basis vector check",
):
    """Draw a clean ROI/class/basis check overlay for notebook previews."""
    apply_publication_style()
    if show_legend:
        fig, (image_ax, legend_ax) = plt.subplots(
            1,
            2,
            figsize=figsize,
            gridspec_kw={"width_ratios": [1.0, float(legend_width_ratio)]},
            constrained_layout=True,
        )
        legend_ax.axis("off")
    else:
        fig, image_ax = plt.subplots(figsize=(figsize[0] * 0.82, figsize[1]), constrained_layout=True)
        legend_ax = None

    _show_image(image_ax, image, show_axes=show_axes, axis_label_mode="pixel")
    if show_roi_outlines:
        plot_roi_outlines_on_image(
            image_ax,
            rois if rois is not None else points,
            show_roi_labels=show_roi_labels,
            label_mode="outside" if show_roi_labels else "none",
            linewidth=roi_linewidth,
            alpha=roi_alpha,
            zorder=5,
        )

    class_color_map: dict = {}
    if show_atoms and points is not None and not points.empty:
        _, class_color_map = plot_class_colored_atoms(
            image_ax,
            points,
            point_size=atom_size,
            alpha=atom_alpha,
            zorder=3,
        )

    roi_color_map = _roi_color_lookup(rois, points)
    artists = []
    if basis_vector_table is not None and not basis_vector_table.empty:
        for index, (_, row) in enumerate(basis_vector_table.iterrows()):
            endpoints = _basis_arrow_endpoints(row, image_ax)
            if endpoints is None:
                continue
            x1, y1, x2, y2 = endpoints
            color = _basis_arrow_color(row, roi_color_map, index)
            role = _basis_role_from_row(row).lower()
            linestyle = "--" if role == "b" else "-"
            arrow = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=basis_mutation_scale,
                linewidth=basis_linewidth,
                linestyle=linestyle,
                color=color,
                alpha=basis_alpha,
                shrinkA=0.0,
                shrinkB=0.0,
                zorder=7,
            )
            image_ax.add_patch(arrow)
            artists.append(arrow)
            if show_basis_labels:
                image_ax.text(
                    x2,
                    y2,
                    str(row.get("basis_name", "basis")),
                    color=color,
                    fontsize=7,
                    weight="bold",
                    va="bottom",
                    ha="left",
                    clip_on=False,
                    zorder=8,
                )

    if legend_ax is not None:
        _draw_compact_basis_check_legend(legend_ax, class_color_map, roi_color_map)
        if show_basis_table and basis_vector_table is not None and not basis_vector_table.empty:
            y0 = 0.02
            legend_ax.text(
                0.0,
                y0,
                f"basis: {len(basis_vector_table)}",
                transform=legend_ax.transAxes,
                fontsize=7,
                va="bottom",
            )
    if title:
        image_ax.set_title(title)
    return fig, image_ax, legend_ax


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
    rois=None,
    show_roi_outlines: bool = True,
    roi_label_mode: str = "outside",
    roi_linewidth: float = 1.8,
    roi_alpha: float = 0.95,
    show_basis_vectors: bool = True,
    basis_label_mode: str = "outside",
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
    if show_roi_outlines:
        plot_roi_outlines_on_image(
            image_ax,
            rois if rois is not None else points,
            label_mode=roi_label_mode,
            linewidth=roi_linewidth,
            alpha=roi_alpha,
        )
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
        if show_basis_vectors:
            plot_basis_vectors_on_image(image_ax, basis_vector_table, label_mode=basis_label_mode)
        plot_basis_glyph(image_ax, basis_vector_table, side_ax=side_ax)
    if side_ax is not None:
        draw_class_legend(side_ax, class_color_map)
        draw_roi_legend(side_ax, points)
        draw_task_legend(side_ax, task_color_map)
        unique_points = int(points["point_id"].nunique(dropna=True)) if points is not None and "point_id" in points else 0
        unique_atoms = int(points["atom_id"].nunique(dropna=True)) if points is not None and "atom_id" in points else 0
        draw_text_summary(
            side_ax,
            [
                f"ROI-expanded rows: {0 if points is None else len(points)}",
                f"unique points: {unique_points}",
                f"unique atoms: {unique_atoms}",
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
    rois=None,
    show_roi_outlines: bool = True,
    roi_label_mode: str = "outside",
    show_basis_vectors: bool = True,
    basis_label_mode: str = "outside",
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
    if show_roi_outlines:
        plot_roi_outlines_on_image(image_ax, rois if rois is not None else points, label_mode=roi_label_mode)
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
        if show_basis_vectors:
            plot_basis_vectors_on_image(image_ax, basis_vector_table, label_mode=basis_label_mode)
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


def _gaussian_curve(x_values: np.ndarray, mean: float, std: float, count: int, bin_width: float) -> np.ndarray:
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0 or count <= 0:
        return np.full_like(x_values, np.nan, dtype=float)
    coefficient = float(count) * float(bin_width) / (std * np.sqrt(2.0 * np.pi))
    return coefficient * np.exp(-0.5 * ((x_values - mean) / std) ** 2)


def _plot_gaussian_histogram(
    table: pd.DataFrame,
    value_column: str,
    *,
    group_columns: tuple[str, ...],
    xlabel: str,
    title_prefix: str,
    bins: int = 24,
    style: FigureStyleConfig | dict | None = None,
) -> dict[str, Any]:
    from .simple_quant import fit_single_gaussian_to_histogram

    config = coerce_figure_style(style)
    apply_publication_style(config)
    figures: dict[str, Any] = {}
    data = table.copy() if table is not None else pd.DataFrame()
    if data.empty or value_column not in data.columns:
        return figures
    if "valid" in data.columns:
        data = data.loc[data["valid"].astype(bool)].copy()
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data = data.loc[data[value_column].notna()]
    if data.empty:
        return figures
    for keys, group in data.groupby(list(group_columns), dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group[value_column].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        stats = fit_single_gaussian_to_histogram(values)
        fig, ax = plt.subplots(figsize=(4.3, 3.2), constrained_layout=True)
        counts, edges, _ = ax.hist(
            values,
            bins=min(int(bins), max(5, int(np.sqrt(values.size) * 2))),
            color=config.histogram_color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.82,
        )
        if len(edges) > 1 and np.isfinite(stats["std"]) and stats["std"] > 0:
            x_values = np.linspace(float(edges[0]), float(edges[-1]), 300)
            fit_y = _gaussian_curve(x_values, float(stats["mean"]), float(stats["std"]), int(stats["n"]), float(np.mean(np.diff(edges))))
            ax.plot(x_values, fit_y, color=config.fit_color, linewidth=config.line_width + 0.5, label="Gaussian fit")
        median = stats["median"]
        robust = stats["robust_std"]
        if np.isfinite(median):
            ax.axvline(float(median), color="#2ca25f", linewidth=config.line_width, linestyle="--", label="median")
        annotation = f"median = {median:.3g}\nrobust std = {robust:.3g}" if np.isfinite(robust) else f"median = {median:.3g}"
        ax.text(0.98, 0.95, annotation, transform=ax.transAxes, ha="right", va="top", fontsize=config.legend_size)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
        ax.set_title(title_prefix)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="best")
        stem = "_".join(str(value).replace(":", "_").replace("+", "_").replace(" ", "_") for value in keys)
        figures[f"{title_prefix}_{stem}_{value_column}"] = fig
    return figures


def plot_period_length_histograms(
    period_segment_table: pd.DataFrame,
    *,
    bins: int = 24,
    style: FigureStyleConfig | dict | None = None,
) -> dict[str, Any]:
    value_column = "length_A" if period_segment_table is not None and "length_A" in period_segment_table.columns and period_segment_table["length_A"].notna().any() else "length_px"
    xlabel = "Period length (Å)" if value_column == "length_A" else "Period length (px)"
    return _plot_gaussian_histogram(
        period_segment_table,
        value_column,
        group_columns=("roi_id", "direction", "class_selection"),
        xlabel=xlabel,
        title_prefix="task1A_period_length",
        bins=bins,
        style=style,
    )


def plot_period_angle_delta_histograms(
    period_segment_table: pd.DataFrame,
    *,
    bins: int = 24,
    style: FigureStyleConfig | dict | None = None,
) -> dict[str, Any]:
    return _plot_gaussian_histogram(
        period_segment_table,
        "angle_delta_deg",
        group_columns=("roi_id", "direction", "class_selection"),
        xlabel="Angle deviation (°)",
        title_prefix="task1A_angle_delta",
        bins=bins,
        style=style,
    )


def plot_projection_spacing_histogram(
    pair_line_table: pd.DataFrame,
    *,
    style: FigureStyleConfig | dict | None = None,
):
    config = coerce_figure_style(style)
    apply_publication_style(config)
    fig, ax = plt.subplots(figsize=(4.5, 3.2), constrained_layout=True)
    data = pair_line_table.copy() if pair_line_table is not None else pd.DataFrame()
    if not data.empty and "projection_s_px" in data.columns:
        for roi_id, group in data.groupby("roi_id", dropna=False, sort=False):
            s = pd.to_numeric(group["projection_s_px"], errors="coerce").dropna().sort_values().to_numpy(dtype=float)
            gaps = np.diff(s)
            gaps = gaps[np.isfinite(gaps) & (gaps > 1e-9)]
            if gaps.size:
                ax.hist(gaps, bins=min(24, max(5, int(np.sqrt(gaps.size) * 2))), alpha=0.65, label=str(roi_id))
    ax.set_xlabel("Adjacent projection spacing (px)")
    ax.set_ylabel("Counts")
    ax.set_title("Task 2 projection spacing QC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ax.has_data():
        ax.legend(frameon=False)
    return fig, ax


def plot_pair_line_distance_errorbar(
    pair_line_summary_table: pd.DataFrame,
    pair_line_table: pd.DataFrame | None = None,
    *,
    style: FigureStyleConfig | dict | None = None,
    title: str = "Task 2 pair distance by line",
):
    config = coerce_figure_style(style)
    apply_publication_style(config)
    fig, ax = plt.subplots(figsize=(4.8, 3.4), constrained_layout=True)
    summary = pair_line_summary_table.copy() if pair_line_summary_table is not None else pd.DataFrame()
    if not summary.empty:
        value_column = "distance_median_A" if summary["distance_median_A"].notna().any() else "distance_median_px"
        q1_column = "distance_q1_A" if value_column.endswith("_A") else "distance_q1_px"
        q3_column = "distance_q3_A" if value_column.endswith("_A") else "distance_q3_px"
        ylabel = "Pair distance (Å)" if value_column.endswith("_A") else "Pair distance (px)"
        for roi_id, group in summary.groupby("roi_id", dropna=False, sort=False):
            group = group.sort_values("line_id")
            x = pd.to_numeric(group["line_id"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(group[value_column], errors="coerce").to_numpy(dtype=float)
            yerr = np.vstack(
                [
                    y - pd.to_numeric(group[q1_column], errors="coerce").to_numpy(dtype=float),
                    pd.to_numeric(group[q3_column], errors="coerce").to_numpy(dtype=float) - y,
                ]
            )
            ax.errorbar(x, y, yerr=yerr, marker="o", markersize=4, linewidth=config.line_width, capsize=3, label=str(roi_id))
        if pair_line_table is not None and not pair_line_table.empty:
            raw = pair_line_table.loc[pair_line_table.get("line_valid", False).astype(bool)].copy()
            if not raw.empty and "line_id" in raw.columns:
                raw_value = "distance_A" if "distance_A" in raw.columns and raw["distance_A"].notna().any() else "distance_px"
                ax.scatter(
                    pd.to_numeric(raw["line_id"], errors="coerce"),
                    pd.to_numeric(raw[raw_value], errors="coerce"),
                    s=8,
                    color="0.25",
                    alpha=0.20,
                    zorder=1,
                )
        ax.set_ylabel(ylabel)
    ax.set_xlabel("Line index")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if summary is not None and not summary.empty:
        ax.legend(frameon=False)
    return fig, ax


def plot_pair_overlay(
    image: np.ndarray | None,
    pair_table: pd.DataFrame,
    *,
    rois=None,
    style: FigureStyleConfig | dict | None = None,
    title: str = "Task 2 pair overlay",
):
    config = coerce_figure_style(style)
    fig, ax = _prepare_axes(figsize=(5.5, 5.5))
    apply_publication_style(config)
    _show_image(ax, image)
    plot_roi_outlines_on_image(ax, rois, label_mode="none")
    data = pair_table.copy() if pair_table is not None else pd.DataFrame()
    if not data.empty:
        required = ["p1_x", "p1_y", "p2_x", "p2_y", "center_x", "center_y"]
        data = data.dropna(subset=[col for col in required if col in data.columns])
        if all(col in data.columns for col in required):
            lines = data[["p1_x", "p1_y", "p2_x", "p2_y"]].to_numpy(dtype=float).reshape(-1, 2, 2)
            colors = np.where(data.get("valid", True).astype(bool), "#4c78a8", "#bdbdbd")
            ax.add_collection(LineCollection(lines, colors=colors, linewidths=config.line_width, alpha=0.75, zorder=4))
            ax.scatter(data["center_x"], data["center_y"], s=config.marker_size, c="#d95f02", edgecolors="white", linewidths=0.25, zorder=5)
    ax.set_title(title)
    return fig, ax


def plot_pair_center_line_assignment(
    image: np.ndarray | None,
    pair_line_table: pd.DataFrame,
    *,
    rois=None,
    style: FigureStyleConfig | dict | None = None,
    title: str = "Task 2 pair-center line assignment",
):
    config = coerce_figure_style(style)
    fig, ax = _prepare_axes(figsize=(5.5, 5.5))
    apply_publication_style(config)
    _show_image(ax, image)
    plot_roi_outlines_on_image(ax, rois, label_mode="none")
    data = pair_line_table.copy() if pair_line_table is not None else pd.DataFrame()
    if not data.empty and {"center_x", "center_y", "line_id"}.issubset(data.columns):
        valid = data.loc[data.get("line_valid", False).astype(bool)].copy()
        if not valid.empty:
            labels = valid["line_id"].astype(str)
            color_map = _categorical_color_map(labels, "tab20")
            colors = [color_map[str(label)] for label in labels]
            ax.scatter(valid["center_x"], valid["center_y"], s=config.marker_size, c=colors, edgecolors="white", linewidths=0.25, zorder=5)
    ax.set_title(title)
    return fig, ax


def plot_polygon_cell_map(
    image: np.ndarray | None,
    cell_table: pd.DataFrame,
    value_column: str,
    *,
    rois=None,
    style: FigureStyleConfig | dict | None = None,
    title: str | None = None,
    cmap: str | None = None,
    alpha: float | None = None,
    edgecolor: str = "none",
    symmetric: bool | None = None,
):
    config = coerce_figure_style(style)
    apply_publication_style(config)
    fig, ax = _prepare_axes(figsize=(5.5, 5.5))
    _show_image(ax, image)
    plot_roi_outlines_on_image(ax, rois, label_mode="none", linewidth=1.0, alpha=0.7)
    data = cell_table.copy() if cell_table is not None else pd.DataFrame()
    if not data.empty and value_column in data.columns:
        data = data.loc[data.get("valid", False).astype(bool)].copy()
        data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
        data = data.dropna(subset=[value_column])
        patches = []
        for _, row in data.iterrows():
            vertices = np.asarray(
                [
                    [row["p00_x"], row["p00_y"]],
                    [row["p10_x"], row["p10_y"]],
                    [row["p11_x"], row["p11_y"]],
                    [row["p01_x"], row["p01_y"]],
                ],
                dtype=float,
            )
            patches.append(Polygon(vertices, closed=True))
        if patches:
            values = data[value_column].to_numpy(dtype=float)[: len(patches)]
            if symmetric is None:
                symmetric = str(value_column).startswith("eps_")
            if cmap is None:
                cmap = "coolwarm" if symmetric else "viridis"
            collection = PatchCollection(patches, cmap=cmap, alpha=config.overlay_alpha if alpha is None else alpha, edgecolor=edgecolor, linewidth=0.25)
            collection.set_array(values)
            if symmetric:
                vmax = float(np.nanmax(np.abs(values))) if values.size else 1.0
                collection.set_clim(-vmax, vmax)
            ax.add_collection(collection)
            label = value_column
            if str(value_column).startswith("eps_"):
                label = f"{value_column} (%)"
                collection.set_array(values * 100.0)
                if symmetric:
                    vmax = float(np.nanmax(np.abs(values * 100.0))) if values.size else 1.0
                    collection.set_clim(-vmax, vmax)
            _add_colorbar(fig, ax, collection, label)
    ax.set_title(title or value_column)
    return fig, ax


def plot_group_centers_and_displacements(
    image: np.ndarray | None,
    group_centroid_table: pd.DataFrame,
    group_displacement_table: pd.DataFrame,
    *,
    rois=None,
    style: FigureStyleConfig | dict | None = None,
    title: str = "Task 3 group centers and displacements",
):
    config = coerce_figure_style(style)
    apply_publication_style(config)
    fig, ax = _prepare_axes(figsize=(5.5, 5.5))
    _show_image(ax, image)
    plot_roi_outlines_on_image(ax, rois, label_mode="outside", linewidth=1.0, alpha=0.75)
    centers = group_centroid_table.copy() if group_centroid_table is not None else pd.DataFrame()
    if not centers.empty:
        centers = centers.loc[centers.get("valid", False).astype(bool)].copy()
        if not centers.empty:
            color_map = _categorical_color_map(centers["group_name"].astype(str), "tab10")
            for group_name, group in centers.groupby("group_name", sort=False):
                ax.scatter(group["center_x"], group["center_y"], s=config.marker_size * 2.0, c=color_map[str(group_name)], edgecolors="white", linewidths=0.5, label=str(group_name), zorder=5)
    displacements = group_displacement_table.copy() if group_displacement_table is not None else pd.DataFrame()
    if not displacements.empty:
        for _, row in displacements.loc[displacements.get("valid", False).astype(bool)].iterrows():
            ax.annotate(
                "",
                xy=(float(row["center_B_x"]), float(row["center_B_y"])),
                xytext=(float(row["center_A_x"]), float(row["center_A_y"])),
                arrowprops={"arrowstyle": "->", "linewidth": config.line_width, "color": "#d95f02"},
                zorder=6,
            )
    if centers is not None and not centers.empty:
        ax.legend(frameon=False, loc="best")
    ax.set_title(title)
    return fig, ax
