from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
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


def _show_image(ax, image: np.ndarray | None) -> None:
    if image is not None:
        ax.imshow(image, cmap="gray", origin="upper")
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
    ax.set_aspect("equal")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")


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
        if class_color and "class_color" in points.columns:
            colors = [value if isinstance(value, str) and value else default_color for value in points["class_color"]]
        else:
            colors = default_color
        ax.scatter(points["x_px"], points["y_px"], s=point_size, c=colors, edgecolors="black", linewidths=0.25)
        if label_column in points.columns and len(points) <= int(max_labels):
            for _, row in points.iterrows():
                ax.text(float(row["x_px"]) + 1.0, float(row["y_px"]) - 1.0, str(row[label_column]), fontsize=7)
        elif len(points) > int(max_labels):
            ax.text(0.02, 0.02, f"{len(points)} points; labels hidden", transform=ax.transAxes, fontsize=8)
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
        ax.scatter(points["x_px"], points["y_px"], s=point_size, c="#00a5cf", alpha=0.75)
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
        scatter = ax.scatter(table["x_px"], table["y_px"], c=table["line_id"], cmap="tab20", s=point_size, alpha=0.9)
        _add_colorbar(fig, ax, scatter, "line_id")
        centers = table.groupby("line_id", dropna=False)[["x_px", "y_px"]].mean().reset_index()
        for _, row in centers.iterrows():
            ax.text(float(row["x_px"]), float(row["y_px"]), str(int(row["line_id"])), fontsize=8, weight="bold")
    if title:
        ax.set_title(title)
    return fig, ax


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
            ax.legend(title=group_column)
        else:
            ax.hist(data[value_column], bins=bins, alpha=alpha, color="#00a5cf")
        mean_value = float(data[value_column].mean())
        median_value = float(data[value_column].median())
        ax.axvline(mean_value, color="#f18f01", linewidth=1.5, label="mean")
        ax.axvline(median_value, color="#2ca25f", linewidth=1.5, linestyle="--", label="median")
        if not group_column:
            ax.legend()
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
            ax.legend(title=group_column)
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
