from __future__ import annotations

from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

from .simple_quant_plotting import plot_roi_outlines_on_image
from .styles import apply_publication_style


def _show_image(ax, image: np.ndarray | None, *, show_axes: bool = False) -> None:
    if image is not None:
        arr = np.asarray(image)
        if arr.ndim != 2:
            raise ValueError("image must be a 2D array for disk intensity plotting.")
        ax.imshow(arr, cmap="gray", origin="upper")
        ax.set_xlim(0, arr.shape[1])
        ax.set_ylim(arr.shape[0], 0)
    ax.set_aspect("equal")
    if show_axes:
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)


def _finite_xy(points: pd.DataFrame | None) -> pd.DataFrame:
    if points is None or points.empty or "x_px" not in points.columns or "y_px" not in points.columns:
        return pd.DataFrame(columns=[] if points is None else points.columns)
    data = points.copy()
    data["x_px"] = pd.to_numeric(data["x_px"], errors="coerce")
    data["y_px"] = pd.to_numeric(data["y_px"], errors="coerce")
    return data.loc[np.isfinite(data["x_px"]) & np.isfinite(data["y_px"])].copy()


def _class_label(row: pd.Series) -> str:
    name = row.get("class_name", pd.NA)
    if pd.notna(name) and str(name):
        return str(name)
    class_id = row.get("class_id", pd.NA)
    return f"class_{int(class_id)}" if pd.notna(class_id) else "class_unknown"


def _class_color_map(data: pd.DataFrame, palette_name: str = "tab10") -> dict[str, str]:
    labels = [_class_label(row) for _, row in data.iterrows()]
    palette = plt.get_cmap(palette_name)
    color_map: dict[str, str] = {}
    for label, (_, row) in zip(labels, data.iterrows(), strict=False):
        value = row.get("class_color", pd.NA)
        if isinstance(value, str) and mcolors.is_color_like(value):
            color_map.setdefault(label, str(value))
    for index, label in enumerate(sorted(set(labels))):
        color_map.setdefault(label, mcolors.to_hex(palette(index % palette.N)))
    return color_map


def _edgecolors_for_class(data: pd.DataFrame) -> list[str]:
    color_map = _class_color_map(data)
    return [color_map.get(_class_label(row), "white") for _, row in data.iterrows()]


def _draw_side_panel(side_ax, data: pd.DataFrame, metric: str) -> None:
    side_ax.axis("off")
    y = 0.98

    def text(line: str, *, weight: str = "normal") -> None:
        nonlocal y
        side_ax.text(0.0, y, line, transform=side_ax.transAxes, fontsize=8, va="top", weight=weight)
        y -= 0.055

    text("Disk intensity", weight="bold")
    text(f"points: {len(data)}")
    if "coordinate_source" in data.columns and data["coordinate_source"].notna().any():
        text(f"source: {data['coordinate_source'].dropna().astype(str).iloc[0]}")
    if "disk_radius_px" in data.columns and data["disk_radius_px"].notna().any():
        radius = float(pd.to_numeric(data["disk_radius_px"], errors="coerce").dropna().iloc[0])
        text(f"radius: {radius:g} px")
    if metric in data.columns:
        values = pd.to_numeric(data[metric], errors="coerce").dropna()
        if len(values):
            text(f"median: {values.median():.4g}")
            text(f"min/max: {values.min():.4g} / {values.max():.4g}")
    if "class_name" in data.columns:
        counts = data["class_name"].fillna("class_unknown").astype(str).value_counts()
        if len(counts):
            y -= 0.025
            text("Classes", weight="bold")
            for label, count in counts.items():
                text(f"{label}: {int(count)}")


def plot_disk_aperture_preview(
    image,
    points,
    *,
    disk_radius_px,
    max_points=20,
    random_seed=0,
    show_axes=False,
    title=None,
):
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    _show_image(ax, image, show_axes=show_axes)
    data = _finite_xy(points)
    if not data.empty:
        if len(data) > int(max_points):
            rng = np.random.default_rng(int(random_seed))
            selected_index = rng.choice(data.index.to_numpy(), size=int(max_points), replace=False)
            data = data.loc[selected_index].sort_index()
        ax.scatter(data["x_px"], data["y_px"], s=12, c="#00a5cf", edgecolors="white", linewidths=0.4, zorder=3)
        for _, row in data.iterrows():
            circle = Circle(
                (float(row["x_px"]), float(row["y_px"])),
                float(disk_radius_px),
                fill=False,
                edgecolor="#ff9f1c",
                linewidth=0.9,
                alpha=0.9,
                zorder=4,
            )
            ax.add_patch(circle)
    if title:
        ax.set_title(title)
    return fig, ax


def plot_disk_intensity_map(
    image,
    intensity_table,
    *,
    metric="disk_intensity_sum",
    cmap="viridis",
    point_size=32,
    show_colorbar=True,
    show_axes=False,
    title=None,
    show_side_panel=True,
    show_roi_outlines=True,
    rois=None,
    edgecolor_mode="class",
    fixed_edgecolor="white",
):
    apply_publication_style()
    if show_side_panel:
        fig, (image_ax, side_ax) = plt.subplots(
            1,
            2,
            figsize=(7.4, 5.8),
            gridspec_kw={"width_ratios": [1.0, 0.28]},
            constrained_layout=True,
        )
    else:
        fig, image_ax = plt.subplots(figsize=(6.0, 6.0))
        side_ax = None
    _show_image(image_ax, image, show_axes=show_axes)
    data = _finite_xy(intensity_table)
    if metric not in data.columns:
        raise ValueError(f"Metric column {metric!r} is not present in intensity_table.")
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.loc[data[metric].notna()].copy()
    if show_roi_outlines:
        plot_roi_outlines_on_image(image_ax, rois if rois is not None else intensity_table, label_mode="outside")
    if not data.empty:
        if edgecolor_mode == "class":
            edgecolors: Any = _edgecolors_for_class(data)
            linewidths = 0.55
        elif edgecolor_mode == "fixed":
            edgecolors = fixed_edgecolor
            linewidths = 0.45
        elif edgecolor_mode == "none":
            edgecolors = "none"
            linewidths = 0.0
        else:
            raise ValueError("edgecolor_mode must be 'class', 'fixed', or 'none'.")
        scatter = image_ax.scatter(
            data["x_px"],
            data["y_px"],
            c=data[metric],
            s=float(point_size),
            cmap=cmap,
            edgecolors=edgecolors,
            linewidths=linewidths,
            zorder=4,
        )
        if show_colorbar:
            colorbar = fig.colorbar(scatter, ax=image_ax, fraction=0.046, pad=0.04)
            colorbar.set_label(metric)
    if side_ax is not None:
        _draw_side_panel(side_ax, data, metric)
    if title:
        image_ax.set_title(title)
    return (fig, image_ax, side_ax) if show_side_panel else (fig, image_ax)


def plot_disk_intensity_histogram(
    intensity_table,
    *,
    metric="disk_intensity_sum",
    bins=30,
    group_by_class=True,
    group_by_roi=False,
    title=None,
):
    apply_publication_style()
    data = intensity_table.copy() if intensity_table is not None else pd.DataFrame()
    if metric not in data.columns:
        raise ValueError(f"Metric column {metric!r} is not present in intensity_table.")
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.loc[data[metric].notna()].copy()

    group_columns: list[str] = []
    if group_by_roi and "roi_id" in data.columns:
        group_columns.append("roi_id")
    if group_by_class and "class_name" in data.columns:
        group_columns.append("class_name")

    if group_columns and not data.empty:
        groups = [(key, group.copy()) for key, group in data.groupby(group_columns, dropna=False, sort=True)]
    else:
        groups = [("all", data)]

    n_groups = max(1, len(groups))
    fig, axes = plt.subplots(n_groups, 1, figsize=(5.8, max(3.2, 2.3 * n_groups)), squeeze=False)
    axes_flat = axes.ravel()
    for ax, (key, group) in zip(axes_flat, groups, strict=False):
        values = pd.to_numeric(group[metric], errors="coerce").dropna()
        label = key
        if isinstance(key, tuple):
            label = " / ".join(str(value) for value in key)
        color = "#4c78a8"
        if "class_color" in group.columns and group["class_color"].notna().any():
            candidate = group["class_color"].dropna().astype(str).iloc[0]
            if mcolors.is_color_like(candidate):
                color = candidate
        ax.hist(values, bins=int(bins), color=color, alpha=0.82, edgecolor="white", linewidth=0.5)
        ax.set_ylabel("counts")
        ax.set_title(str(label))
    axes_flat[-1].set_xlabel(metric)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes_flat if n_groups > 1 else axes_flat[0]
