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



def _validate_stack_for_plot(stack) -> np.ndarray:
    arr = np.asarray(stack)
    if arr.ndim != 3:
        raise ValueError("stack must be a 3D array with shape (n_slices, height, width).")
    return arr


def _slice_filtered_table(table: pd.DataFrame | None, slice_index: int | None) -> pd.DataFrame:
    data = table.copy() if table is not None else pd.DataFrame()
    if slice_index is None:
        return data
    if "slice_index" not in data.columns:
        raise ValueError("intensity_table must contain slice_index.")
    slice_values = pd.to_numeric(data["slice_index"], errors="coerce")
    return data.loc[slice_values == int(slice_index)].copy()


def _color_for_group(group: pd.DataFrame, label: str, color_map: dict[str, str]) -> str:
    if "class_color" in group.columns and group["class_color"].notna().any():
        candidate = group["class_color"].dropna().astype(str).iloc[0]
        if mcolors.is_color_like(candidate):
            return candidate
    return color_map.get(str(label), "#4c78a8")


def plot_stack_intensity_profiles(
    summary_table,
    *,
    metric: str = "disk_intensity_mean",
    x: str = "slice_index",
    group_by: str = "class_name",
    stat: str = "mean",
    error: str | None = "std",
    ax=None,
    title: str | None = None,
):
    """Plot per-slice intensity profiles, one curve per class/species."""

    apply_publication_style()
    data = summary_table.copy() if summary_table is not None else pd.DataFrame()
    if "metric" in data.columns:
        data = data.loc[data["metric"].astype(str) == str(metric)].copy()
    for column in (x, group_by, stat):
        if column not in data.columns:
            raise ValueError(f"Column {column!r} is not present in summary_table.")
    if error is not None and error not in data.columns:
        raise ValueError(f"Error column {error!r} is not present in summary_table.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
    else:
        fig = ax.figure
    color_map = _class_color_map(data) if not data.empty else {}
    for label, group in data.groupby(group_by, dropna=False, sort=True):
        group = group.sort_values(x)
        xs = pd.to_numeric(group[x], errors="coerce")
        ys = pd.to_numeric(group[stat], errors="coerce")
        yerr = pd.to_numeric(group[error], errors="coerce") if error else None
        color = _color_for_group(group, str(label), color_map)
        ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=1.3, markersize=3.5, capsize=2.0, label=str(label), color=color)
    ax.set_xlabel(x)
    ax.set_ylabel(f"{metric} {stat}")
    if title:
        ax.set_title(title)
    if not data.empty:
        ax.legend(frameon=False, fontsize=8)
    return fig, ax


def plot_stack_slice_intensity_map(
    stack,
    intensity_table,
    *,
    slice_index: int,
    metric: str = "disk_intensity_mean",
    cmap: str = "viridis",
    point_size: float = 32,
    show_colorbar: bool = True,
    show_axes: bool = False,
    title: str | None = None,
    edgecolor_mode: str = "class",
    fixed_edgecolor: str = "white",
):
    """Show one stack slice with per-atom intensity overlay."""

    apply_publication_style()
    arr = _validate_stack_for_plot(stack)
    idx = int(slice_index)
    if idx < 0 or idx >= arr.shape[0]:
        raise ValueError(f"slice_index {slice_index} is out of range for stack with {arr.shape[0]} slices.")
    data = _slice_filtered_table(intensity_table, idx)
    if metric not in data.columns:
        raise ValueError(f"Metric column {metric!r} is not present in intensity_table.")
    data = _finite_xy(data)
    data[metric] = pd.to_numeric(data[metric], errors="coerce") if metric in data.columns else np.nan
    data = data.loc[data[metric].notna()].copy()

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    _show_image(ax, arr[idx], show_axes=show_axes)
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
        scatter = ax.scatter(
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
            colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            colorbar.set_label(metric)
    if title:
        ax.set_title(title)
    return fig, ax


def plot_stack_intensity_histogram(
    intensity_table,
    *,
    metric: str = "disk_intensity_mean",
    slice_index: int | None = None,
    bins: int = 30,
    group_by_class: bool = True,
    title: str | None = None,
):
    """Plot stack intensity histograms, optionally for a selected slice."""

    apply_publication_style()
    data = _slice_filtered_table(intensity_table, slice_index)
    if metric not in data.columns:
        raise ValueError(f"Metric column {metric!r} is not present in intensity_table.")
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.loc[data[metric].notna()].copy()
    if group_by_class and "class_name" in data.columns and not data.empty:
        groups = [(key, group.copy()) for key, group in data.groupby("class_name", dropna=False, sort=True)]
    else:
        groups = [("all", data)]

    n_groups = max(1, len(groups))
    fig, axes = plt.subplots(n_groups, 1, figsize=(5.8, max(3.2, 2.3 * n_groups)), squeeze=False)
    axes_flat = axes.ravel()
    color_map = _class_color_map(data) if not data.empty else {}
    for ax, (key, group) in zip(axes_flat, groups, strict=False):
        values = pd.to_numeric(group[metric], errors="coerce").dropna()
        color = _color_for_group(group, str(key), color_map)
        ax.hist(values, bins=int(bins), color=color, alpha=0.82, edgecolor="white", linewidth=0.5)
        ax.set_ylabel("counts")
        ax.set_title(str(key))
    axes_flat[-1].set_xlabel(metric)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes_flat if n_groups > 1 else axes_flat[0]


def plot_stack_refinement_shift_profile(
    stack_refined_points,
    *,
    group_by: str = "class_name",
    stat: str = "median",
    error: str | None = "q25_q75",
    ax=None,
    title: str | None = None,
):
    """Plot center_shift_px versus slice_index for refined stack coordinates."""

    apply_publication_style()
    data = stack_refined_points.copy() if stack_refined_points is not None else pd.DataFrame()
    for column in ("slice_index", "center_shift_px", group_by):
        if column not in data.columns:
            raise ValueError(f"Column {column!r} is not present in stack_refined_points.")
    data["center_shift_px"] = pd.to_numeric(data["center_shift_px"], errors="coerce")
    grouped = (
        data.groupby(["slice_index", group_by], dropna=False, sort=True)["center_shift_px"]
        .agg(
            mean="mean",
            std="std",
            sem="sem",
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    if stat not in grouped.columns:
        raise ValueError(f"stat must be one of {sorted(set(grouped.columns) - {'slice_index', group_by})}.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
    else:
        fig = ax.figure
    color_map = _class_color_map(data) if not data.empty else {}
    for label, group in grouped.groupby(group_by, dropna=False, sort=True):
        group = group.sort_values("slice_index")
        xs = pd.to_numeric(group["slice_index"], errors="coerce")
        ys = pd.to_numeric(group[stat], errors="coerce")
        color = color_map.get(str(label), None)
        ax.plot(xs, ys, marker="o", linewidth=1.3, markersize=3.5, label=str(label), color=color)
        if error == "q25_q75":
            q25 = pd.to_numeric(group["q25"], errors="coerce")
            q75 = pd.to_numeric(group["q75"], errors="coerce")
            ax.fill_between(xs, q25, q75, alpha=0.18, color=color)
        elif error is not None:
            if error not in group.columns:
                raise ValueError(f"Error column {error!r} is not available.")
            yerr = pd.to_numeric(group[error], errors="coerce")
            ax.errorbar(xs, ys, yerr=yerr, fmt="none", capsize=2.0, color=color)
    ax.set_xlabel("slice_index")
    ax.set_ylabel(f"center_shift_px {stat}")
    if title:
        ax.set_title(title)
    if not data.empty:
        ax.legend(frameon=False, fontsize=8)
    return fig, ax


def _napari_properties(points: pd.DataFrame, columns: list[str]) -> dict[str, list[Any]]:
    properties: dict[str, list[Any]] = {}
    for column in columns:
        if column in points.columns:
            properties[column] = points[column].astype("object").where(points[column].notna(), None).tolist()
    return properties


def launch_stack_refinement_napari_viewer(
    stack: np.ndarray,
    refined_points: pd.DataFrame,
    *,
    seed_points: pd.DataFrame | None = None,
    intensity_table: pd.DataFrame | None = None,
    slice_index: int = 0,
    point_size: float = 5.0,
    show_seed_points: bool = True,
    show_refined_points: bool = True,
    color_by: str = "class_name",
    image_name: str = "stack_slice",
):
    """Launch an optional napari viewer for one refined stack slice."""

    try:
        import napari
    except ImportError as exc:  # pragma: no cover - depends on optional GUI install.
        raise ImportError("napari is required for stack refinement review. Install em-atom-workbench[interactive] or install napari.") from exc

    arr = _validate_stack_for_plot(stack)
    idx = int(slice_index)
    if idx < 0 or idx >= arr.shape[0]:
        raise ValueError(f"slice_index {slice_index} is out of range for stack with {arr.shape[0]} slices.")
    viewer = napari.Viewer()
    viewer.add_image(arr[idx], name=image_name, colormap="gray")

    refined = _slice_filtered_table(refined_points, idx)
    if intensity_table is not None and not refined.empty:
        intensity_slice = _slice_filtered_table(intensity_table, idx)
        merge_keys = [key for key in ("point_id", "atom_id", "slice_index") if key in refined.columns and key in intensity_slice.columns]
        if merge_keys and "disk_intensity_mean" in intensity_slice.columns:
            refined = refined.merge(
                intensity_slice[merge_keys + ["disk_intensity_mean"]].drop_duplicates(merge_keys),
                on=merge_keys,
                how="left",
            )

    def add_point_layers(data: pd.DataFrame, *, prefix: str, face_fallback: str, opacity: float) -> None:
        data = _finite_xy(data)
        if data.empty:
            return
        layer_groups = data.groupby(color_by, dropna=False, sort=True) if color_by in data.columns else [("all", data)]
        for label, group in layer_groups:
            coords = group[["y_px", "x_px"]].to_numpy(dtype=float)
            color = face_fallback
            if "class_color" in group.columns and group["class_color"].notna().any():
                candidate = group["class_color"].dropna().astype(str).iloc[0]
                if mcolors.is_color_like(candidate):
                    color = candidate
            properties = _napari_properties(
                group,
                [
                    "atom_id",
                    "class_id",
                    "class_name",
                    "center_shift_px",
                    "center_shift_rejected",
                    "quality_score",
                    "refinement_path",
                    "disk_intensity_mean",
                ],
            )
            viewer.add_points(
                coords,
                name=f"{prefix}_{label}",
                size=float(point_size),
                face_color=color,
                edge_color="white",
                opacity=float(opacity),
                properties=properties,
            )

    if show_seed_points and seed_points is not None:
        add_point_layers(seed_points.copy(), prefix="seed", face_fallback="#00a5cf", opacity=0.55)
    if show_refined_points:
        add_point_layers(refined, prefix="refined", face_fallback="#ff9f1c", opacity=0.95)
    return viewer
