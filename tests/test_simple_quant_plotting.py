from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import pytest

from em_atom_workbench.simple_quant import AnalysisROI, BasisVectorSpec, resolve_basis_vector_specs
from em_atom_workbench.simple_quant_plotting import (
    add_nm_scalebar,
    build_period_histogram_title_table,
    plot_cropped_group_centers_and_displacements,
    plot_basis_check_on_image,
    plot_basis_vectors_on_image,
    plot_pair_line_distance_errorbar,
    plot_projection_spacing_histogram,
    plot_period_angle_delta_histograms,
    plot_period_length_histograms,
    plot_polygon_cell_map,
    plot_line_guides_on_image,
    plot_measurement_segments_on_image,
    plot_roi_outlines_on_image,
)


def _points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "point_id": ["atom:0", "atom:1"],
            "point_set": ["atoms", "atoms"],
            "x_px": [2.0, 8.0],
            "y_px": [3.0, 3.0],
            "class_id": [0, 1],
            "class_name": ["class_0", "class_1"],
            "class_color": ["#1f77b4", "#ff7f0e"],
            "roi_id": ["global", "global"],
            "roi_name": ["global", "global"],
            "roi_color": ["#ff9f1c", "#ff9f1c"],
        }
    )


def _segments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "task_name": ["a_forward"],
            "task_type": ["nearest_forward"],
            "source_x_px": [2.0],
            "source_y_px": [3.0],
            "target_x_px": [8.0],
            "target_y_px": [3.0],
            "distance_px": [6.0],
            "source_class_name": ["class_0"],
            "target_class_name": ["class_1"],
            "roi_id": ["global"],
            "roi_name": ["global"],
            "roi_color": ["#ff9f1c"],
        }
    )


def test_plot_measurement_segments_on_image_returns_side_panel_and_lines() -> None:
    image = np.zeros((20, 20), dtype=float)

    fig, image_ax, side_ax = plot_measurement_segments_on_image(image, _points(), _segments())

    assert fig is not None
    assert image_ax is not None
    assert side_ax is not None
    assert image_ax.collections
    plt.close(fig)


def test_plot_measurement_segments_class_colored_atoms_no_error() -> None:
    image = np.zeros((20, 20), dtype=float)

    fig, image_ax, _side_ax = plot_measurement_segments_on_image(image, _points(), _segments(), color_by="class_pair")

    assert len(image_ax.collections) >= 2
    plt.close(fig)


def test_plot_line_guides_on_image_returns_figure() -> None:
    image = np.zeros((20, 20), dtype=float)
    guides = pd.DataFrame(
        {
            "line_id": [0],
            "task_name": ["rows"],
            "roi_id": ["global"],
            "roi_name": ["global"],
            "roi_color": ["#ff9f1c"],
            "line_start_x_px": [0.0],
            "line_start_y_px": [3.0],
            "line_end_x_px": [18.0],
            "line_end_y_px": [3.0],
            "line_label_x_px": [-2.0],
            "line_label_y_px": [3.0],
        }
    )

    fig, image_ax, side_ax = plot_line_guides_on_image(image, _points(), guides)

    assert fig is not None
    assert image_ax is not None
    assert side_ax is not None
    assert image_ax.collections
    plt.close(fig)


def test_plot_measurement_segments_on_image_with_rois_does_not_error() -> None:
    image = np.zeros((20, 20), dtype=float)
    rois = [AnalysisROI("roi_0", polygon_xy_px=((1.0, 1.0), (10.0, 1.0), (10.0, 10.0), (1.0, 10.0)))]

    fig, image_ax, side_ax = plot_measurement_segments_on_image(image, _points(), _segments(), rois=rois)

    assert fig is not None
    assert image_ax.patches
    assert side_ax is not None
    plt.close(fig)


def test_plot_roi_outlines_on_image_draws_polygon_patch() -> None:
    fig, ax = plt.subplots()
    rois = [AnalysisROI("roi_0", polygon_xy_px=((1.0, 1.0), (10.0, 1.0), (10.0, 10.0), (1.0, 10.0)))]

    artists = plot_roi_outlines_on_image(ax, rois)

    assert artists
    assert ax.patches
    plt.close(fig)


def test_plot_basis_vectors_on_image_draws_artists() -> None:
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((20, 20)), cmap="gray")
    points = pd.DataFrame({"point_id": ["atom:0", "atom:1"], "atom_id": [0, 1], "x_px": [2.0, 8.0], "y_px": [3.0, 3.0]})
    basis = resolve_basis_vector_specs(points, [BasisVectorSpec(name="a", from_point_ids=("atom:0", "atom:1"))])

    artists = plot_basis_vectors_on_image(ax, basis)

    assert artists
    plt.close(fig)


def test_plot_basis_check_on_image_clean_overlay_without_long_labels() -> None:
    image = np.zeros((20, 20), dtype=float)
    points = pd.DataFrame(
        {
            "point_id": ["atom:0", "atom:1"],
            "atom_id": [0, 1],
            "x_px": [2.0, 8.0],
            "y_px": [3.0, 3.0],
            "class_id": [0, 1],
            "class_name": ["class_0", "class_1"],
            "class_color": ["#1f77b4", "#ff7f0e"],
            "roi_id": ["roi_0", "roi_0"],
            "roi_name": ["ROI_0", "ROI_0"],
            "roi_color": ["#2ca02c", "#2ca02c"],
        }
    )
    rois = [
        AnalysisROI(
            "roi_0",
            roi_name="ROI_0",
            polygon_xy_px=((1.0, 1.0), (10.0, 1.0), (10.0, 10.0), (1.0, 10.0)),
            color="#2ca02c",
        )
    ]
    basis = resolve_basis_vector_specs(
        points,
        [BasisVectorSpec(name="roi_0_a", from_point_ids=("atom:0", "atom:1"), roi_id="roi_0", basis_role="a")],
    )

    fig, image_ax, legend_ax = plot_basis_check_on_image(
        image,
        points,
        basis,
        rois=rois,
        show_basis_labels=False,
    )

    assert fig is not None
    assert image_ax is not None
    assert legend_ax is not None
    assert len(image_ax.patches) >= 2
    assert not any("L=" in text.get_text() or "P=" in text.get_text() for text in image_ax.texts)
    plt.close(fig)


def test_plot_period_length_histograms_returns_grouped_figures() -> None:
    table = pd.DataFrame(
        {
            "roi_id": ["roi_1", "roi_1"],
            "roi_name": ["ROI_1", "ROI_1"],
            "direction": ["a", "a"],
            "class_selection": ["class_id:0", "class_id:0"],
            "length_A": [10.0, 10.5],
            "angle_delta_deg": [0.0, 1.0],
            "valid": [True, True],
        }
    )

    figures = plot_period_length_histograms(table)

    assert figures
    ax = next(iter(figures.values())).axes[0]
    assert ax.get_title() == "ROI_1 a Length"
    annotations = "\n".join(text.get_text() for text in ax.texts)
    assert "median =" in annotations
    assert "mean =" in annotations
    assert "std =" in annotations
    assert "Å" in annotations
    for fig in figures.values():
        plt.close(fig)


def test_plot_period_angle_delta_histograms_use_short_title_and_degree_stats() -> None:
    table = pd.DataFrame(
        {
            "roi_id": ["roi_1", "roi_1"],
            "roi_name": ["ROI_1", "ROI_1"],
            "direction": ["b", "b"],
            "class_selection": ["class_id:0", "class_id:0"],
            "angle_delta_deg": [0.0, 1.0],
            "valid": [True, True],
        }
    )

    figures = plot_period_angle_delta_histograms(table)

    assert figures
    ax = next(iter(figures.values())).axes[0]
    assert ax.get_title() == "ROI_1 b angle"
    annotations = "\n".join(text.get_text() for text in ax.texts)
    assert "degree" in annotations
    for fig in figures.values():
        plt.close(fig)


def test_plot_period_length_histograms_skip_px_fallback_without_angstrom_values() -> None:
    table = pd.DataFrame(
        {
            "roi_id": ["roi_1", "roi_1"],
            "direction": ["a", "a"],
            "class_selection": ["class_id:0", "class_id:0"],
            "length_px": [10.0, 10.5],
            "valid": [True, True],
        }
    )

    assert plot_period_length_histograms(table) == {}


def test_plot_period_length_histogram_title_override() -> None:
    table = pd.DataFrame(
        {
            "roi_id": ["roi_01", "roi_01"],
            "roi_name": ["ROI 01", "ROI 01"],
            "direction": ["b", "b"],
            "class_selection": ["class_id:2", "class_id:2"],
            "length_A": [5.0, 5.2],
            "valid": [True, True],
        }
    )

    figures = plot_period_length_histograms(
        table,
        title_overrides={("roi_01", "b", "class_id:2", "length"): "Custom b-axis period"},
    )

    assert next(iter(figures.values())).axes[0].get_title() == "Custom b-axis period"
    for fig in figures.values():
        plt.close(fig)


def test_build_period_histogram_title_table_falls_back_to_roi_id_without_roi_name() -> None:
    table = pd.DataFrame(
        {
            "roi_id": ["roi_without_name", "roi_without_name"],
            "direction": ["a", "a"],
            "class_selection": ["class_id:1", "class_id:1"],
            "length_A": [8.0, 8.1],
            "angle_delta_deg": [0.1, 0.2],
            "valid": [True, True],
        }
    )

    title_table = build_period_histogram_title_table(table)

    assert set(title_table["metric"]) == {"length", "angle"}
    assert set(title_table["roi_name"]) == {"roi_without_name"}
    assert title_table["resolved_title"].str.contains("roi_without_name").all()


def test_plot_basis_vectors_on_image_can_display_angstrom_labels_without_px() -> None:
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((10, 10)), cmap="gray")
    basis = pd.DataFrame(
        {
            "basis_name": ["roi_1_a"],
            "vector_x_px": [5.0],
            "vector_y_px": [0.0],
            "length_px": [5.0],
            "period_px": [5.0],
        }
    )

    plot_basis_vectors_on_image(
        ax,
        basis,
        display_unit="A",
        pixel_to_nm=0.1,
    )

    labels = "\n".join(text.get_text() for text in ax.texts)
    assert "5.00Å" in labels
    assert "px" not in labels
    plt.close(fig)


def test_plot_basis_vectors_on_image_hides_distance_without_calibration() -> None:
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((10, 10)), cmap="gray")
    basis = pd.DataFrame(
        {
            "basis_name": ["roi_1_a"],
            "vector_x_px": [5.0],
            "vector_y_px": [0.0],
            "length_px": [5.0],
            "period_px": [5.0],
        }
    )

    plot_basis_vectors_on_image(ax, basis, display_unit="A", pixel_to_nm=None)

    labels = "\n".join(text.get_text() for text in ax.texts)
    assert "px" not in labels
    assert "L=" not in labels
    plt.close(fig)


def test_plot_polygon_cell_map_draws_valid_cells_only() -> None:
    image = np.zeros((20, 20), dtype=float)
    cell_table = pd.DataFrame(
        {
            "p00_x": [1.0, 10.0],
            "p00_y": [1.0, 10.0],
            "p10_x": [5.0, 14.0],
            "p10_y": [1.0, 10.0],
            "p11_x": [5.0, 14.0],
            "p11_y": [5.0, 14.0],
            "p01_x": [1.0, 10.0],
            "p01_y": [5.0, 14.0],
            "eps_area": [0.01, 0.02],
            "valid": [True, False],
        }
    )

    fig, ax = plot_polygon_cell_map(image, cell_table, "eps_area")

    assert ax.collections
    plt.close(fig)


def test_plot_pair_line_distance_errorbar_returns_figure() -> None:
    summary = pd.DataFrame(
        {
            "roi_id": ["global"],
            "line_id": [1],
            "distance_median_A": [2.0],
            "distance_q1_A": [1.5],
            "distance_q3_A": [2.5],
        }
    )

    fig, ax = plot_pair_line_distance_errorbar(summary)

    assert ax.lines
    plt.close(fig)


def test_plot_pair_line_distance_errorbar_uses_global_line_id_when_available() -> None:
    summary = pd.DataFrame(
        {
            "roi_id": ["roi_1", "roi_2"],
            "line_id": [1, 1],
            "global_line_id": [3, 3],
            "distance_median_A": [2.0, 2.2],
            "distance_q1_A": [1.5, 1.8],
            "distance_q3_A": [2.5, 2.6],
        }
    )

    fig, ax = plot_pair_line_distance_errorbar(summary)

    assert ax.get_xlabel() == "Global line index"
    plt.close(fig)


def test_plot_projection_spacing_histogram_uses_angstrom_projection_when_available() -> None:
    table = pd.DataFrame(
        {
            "roi_id": ["roi_1", "roi_1", "roi_1"],
            "projection_s_px": [0.0, 10.0, 20.0],
            "projection_s_A": [0.0, 1.0, 2.0],
        }
    )

    fig, ax = plot_projection_spacing_histogram(table)

    assert ax.get_xlabel() == "Adjacent projection spacing (Å)"
    plt.close(fig)


def test_plot_projection_spacing_histogram_falls_back_to_px_without_angstrom() -> None:
    table = pd.DataFrame(
        {
            "roi_id": ["roi_1", "roi_1", "roi_1"],
            "projection_s_px": [0.0, 10.0, 20.0],
        }
    )

    with pytest.warns(RuntimeWarning, match="falls back to px"):
        fig, ax = plot_projection_spacing_histogram(table)

    assert ax.get_xlabel() == "Adjacent projection spacing (px)"
    plt.close(fig)


def test_add_nm_scalebar_requires_pixel_calibration() -> None:
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((10, 10)), cmap="gray")

    with pytest.raises(ValueError, match="pixel_to_nm"):
        add_nm_scalebar(ax, pixel_to_nm=None)

    plt.close(fig)


def test_plot_cropped_group_centers_draws_filled_arrows_distance_field_and_no_scalebar() -> None:
    image = np.zeros((20, 20), dtype=float)
    image[14:, 8:12] = np.nan
    centroids = pd.DataFrame(
        {
            "roi_id": ["roi_1", "roi_1"],
            "roi_name": ["ROI_1", "ROI_1"],
            "group_name": ["A", "B"],
            "center_x": [5.0, 12.0],
            "center_y": [6.0, 14.0],
            "valid": [True, True],
        }
    )
    displacements = pd.DataFrame(
        {
            "roi_id": ["roi_1"],
            "roi_name": ["ROI_1"],
            "group_A": ["A"],
            "group_B": ["B"],
            "center_A_x": [5.0],
            "center_A_y": [6.0],
            "center_B_x": [12.0],
            "center_B_y": [14.0],
            "dx_px": [7.0],
            "dy_px": [8.0],
            "distance_px": [np.hypot(7.0, 8.0)],
            "distance_A": [np.hypot(7.0, 8.0)],
            "distance_nm": [np.hypot(7.0, 8.0) * 0.1],
            "angle_deg": [0.0],
            "valid": [True],
            "invalid_reason": [""],
        }
    )

    fig, ax = plot_cropped_group_centers_and_displacements(
        image,
        centroids,
        displacements,
        pixel_to_nm=0.1,
        arrow_color="#ffe600",
        arrow_edge_color="#123456",
        arrow_tail_width=0.7,
        arrow_head_width=4.2,
        arrow_head_length=5.0,
        show_centers=False,
    )

    assert ax.get_legend() is None
    assert len(ax.collections) == 0
    assert len(ax.images) == 2
    display_image = np.asarray(ax.images[0].get_array())
    assert np.isfinite(display_image).all()
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == "Displacement (nm)"
    labels = "\n".join(text.get_text() for text in ax.texts)
    assert "nm" not in labels
    arrows = [child for child in ax.get_children() if isinstance(child, FancyArrowPatch)]
    assert arrows
    assert arrows[0].get_facecolor() == mcolors.to_rgba("#ffe600", alpha=0.95)
    assert arrows[0].get_edgecolor() == mcolors.to_rgba("#123456", alpha=0.95)
    assert not ax.lines
    plt.close(fig)
