from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from em_atom_workbench.simple_quant import AnalysisROI, BasisVectorSpec, resolve_basis_vector_specs
from em_atom_workbench.simple_quant_plotting import (
    plot_basis_check_on_image,
    plot_basis_vectors_on_image,
    plot_pair_line_distance_errorbar,
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
            "roi_id": ["global", "global"],
            "direction": ["a", "a"],
            "class_selection": ["class_id:0", "class_id:0"],
            "length_A": [10.0, 10.5],
            "angle_delta_deg": [0.0, 1.0],
            "valid": [True, True],
        }
    )

    figures = plot_period_length_histograms(table)

    assert figures
    for fig in figures.values():
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
