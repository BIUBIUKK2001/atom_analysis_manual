from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from em_atom_workbench.simple_quant_plotting import (
    plot_line_guides_on_image,
    plot_measurement_segments_on_image,
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
