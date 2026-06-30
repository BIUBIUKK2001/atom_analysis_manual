from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from em_atom_workbench.intensity_plotting import (
    launch_stack_refinement_napari_viewer,
    plot_stack_intensity_histogram,
    plot_stack_intensity_profiles,
    plot_stack_refinement_shift_profile,
    plot_stack_slice_intensity_map,
)


def _intensity_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "slice_index": [0, 0, 1, 1],
            "slice_label": [0, 0, 1, 1],
            "x_px": [1.0, 3.0, 1.0, 3.0],
            "y_px": [1.0, 3.0, 1.0, 3.0],
            "class_name": ["A", "B", "A", "B"],
            "class_color": ["#ff0000", "#0000ff", "#ff0000", "#0000ff"],
            "disk_intensity_mean": [1.0, 2.0, 3.0, 4.0],
        }
    )


def _summary_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "slice_index": [0, 1, 0, 1],
            "slice_label": [0, 1, 0, 1],
            "class_id": [0, 0, 1, 1],
            "class_name": ["A", "A", "B", "B"],
            "channel_name": ["primary"] * 4,
            "metric": ["disk_intensity_mean"] * 4,
            "mean": [1.0, 3.0, 2.0, 4.0],
            "std": [0.1, 0.2, 0.1, 0.2],
            "median": [1.0, 3.0, 2.0, 4.0],
            "q25": [0.9, 2.8, 1.9, 3.8],
            "q75": [1.1, 3.2, 2.1, 4.2],
        }
    )


def test_plotting_functions_return_fig_ax() -> None:
    stack = np.zeros((2, 5, 5), dtype=float)

    fig, ax = plot_stack_intensity_profiles(_summary_table())
    assert fig is ax.figure

    fig, ax = plot_stack_slice_intensity_map(stack, _intensity_table(), slice_index=0)
    assert fig is ax.figure

    fig, axes = plot_stack_intensity_histogram(_intensity_table())
    assert fig is not None
    assert axes is not None

    refined = _intensity_table().assign(center_shift_px=[0.1, 0.2, 0.3, 0.4])
    fig, ax = plot_stack_refinement_shift_profile(refined)
    assert fig is ax.figure


def test_invalid_metric_raises_value_error() -> None:
    stack = np.zeros((2, 5, 5), dtype=float)
    with pytest.raises(ValueError, match="Metric column"):
        plot_stack_slice_intensity_map(stack, _intensity_table(), slice_index=0, metric="missing")
    with pytest.raises(ValueError, match="Metric column"):
        plot_stack_intensity_histogram(_intensity_table(), metric="missing")


def test_napari_viewer_function_is_importable() -> None:
    assert callable(launch_stack_refinement_napari_viewer)
