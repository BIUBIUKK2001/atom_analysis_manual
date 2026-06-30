from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from em_atom_workbench.stack_intensity import compute_per_slice_disk_intensity_table, summarize_stack_disk_intensity


def _slice_points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "point_id": ["p0", "p0"],
            "atom_id": [0, 0],
            "slice_index": [0, 1],
            "slice_label": ["a", "b"],
            "x_seed_px": [1.0, 1.0],
            "y_seed_px": [1.0, 1.0],
            "x_px": [1.0, 3.0],
            "y_px": [1.0, 3.0],
            "class_id": [1, 1],
            "class_name": ["Hf", "Hf"],
            "quality_score": [0.9, 0.8],
            "center_shift_px": [0.0, 2.8],
            "attempted_center_shift_px": [0.0, 2.8],
            "center_shift_rejected": [False, False],
            "position_source": ["refined", "refined"],
            "fit_success": [True, True],
            "refinement_method": ["gaussian", "gaussian"],
            "refinement_path": ["adaptive_atomap", "adaptive_atomap"],
            "fit_residual": [0.1, 0.2],
        }
    )


def test_compute_per_slice_disk_intensity_table_samples_correct_slice_only() -> None:
    stack = np.zeros((2, 5, 5), dtype=float)
    stack[0, 1, 1] = 10.0
    stack[1, 3, 3] = 20.0

    table = compute_per_slice_disk_intensity_table(_slice_points(), stack, disk_radius_px=0.1)

    assert len(table) == 2
    assert table["slice_index"].tolist() == [0, 1]
    assert table["disk_intensity_sum"].tolist() == pytest.approx([10.0, 20.0])
    assert table["coordinate_mode"].tolist() == ["slice_refined", "slice_refined"]


def test_refined_coordinates_change_intensity_relative_to_seed_coordinates() -> None:
    stack = np.zeros((1, 5, 5), dtype=float)
    stack[0, 1, 1] = 1.0
    stack[0, 3, 3] = 9.0
    points = pd.DataFrame({"slice_index": [0], "x_seed_px": [1.0], "y_seed_px": [1.0], "x_px": [3.0], "y_px": [3.0]})

    table = compute_per_slice_disk_intensity_table(points, stack, disk_radius_px=0.1)

    assert table.iloc[0]["disk_intensity_sum"] == pytest.approx(9.0)


def test_invalid_slice_index_kept_and_flagged() -> None:
    stack = np.zeros((2, 5, 5), dtype=float)
    points = pd.DataFrame({"slice_index": [5], "x_px": [1.0], "y_px": [1.0], "quality_score": [0.5]})

    table = compute_per_slice_disk_intensity_table(points, stack, disk_radius_px=1.0)

    assert len(table) == 1
    assert table.iloc[0]["status"] == "invalid_slice"
    assert pd.isna(table.iloc[0]["disk_intensity_mean"])
    assert table.iloc[0]["quality_score"] == pytest.approx(0.5)


def test_refinement_diagnostic_columns_preserved_and_summary_works() -> None:
    stack = np.ones((2, 5, 5), dtype=float)
    table = compute_per_slice_disk_intensity_table(_slice_points(), stack, disk_radius_px=1.0)

    for column in ["x_seed_px", "y_seed_px", "quality_score", "center_shift_px", "refinement_path", "fit_residual"]:
        assert column in table.columns
    summary = summarize_stack_disk_intensity(table)
    assert len(summary) == 2
    assert summary["count"].tolist() == [1, 1]
