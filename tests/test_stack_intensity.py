from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from em_atom_workbench.stack_intensity import compute_stack_disk_intensity_table, summarize_stack_disk_intensity


def _points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "point_id": ["p0"],
            "atom_id": [7],
            "x_px": [2.0],
            "y_px": [2.0],
            "class_id": [1],
            "class_name": ["Hf"],
            "class_color": ["#123456"],
        }
    )


def test_compute_stack_disk_intensity_table_one_row_per_slice() -> None:
    stack = np.stack([np.arange(25, dtype=float).reshape(5, 5) + 100.0 * idx for idx in range(3)])

    table = compute_stack_disk_intensity_table(_points(), stack, disk_radius_px=1.0, channel_name="primary")

    assert len(table) == 3
    assert table["slice_index"].tolist() == [0, 1, 2]
    assert table["disk_intensity_mean"].tolist() == pytest.approx([12.0, 112.0, 212.0])
    assert table["class_id"].tolist() == [1, 1, 1]
    assert table["class_name"].tolist() == ["Hf", "Hf", "Hf"]
    assert table["class_color"].tolist() == ["#123456", "#123456", "#123456"]
    assert table["coordinate_mode"].tolist() == ["fixed", "fixed", "fixed"]


def test_slice_indices_and_labels() -> None:
    stack = np.ones((4, 5, 5), dtype=float)

    table = compute_stack_disk_intensity_table(
        _points(),
        stack,
        disk_radius_px=1.0,
        slice_indices=[1, 3],
        slice_labels=["one", "three"],
    )

    assert table["slice_index"].tolist() == [1, 3]
    assert table["slice_label"].tolist() == ["one", "three"]


def test_edge_clipped_status_is_kept() -> None:
    points = pd.DataFrame({"point_id": ["edge"], "x_px": [0.0], "y_px": [0.0]})
    stack = np.ones((2, 5, 5), dtype=float)

    table = compute_stack_disk_intensity_table(points, stack, disk_radius_px=2.0)

    assert set(table["status"]) == {"edge_clipped"}
    assert table["is_edge"].astype(bool).all()


def test_summarize_stack_disk_intensity_groups_by_class_and_slice() -> None:
    table = pd.DataFrame(
        {
            "slice_index": [0, 0, 1, 1],
            "slice_label": [0, 0, 1, 1],
            "class_id": [1, 1, 1, 1],
            "class_name": ["Hf", "Hf", "Hf", "Hf"],
            "channel_name": ["primary"] * 4,
            "disk_intensity_mean": [1.0, 3.0, 10.0, 14.0],
        }
    )

    summary = summarize_stack_disk_intensity(table)

    assert summary["slice_index"].tolist() == [0, 1]
    assert summary["mean"].tolist() == pytest.approx([2.0, 12.0])
    assert summary["sem"].notna().all()
    assert summary["metric"].tolist() == ["disk_intensity_mean", "disk_intensity_mean"]


def test_invalid_non_3d_stack_raises() -> None:
    with pytest.raises(ValueError, match="3D"):
        compute_stack_disk_intensity_table(_points(), np.zeros((5, 5)), disk_radius_px=1.0)
