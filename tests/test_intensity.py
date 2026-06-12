from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from em_atom_workbench.intensity import (
    build_disk_offsets,
    compute_disk_intensity_table,
    prepare_disk_intensity_points,
    summarize_disk_intensity,
)
from em_atom_workbench.session import AnalysisSession


def test_build_disk_offsets() -> None:
    offsets = build_disk_offsets(2)

    assert offsets.ndim == 2
    assert offsets.shape[1] == 2
    assert len(offsets) > 0
    assert np.all(np.sum(offsets.astype(float) ** 2, axis=1) <= 4.0)
    assert any((offsets == np.array([0, 0])).all(axis=1))


def test_compute_disk_intensity_table_single_point() -> None:
    image = np.arange(25, dtype=float).reshape(5, 5)
    points = pd.DataFrame({"point_id": ["p0"], "atom_id": [0], "x_px": [2.0], "y_px": [2.0]})

    table = compute_disk_intensity_table(points, image, disk_radius_px=1.0, channel_name="primary")

    assert table.iloc[0]["n_pixels"] == 5
    assert table.iloc[0]["disk_intensity_sum"] == pytest.approx(60.0)
    assert table.iloc[0]["disk_intensity_mean"] == pytest.approx(12.0)
    assert table.iloc[0]["status"] == "ok"
    assert bool(table.iloc[0]["is_edge"]) is False


def test_compute_disk_intensity_table_edge_flag() -> None:
    image = np.ones((5, 5), dtype=float)
    points = pd.DataFrame({"point_id": ["edge"], "atom_id": [0], "x_px": [0.0], "y_px": [0.0]})

    table = compute_disk_intensity_table(points, image, disk_radius_px=2.0)

    assert bool(table.iloc[0]["is_edge"]) is True
    assert table.iloc[0]["status"] == "edge_clipped"
    assert table.iloc[0]["n_pixels"] > 0


def test_summarize_disk_intensity_basic() -> None:
    table = pd.DataFrame(
        {
            "coordinate_source": ["refined", "refined", "refined"],
            "class_id": [1, 1, 1],
            "class_name": ["Hf", "Hf", "Hf"],
            "channel_name": ["primary", "primary", "primary"],
            "disk_intensity_sum": [10.0, 20.0, 30.0],
        }
    )

    summary = summarize_disk_intensity(table)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["count"] == 3
    assert row["mean"] == pytest.approx(20.0)
    assert row["median"] == pytest.approx(20.0)
    assert row["min"] == pytest.approx(10.0)
    assert row["max"] == pytest.approx(30.0)


def test_coordinate_source_candidate_and_refined_are_supported() -> None:
    session = AnalysisSession(name="coords", raw_image=np.zeros((8, 8), dtype=float))
    session.candidate_points = pd.DataFrame(
        {
            "atom_id": [0],
            "x_px": [1.0],
            "y_px": [2.0],
            "class_id": [1],
            "class_name": ["candidate_class"],
        }
    )
    session.refined_points = pd.DataFrame(
        {
            "atom_id": [0],
            "x_px": [3.5],
            "y_px": [4.5],
            "class_id": [1],
            "class_name": ["refined_class"],
        }
    )

    candidate = prepare_disk_intensity_points(session, coordinate_source="candidate")
    refined = prepare_disk_intensity_points(session, coordinate_source="refined")

    assert candidate.iloc[0]["coordinate_source"] == "candidate"
    assert candidate.iloc[0]["source_table"] == "candidate_points"
    assert candidate.iloc[0]["x_px"] == pytest.approx(1.0)
    assert candidate.iloc[0]["y_px"] == pytest.approx(2.0)
    assert refined.iloc[0]["coordinate_source"] == "refined"
    assert refined.iloc[0]["source_table"] == "refined_points"
    assert refined.iloc[0]["x_px"] == pytest.approx(3.5)
    assert refined.iloc[0]["y_px"] == pytest.approx(4.5)
