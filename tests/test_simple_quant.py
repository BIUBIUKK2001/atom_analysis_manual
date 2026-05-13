from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from em_atom_workbench.session import AnalysisSession, PixelCalibration
from em_atom_workbench.simple_quant import (
    AnalysisROI,
    BasisVectorSpec,
    LineGuideTask,
    NearestForwardTask,
    PairSegmentTask,
    PeriodicVectorTask,
    compute_line_guides,
    compute_nearest_forward_segments,
    compute_pair_segments,
    compute_periodic_vector_segments,
    make_pair_center_points,
    prepare_analysis_points,
    resolve_basis_vector_specs,
)


def _analysis_points(xs: list[float], ys: list[float], class_ids: list[int] | None = None, roi_id: str = "global") -> pd.DataFrame:
    class_ids = class_ids or [0] * len(xs)
    points = pd.DataFrame(
        {
            "point_id": [f"atom:{index}" for index in range(len(xs))],
            "source_type": "atom",
            "point_set": "atoms",
            "atom_id": np.arange(len(xs), dtype=int),
            "x_px": xs,
            "y_px": ys,
            "x_nm": np.asarray(xs, dtype=float) * 0.1,
            "y_nm": np.asarray(ys, dtype=float) * 0.1,
            "class_id": class_ids,
            "class_name": [f"class_{value}" for value in class_ids],
            "class_color": pd.NA,
            "column_role": pd.NA,
            "keep": True,
            "quality_score": np.nan,
            "roi_id": roi_id,
            "roi_name": roi_id,
            "roi_color": "#ff9f1c",
            "scope_id": f"{roi_id}:atoms:curated",
            "source_table": "curated",
        }
    )
    return points


def _grid_points(nx: int = 4, ny: int = 3, spacing_x: float = 10.0, spacing_y: float = 12.0) -> pd.DataFrame:
    xs, ys = np.meshgrid(np.arange(nx) * spacing_x, np.arange(ny) * spacing_y)
    class_ids = np.tile([0, 1, 0, 1][:nx], ny).tolist()
    return _analysis_points(xs.ravel().tolist(), ys.ravel().tolist(), class_ids)


def _grid_session() -> AnalysisSession:
    points = _grid_points()[["atom_id", "x_px", "y_px", "class_id", "class_name", "keep"]].copy()
    session = AnalysisSession(
        name="simple_quant_v2_grid",
        pixel_calibration=PixelCalibration(size=0.1, unit="nm", source="unit_test"),
        raw_image=np.zeros((50, 50)),
    )
    session.curated_points = points
    session.set_stage("curated")
    return session


def test_basis_vector_preserves_length() -> None:
    points = _analysis_points([0.0, 3.0], [0.0, 4.0])

    table = resolve_basis_vector_specs(points, [BasisVectorSpec(name="a", from_point_ids=("atom:0", "atom:1"))])

    assert table.iloc[0]["vector_x_px"] == pytest.approx(3.0)
    assert table.iloc[0]["vector_y_px"] == pytest.approx(4.0)
    assert table.iloc[0]["length_px"] == pytest.approx(5.0)
    assert table.iloc[0]["period_px"] == pytest.approx(5.0)


def test_nearest_forward_segments_grid() -> None:
    points = _grid_points()
    basis = resolve_basis_vector_specs(points, [BasisVectorSpec(name="a", vector_px=(10.0, 0.0))])

    segments = compute_nearest_forward_segments(
        points,
        basis,
        NearestForwardTask(name="a_forward", basis="a", perpendicular_tolerance_px=1.0),
    )

    assert len(segments) == 9
    assert set(segments["task_type"]) == {"nearest_forward"}
    assert np.allclose(segments["distance_px"], 10.0)


def test_periodic_vector_segments_skip_nearest_neighbor() -> None:
    points = _analysis_points([0.0, 5.0, 10.0, 20.0], [0.0, 2.0, 0.0, 0.0])
    basis = resolve_basis_vector_specs(points, [BasisVectorSpec(name="a", vector_px=(10.0, 0.0))])

    segments = compute_periodic_vector_segments(
        points,
        basis,
        PeriodicVectorTask(name="periodic_a", basis="a", match_radius_fraction=0.25),
    )

    assert list(segments["source_point_id"]) == ["atom:0", "atom:2"]
    assert list(segments["target_point_id"]) == ["atom:2", "atom:3"]
    assert np.allclose(segments["distance_px"], 10.0)
    assert segments["period_residual_px"].max() < 1e-9


def test_multi_roi_segments() -> None:
    session = _grid_session()
    rois = [
        AnalysisROI("left", polygon_xy_px=((-1.0, -1.0), (11.0, -1.0), (11.0, 30.0), (-1.0, 30.0))),
        AnalysisROI("right", polygon_xy_px=((19.0, -1.0), (31.0, -1.0), (31.0, 30.0), (19.0, 30.0))),
    ]
    points = prepare_analysis_points(session, rois=rois)
    basis = resolve_basis_vector_specs(points, [BasisVectorSpec(name="a", vector_px=(10.0, 0.0))])

    segments = compute_nearest_forward_segments(
        points,
        basis,
        NearestForwardTask(name="roi_a_forward", basis="a", perpendicular_tolerance_px=1.0),
    )

    assert set(segments["roi_id"]) == {"left", "right"}
    assert set(segments["scope_id"]) == {"left:atoms:curated", "right:atoms:curated"}
    assert segments.groupby("roi_id")["segment_id"].count().to_dict() == {"left": 3, "right": 3}


def test_pair_center_points() -> None:
    points = _analysis_points([0.0, 4.0], [0.0, 0.0], class_ids=[0, 1])
    segments = compute_pair_segments(
        points,
        None,
        PairSegmentTask(name="pair01", source_class_ids=(0,), target_class_ids=(1,)),
    )

    centers = make_pair_center_points(segments, class_name="pair_center")

    assert len(centers) == 1
    assert centers.iloc[0]["point_set"] == "pair_centers"
    assert centers.iloc[0]["x_px"] == pytest.approx(2.0)
    assert centers.iloc[0]["y_px"] == pytest.approx(0.0)
    assert centers.iloc[0]["parent_source_point_id"] == "atom:0"
    assert centers.iloc[0]["parent_target_point_id"] == "atom:1"


def test_line_guides_group_axis_t_and_s() -> None:
    points = _grid_points()
    basis = resolve_basis_vector_specs(points, [BasisVectorSpec(name="a", vector_px=(10.0, 0.0))])

    rows = compute_line_guides(
        points,
        basis,
        LineGuideTask(name="rows", basis="a", group_axis="t", line_tolerance_px=1.0, min_points_per_line=4),
    )
    columns = compute_line_guides(
        points,
        basis,
        LineGuideTask(name="columns", basis="a", group_axis="s", line_tolerance_px=1.0, min_points_per_line=3),
    )

    assert len(rows) == 3
    assert len(columns) == 4
    assert set(rows["group_axis"]) == {"t"}
    assert set(columns["group_axis"]) == {"s"}
