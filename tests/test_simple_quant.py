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
    build_roi_basis_table,
    compute_line_guides,
    compute_nearest_forward_segments,
    compute_pair_segments,
    compute_periodic_vector_segments,
    compute_group_centroids_by_roi,
    compute_group_pair_displacements,
    add_crop_coordinate_columns_to_group_results,
    crop_image_and_points_by_roi,
    expand_tasks_by_roi_basis,
    find_strict_mutual_nearest_pairs,
    flip_basis_vectors,
    full_image_roi,
    make_pair_center_points,
    prepare_analysis_points,
    resolve_basis_vector_specs,
    run_period_statistics_ab,
    assign_pair_center_lines_by_projection,
    summarize_pair_lines_median_iqr,
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


def test_periodic_vector_uses_period_px_override() -> None:
    points = _analysis_points([0.0, 10.0, 20.0], [0.0, 0.0, 0.0])
    basis = resolve_basis_vector_specs(points, [BasisVectorSpec(name="a", vector_px=(10.0, 0.0), period_px=20.0)])

    segments = compute_periodic_vector_segments(
        points,
        basis,
        PeriodicVectorTask(name="periodic_a", basis="a", match_radius_fraction=0.25),
    )

    assert len(segments) == 1
    assert segments.iloc[0]["source_point_id"] == "atom:0"
    assert segments.iloc[0]["target_point_id"] == "atom:2"
    assert segments.iloc[0]["distance_px"] == pytest.approx(20.0)


def test_flip_basis_vectors_swaps_direction_and_endpoints() -> None:
    points = _analysis_points([0.0, 10.0], [0.0, 0.0])
    basis = resolve_basis_vector_specs(points, [BasisVectorSpec(name="a", from_point_ids=("atom:0", "atom:1"))])

    flipped = flip_basis_vectors(basis, ("a",))

    assert flipped.iloc[0]["vector_x_px"] == pytest.approx(-10.0)
    assert flipped.iloc[0]["ux"] == pytest.approx(-1.0)
    assert flipped.iloc[0]["vx"] == pytest.approx(0.0)
    assert flipped.iloc[0]["vy"] == pytest.approx(-1.0)
    assert flipped.iloc[0]["from_point_id_1"] == "atom:1"
    assert flipped.iloc[0]["from_point_id_2"] == "atom:0"
    assert flipped.iloc[0]["period_px"] == pytest.approx(10.0)


def test_build_roi_basis_table_global_and_per_roi() -> None:
    points = _analysis_points([0.0, 10.0], [0.0, 0.0])
    rois = [AnalysisROI("roi_0"), AnalysisROI("roi_1")]
    basis = resolve_basis_vector_specs(
        points,
        [
            BasisVectorSpec(name="a", basis_role="a", vector_px=(10.0, 0.0)),
            BasisVectorSpec(name="roi_1_a", roi_id="roi_1", basis_role="a", vector_px=(12.0, 0.0)),
        ],
    )

    table = build_roi_basis_table(rois, basis, basis_roles=("a",), global_fallback=True)

    by_roi = table.set_index("roi_id")
    assert by_roi.loc["roi_0", "basis_name"] == "a"
    assert bool(by_roi.loc["roi_0", "is_global"]) is True
    assert by_roi.loc["roi_1", "basis_name"] == "roi_1_a"
    assert bool(by_roi.loc["roi_1", "is_global"]) is False


def test_expand_tasks_by_roi_basis_periodic() -> None:
    roi_basis_table = pd.DataFrame(
        {
            "roi_id": ["roi_0", "roi_1"],
            "roi_name": ["ROI_0", "ROI_1"],
            "basis_role": ["a", "a"],
            "basis_name": ["roi_0_a", "roi_1_a"],
            "is_global": [False, False],
            "found": [True, True],
        }
    )

    tasks = expand_tasks_by_roi_basis(
        [AnalysisROI("roi_0"), AnalysisROI("roi_1")],
        roi_basis_table,
        task_kind="periodic_vector",
        basis_role="a",
        template_name="periodic_atoms",
        periodic_kwargs={"match_radius_fraction": 0.2},
    )

    assert [task.name for task in tasks] == ["roi_0_periodic_atoms_a", "roi_1_periodic_atoms_a"]
    assert [task.roi_ids for task in tasks] == [("roi_0",), ("roi_1",)]
    assert [task.basis for task in tasks] == ["roi_0_a", "roi_1_a"]
    assert all(isinstance(task, PeriodicVectorTask) for task in tasks)


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


def test_run_period_statistics_ab_defaults_to_per_class() -> None:
    points = _analysis_points(
        [0.0, 10.0, 20.0, 1.0, 11.0, 21.0],
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        class_ids=[0, 0, 0, 1, 1, 1],
    )
    basis = resolve_basis_vector_specs(points, [BasisVectorSpec(name="a", basis_role="a", vector_px=(10.0, 0.0))])
    roi_basis = build_roi_basis_table([AnalysisROI("global")], basis, basis_roles=("a",))

    result = run_period_statistics_ab(points, basis, roi_basis, basis_roles=("a",), class_group_mode="per_class")

    summary = result["period_summary_table"]
    assert set(summary["class_selection"]) == {"class_id:0", "class_id:1"}
    assert summary.groupby("class_selection")["n_segments"].first().to_dict() == {"class_id:0": 2, "class_id:1": 2}


def test_run_period_statistics_ab_union_mode_pools_selected_classes() -> None:
    points = _analysis_points([0.0, 10.0, 20.0, 1.0, 11.0, 21.0], [0.0, 0.0, 0.0, 2.0, 2.0, 2.0], class_ids=[0, 0, 0, 1, 1, 1])
    basis = resolve_basis_vector_specs(points, [BasisVectorSpec(name="a", basis_role="a", vector_px=(10.0, 0.0))])
    roi_basis = build_roi_basis_table([AnalysisROI("global")], basis, basis_roles=("a",))

    result = run_period_statistics_ab(
        points,
        basis,
        roi_basis,
        roi_class_selection={"default": (0, 1)},
        basis_roles=("a",),
        class_group_mode="union",
    )

    summary = result["period_summary_table"]
    assert set(summary["class_selection"]) == {"class_id:0+class_id:1"}


def test_strict_mutual_within_class_deduplicates_unordered_pairs() -> None:
    points = _analysis_points([0.0, 5.0], [0.0, 0.0], class_ids=[0, 0])

    pairs = find_strict_mutual_nearest_pairs(points, pair_mode="within_class")

    assert len(pairs) == 1
    assert {pairs.iloc[0]["p1_id"], pairs.iloc[0]["p2_id"]} == {"atom:0", "atom:1"}
    assert bool(pairs.iloc[0]["valid"]) is True


def test_strict_mutual_pair_marks_too_far_invalid() -> None:
    points = _analysis_points([0.0, 5.0], [0.0, 0.0], class_ids=[0, 0])

    pairs = find_strict_mutual_nearest_pairs(points, pair_mode="within_class", max_pair_distance_px=4.0)

    assert len(pairs) == 1
    assert bool(pairs.iloc[0]["valid"]) is False
    assert pairs.iloc[0]["invalid_reason"] == "too_far"


def test_pair_center_line_summary_uses_valid_lines_only() -> None:
    pair_table = pd.DataFrame(
        {
            "roi_id": ["global"] * 4,
            "center_x": [0.0, 0.2, 10.0, 10.2],
            "center_y": [0.0, 0.0, 0.0, 0.0],
            "distance_px": [2.0, 4.0, 6.0, 8.0],
            "distance_A": [2.0, 4.0, 6.0, 8.0],
            "valid": [True, True, True, True],
        }
    )

    assigned, _ = assign_pair_center_lines_by_projection(pair_table, projection_vector=(1.0, 0.0), line_tolerance_px=1.0, min_pairs_per_line=2)
    summary = summarize_pair_lines_median_iqr(assigned)

    assert list(summary["line_id"]) == [1, 2]
    assert list(summary["distance_median_A"]) == [3.0, 7.0]
    assert list(summary["distance_iqr_A"]) == [1.0, 1.0]


def test_pair_center_line_assignment_global_mode_aligns_rois() -> None:
    pair_table = pd.DataFrame(
        {
            "roi_id": ["roi_1", "roi_1", "roi_1", "roi_1", "roi_2", "roi_2"],
            "center_x": [0.0, 0.2, 10.0, 10.2, 0.1, 10.1],
            "center_y": [0.0, 0.0, 0.0, 0.0, 5.0, 5.0],
            "distance_px": [2.0, 4.0, 6.0, 8.0, 3.0, 7.0],
            "distance_A": [2.0, 4.0, 6.0, 8.0, 3.0, 7.0],
            "valid": [True, True, True, True, True, True],
        }
    )
    pair_table.attrs["pixel_to_nm"] = 0.1

    assigned, grouping = assign_pair_center_lines_by_projection(
        pair_table,
        projection_vector=(1.0, 0.0),
        line_tolerance_px=0.5,
        min_pairs_per_line=2,
        line_index_mode="global",
    )
    summary = summarize_pair_lines_median_iqr(assigned)

    assert set(assigned["global_line_id"].dropna().astype(int)) == {1, 2}
    assert assigned.loc[assigned["center_x"] < 1.0, "global_line_id"].dropna().astype(int).nunique() == 1
    assert assigned.loc[assigned["center_x"] > 9.0, "global_line_id"].dropna().astype(int).nunique() == 1
    assert grouping["line_index_mode"].eq("global").all()
    assert grouping["projection_s_median_A"].notna().all()
    assert set(summary["global_line_id"].dropna().astype(int)) == {1, 2}


def test_task3_centroids_are_unweighted_and_displacement_direction_is_a_to_b() -> None:
    points = _analysis_points([0.0, 2.0, 10.0, 14.0], [0.0, 0.0, 1.0, 1.0], class_ids=[0, 0, 1, 1])

    centroids = compute_group_centroids_by_roi(points, center_groups={"A": [0], "B": [1]})
    displacements = compute_group_pair_displacements(centroids, center_pairs=[("A", "B")], pixel_to_nm=0.1)

    by_group = centroids.set_index("group_name")
    assert by_group.loc["A", "center_x"] == pytest.approx(1.0)
    assert by_group.loc["B", "center_x"] == pytest.approx(12.0)
    assert displacements.iloc[0]["dx_px"] == pytest.approx(11.0)
    assert displacements.iloc[0]["distance_A"] == pytest.approx(np.hypot(11.0, 1.0))


def test_crop_image_and_points_by_roi_keeps_polygon_atoms_and_local_coordinates() -> None:
    image = np.arange(40 * 50, dtype=float).reshape(40, 50)
    points = _analysis_points([12.0, 18.0, 30.0], [12.0, 16.0, 30.0], class_ids=[0, 1, 1])
    points.attrs["pixel_to_nm"] = 0.2
    crop_roi = AnalysisROI(
        roi_id="crop_1",
        roi_name="crop_1",
        polygon_xy_px=((10.0, 10.0), (22.0, 10.0), (22.0, 22.0), (10.0, 22.0)),
    )

    result = crop_image_and_points_by_roi(image, points, crop_roi)
    cropped = result["points"]

    assert result["image"].shape == (12, 12)
    assert tuple(result["origin_xy_px"]) == (10.0, 10.0)
    assert list(cropped["atom_id"]) == [0, 1]
    assert list(cropped["global_x_px"]) == [12.0, 18.0]
    assert list(cropped["x_px"]) == [2.0, 8.0]
    assert list(cropped["y_px"]) == [2.0, 6.0]
    assert cropped.attrs["pixel_to_nm"] == pytest.approx(0.2)
    assert cropped["x_nm"].tolist() == pytest.approx([0.4, 1.6])
    assert result["crop_table"].iloc[0]["width_nm"] == pytest.approx(2.4)
    assert result["crop_table"].iloc[0]["crop_mode"] == "oriented_rectangle"


def test_crop_image_and_points_by_rotated_rectangle_aligns_long_axis() -> None:
    image = np.zeros((60, 60), dtype=float)
    center = np.asarray([30.0, 30.0])
    u = np.asarray([np.sqrt(0.5), np.sqrt(0.5)])
    v = np.asarray([-np.sqrt(0.5), np.sqrt(0.5)])
    half_long = 10.0
    half_short = 3.0
    corners = np.vstack(
        [
            center - half_long * u - half_short * v,
            center + half_long * u - half_short * v,
            center + half_long * u + half_short * v,
            center - half_long * u + half_short * v,
        ]
    )
    points = _analysis_points(
        [float(center[0]), float((center + 5 * u)[0])],
        [float(center[1]), float((center + 5 * u)[1])],
        class_ids=[0, 1],
    )
    points.attrs["pixel_to_nm"] = 0.1
    crop_roi = AnalysisROI(
        roi_id="rot",
        roi_name="rot",
        polygon_xy_px=tuple((float(x), float(y)) for x, y in corners),
    )

    result = crop_image_and_points_by_roi(image, points, crop_roi)
    cropped = result["points"]

    assert result["crop_table"].iloc[0]["crop_mode"] == "oriented_rectangle"
    assert result["image"].shape[1] >= result["image"].shape[0]
    assert cropped.loc[1, "x_px"] - cropped.loc[0, "x_px"] == pytest.approx(5.0, abs=1e-6)
    assert abs(cropped.loc[1, "y_px"] - cropped.loc[0, "y_px"]) < 1e-6


def test_crop_full_image_roi_defaults_to_image_extent() -> None:
    roi = full_image_roi(np.zeros((8, 10)), roi_id="full")

    assert roi.roi_id == "full"
    assert roi.polygon_xy_px == ((0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0))


def test_cropped_group_results_include_global_coordinates_and_nm_displacements() -> None:
    points = _analysis_points([2.0, 4.0, 8.0], [1.0, 1.0, 5.0], class_ids=[0, 0, 1])
    points.attrs["pixel_to_nm"] = 0.5

    centroids = compute_group_centroids_by_roi(points, center_groups={"A": [0], "B": [1]})
    displacements = compute_group_pair_displacements(centroids, center_pairs=[("A", "B")], pixel_to_nm=0.5)
    annotated_centroids, annotated_displacements = add_crop_coordinate_columns_to_group_results(
        centroids,
        displacements,
        crop_origin_xy_px=(10.0, 20.0),
        pixel_to_nm=0.5,
    )

    center_a = annotated_centroids.set_index("group_name").loc["A"]
    assert center_a["center_x_local_px"] == pytest.approx(3.0)
    assert center_a["center_x_global_px"] == pytest.approx(13.0)
    assert center_a["center_x_global_nm"] == pytest.approx(6.5)
    row = annotated_displacements.iloc[0]
    assert row["dx_px"] == pytest.approx(5.0)
    assert row["dy_px"] == pytest.approx(4.0)
    assert row["dx_nm"] == pytest.approx(2.5)
    assert row["dy_nm"] == pytest.approx(2.0)
    assert row["distance_nm"] == pytest.approx(np.hypot(5.0, 4.0) * 0.5)
    assert row["angle_deg"] == pytest.approx(np.degrees(np.arctan2(4.0, 5.0)))


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
