from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from em_atom_workbench.session import AnalysisSession, PixelCalibration
from em_atom_workbench.simple_quant import (
    DirectionSpec,
    DirectionalSpacingTask,
    LineGroupingTask,
    PairDistanceTask,
    assign_lines_by_projection,
    compute_directional_spacing,
    compute_line_spacing,
    compute_pair_distances,
    prepare_quant_points,
    resolve_direction_specs,
)


def _grid_session(*, calibrated: bool = True) -> AnalysisSession:
    xs, ys = np.meshgrid(np.arange(4) * 10.0, np.arange(3) * 12.0)
    points = pd.DataFrame(
        {
            "atom_id": np.arange(xs.size, dtype=int),
            "x_px": xs.ravel(),
            "y_px": ys.ravel(),
            "class_id": np.tile([0, 1, 0, 1], 3),
            "class_name": np.tile(["class_0", "class_1", "class_0", "class_1"], 3),
            "keep": True,
        }
    )
    calibration = PixelCalibration(size=0.2, unit="nm", source="unit_test") if calibrated else PixelCalibration()
    session = AnalysisSession(name="simple_quant_grid", pixel_calibration=calibration, raw_image=np.zeros((40, 40)))
    session.curated_points = points
    session.set_stage("curated")
    return session


def _directions(points: pd.DataFrame) -> pd.DataFrame:
    return resolve_direction_specs(
        points,
        [
            DirectionSpec(name="u", vector_px=(1.0, 0.0)),
            DirectionSpec(name="v", vector_px=(0.0, 1.0)),
        ],
    )


def test_directional_spacing_u_on_regular_grid() -> None:
    points = prepare_quant_points(_grid_session(), source_table="curated")
    direction_table = _directions(points)

    table = compute_directional_spacing(
        points,
        direction_table,
        [DirectionalSpacingTask(name="u_spacing", direction="u", perpendicular_tolerance_px=1.0)],
    )

    assert len(table) == 9
    assert np.allclose(table["distance_px"], 10.0)


def test_directional_spacing_v_on_regular_grid() -> None:
    points = prepare_quant_points(_grid_session(), source_table="curated")
    direction_table = _directions(points)

    table = compute_directional_spacing(
        points,
        direction_table,
        [DirectionalSpacingTask(name="v_spacing", direction="v", perpendicular_tolerance_px=1.0)],
    )

    assert len(table) == 8
    assert np.allclose(table["distance_px"], 12.0)


def test_class_to_class_nearest_pair_distance() -> None:
    points = prepare_quant_points(_grid_session(), source_table="curated")

    table = compute_pair_distances(
        points,
        [
            PairDistanceTask(
                name="class0_class1",
                source_class_ids=(0,),
                target_class_ids=(1,),
                max_distance_px=11.0,
                unique_pairs=True,
            )
        ],
    )

    assert not table.empty
    assert table["distance_px"].min() == pytest.approx(10.0)
    assert set(table["source_class_id"]) == {0}
    assert set(table["target_class_id"]) == {1}


def test_explicit_atom_pairs_distance() -> None:
    points = prepare_quant_points(_grid_session(), source_table="curated")

    table = compute_pair_distances(
        points,
        [PairDistanceTask(name="explicit", explicit_atom_pairs=((0, 5),), unique_pairs=True)],
    )

    assert len(table) == 1
    assert table.iloc[0]["source_atom_id"] == 0
    assert table.iloc[0]["target_atom_id"] == 5
    assert table.iloc[0]["distance_px"] == pytest.approx(np.hypot(10.0, 12.0))


def test_group_axis_t_groups_rows() -> None:
    points = prepare_quant_points(_grid_session(), source_table="curated")
    direction_table = _directions(points)

    assignments = assign_lines_by_projection(
        points,
        direction_table,
        LineGroupingTask(name="rows", direction="u", group_axis="t", line_tolerance_px=1.0, min_atoms_per_line=4),
    )

    assert assignments["line_id"].nunique() == 3
    assert assignments.groupby("line_id")["atom_id"].count().tolist() == [4, 4, 4]


def test_group_axis_s_groups_columns() -> None:
    points = prepare_quant_points(_grid_session(), source_table="curated")
    direction_table = _directions(points)

    assignments = assign_lines_by_projection(
        points,
        direction_table,
        LineGroupingTask(name="columns", direction="u", group_axis="s", line_tolerance_px=1.0, min_atoms_per_line=3),
    )

    assert assignments["line_id"].nunique() == 4
    assert assignments.groupby("line_id")["atom_id"].count().tolist() == [3, 3, 3, 3]


def test_calibrated_spacing_outputs_pm_and_uncalibrated_stays_nan() -> None:
    calibrated_points = prepare_quant_points(_grid_session(calibrated=True), source_table="curated")
    direction_table = _directions(calibrated_points)
    table = compute_directional_spacing(
        calibrated_points,
        direction_table,
        [DirectionalSpacingTask(name="u_spacing", direction="u", perpendicular_tolerance_px=1.0)],
    )
    assert table["distance_pm"].iloc[0] == pytest.approx(2000.0)

    uncalibrated_points = prepare_quant_points(_grid_session(calibrated=False), source_table="curated")
    table = compute_directional_spacing(
        uncalibrated_points,
        _directions(uncalibrated_points),
        [DirectionalSpacingTask(name="u_spacing", direction="u", perpendicular_tolerance_px=1.0)],
    )
    assert table["distance_px"].notna().all()
    assert table["distance_pm"].isna().all()


def test_line_spacing_computes_width_and_next_spacing() -> None:
    points = prepare_quant_points(_grid_session(), source_table="curated")
    direction_table = _directions(points)
    task = LineGroupingTask(name="rows", direction="u", group_axis="t", line_tolerance_px=1.0, min_atoms_per_line=4)

    assignments = assign_lines_by_projection(points, direction_table, task)
    spacing = compute_line_spacing(points, assignments, direction_table, task)

    valid = spacing.loc[spacing["next_atom_id"].notna()]
    assert np.allclose(valid["spacing_to_next_px"], 10.0)
    assert valid["spacing_to_next_pm"].iloc[0] == pytest.approx(2000.0)
