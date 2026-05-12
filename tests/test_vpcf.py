from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from em_atom_workbench.session import AnalysisSession, PixelCalibration
from em_atom_workbench.utils import load_or_connect_session, save_active_session
from em_atom_workbench.vpcf import (
    VPCFConfig,
    compute_batch_vpcf,
    compute_local_vpcf,
    compute_session_vpcf,
    points_for_vpcf,
)


def test_square_lattice_local_vpcf_counts_neighbors_and_normalizes() -> None:
    coords = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [-10.0, 0.0],
            [0.0, 10.0],
            [0.0, -10.0],
            [10.0, 10.0],
        ]
    )
    result = compute_local_vpcf(
        coords,
        center_index=0,
        config=VPCFConfig(r_max_px=10.1, extent_px=12.0, grid_size=65, sigma_grid_px=1.25),
    )

    assert set(result.neighbor_indices.tolist()) == {1, 2, 3, 4}
    assert result.vectors.shape == (4, 2)
    assert result.H.shape == (65, 65)
    assert np.isclose(result.H.max(), 1.0)
    assert result.metadata["neighbor_count"] == 4


def test_rotated_lattice_vpcf_vectors_rotate_with_coordinates() -> None:
    theta = np.deg2rad(35.0)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    base = np.array(
        [
            [0.0, 0.0],
            [8.0, 0.0],
            [0.0, 8.0],
            [-8.0, 0.0],
            [0.0, -8.0],
        ]
    )
    coords = base @ rotation.T
    result = compute_local_vpcf(
        coords,
        center_index=0,
        config=VPCFConfig(r_max_px=8.5, extent_px=10.0, grid_size=65, sigma_grid_px=1.0),
    )

    angles = np.sort(np.mod(np.arctan2(result.vectors[:, 1], result.vectors[:, 0]), 2 * np.pi))
    expected = np.sort(np.mod(np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]) + theta, 2 * np.pi))
    assert result.metadata["neighbor_count"] == 4
    assert np.allclose(angles, expected)
    assert np.isclose(result.H.max(), 1.0)


def test_local_vpcf_empty_edge_center_returns_zero_matrix() -> None:
    coords = np.array([[0.0, 0.0], [30.0, 0.0]])
    result = compute_local_vpcf(
        coords,
        center_index=0,
        config=VPCFConfig(r_max_px=5.0, extent_px=8.0, grid_size=33),
    )

    assert result.neighbor_indices.size == 0
    assert result.vectors.shape == (0, 2)
    assert np.count_nonzero(result.H) == 0
    assert result.metadata["neighbor_count"] == 0


def test_batch_vpcf_uses_shared_grid_and_returns_expected_shape() -> None:
    xs, ys = np.meshgrid(np.arange(3) * 8.0, np.arange(3) * 8.0)
    coords = np.column_stack([xs.ravel(), ys.ravel()])
    result = compute_batch_vpcf(
        coords,
        center_indices=[1, 4, 7],
        config=VPCFConfig(r_max_px=8.5, extent_px=10.0, grid_size=41),
    )

    assert result.average_H.shape == (41, 41)
    assert result.x_axis.shape == (41,)
    assert np.array_equal(result.x_axis, result.y_axis)
    assert result.metadata["center_count"] == 3
    assert result.center_summary["neighbor_count"].tolist() == [3, 4, 3]


def test_session_vpcf_filters_keep_false_points_and_updates_active_session() -> None:
    results_root = Path(".test-artifacts") / "vpcf_session_flow"
    results_root.mkdir(parents=True, exist_ok=True)
    for filename in ("_active_session.pkl", "_active_session.json"):
        (results_root / filename).unlink(missing_ok=True)

    session = AnalysisSession(
        name="vpcf_session",
        pixel_calibration=PixelCalibration(size=0.5, unit="nm", source="unit_test"),
    )
    session.set_stage("curated")
    session.curated_points = pd.DataFrame(
        {
            "atom_id": [10, 11, 12, 13, 14],
            "x_px": [0.0, 10.0, 20.0, 0.0, 10.0],
            "y_px": [0.0, 0.0, 0.0, 10.0, 10.0],
            "keep": [True, True, False, True, True],
        }
    )

    points = points_for_vpcf(session, use_keep=True)
    assert points["atom_id"].tolist() == [10, 11, 13, 14]

    session = compute_session_vpcf(
        session,
        VPCFConfig(r_max_px=12.0, extent_px=14.0, grid_size=33),
        center_atom_id=10,
        use_keep=True,
    )
    active_path = save_active_session(session, results_root)
    reloaded = load_or_connect_session(results_root, required_stage="vpcf", session_path=active_path)

    assert reloaded.current_stage == "vpcf"
    assert reloaded.vpcf_results
    assert reloaded.vpcf_results["points_used"]["atom_id"].tolist() == [10, 11, 13, 14]
    assert 12 not in reloaded.vpcf_results["points_used"]["atom_id"].tolist()
    assert reloaded.vpcf_results["example_metadata"]["atom_id"] == 10
    assert reloaded.vpcf_results["coordinate_unit"] == "nm"
    assert np.isclose(reloaded.vpcf_results["pixel_to_nm"], 0.5)
    assert np.isclose(reloaded.vpcf_results["x_axis"][0], -7.0)
    assert "x_nm" in reloaded.vpcf_results["points_used"].columns
    assert "x_px" not in reloaded.vpcf_results["points_used"].columns
    assert "center_x_nm" in reloaded.vpcf_results["center_summary"].columns
    assert "center_x_px" not in reloaded.vpcf_results["center_summary"].columns
    assert np.isclose(reloaded.vpcf_results["config"]["r_max_nm"], 6.0)
    assert np.allclose(reloaded.vpcf_results["example_vectors"][0], [5.0, 0.0])


def test_points_for_vpcf_explains_when_keep_filter_removes_everything() -> None:
    session = AnalysisSession(name="all_dropped")
    session.curated_points = pd.DataFrame(
        {
            "atom_id": [1, 2],
            "x_px": [5.0, 10.0],
            "y_px": [5.0, 10.0],
            "keep": [False, False],
        }
    )

    with pytest.raises(ValueError, match="keep == True rows: 0"):
        points_for_vpcf(session, use_keep=True)


def test_session_vpcf_converts_angstrom_calibration_to_nm() -> None:
    session = AnalysisSession(
        name="angstrom_vpcf",
        pixel_calibration=PixelCalibration(size=2.0, unit="A", source="unit_test"),
    )
    session.set_stage("curated")
    session.curated_points = pd.DataFrame(
        {
            "atom_id": [0, 1],
            "x_px": [0.0, 10.0],
            "y_px": [0.0, 0.0],
            "keep": [True, True],
        }
    )

    session = compute_session_vpcf(
        session,
        VPCFConfig(r_max_px=12.0, extent_px=12.0, grid_size=17),
        center_atom_id=0,
    )

    assert np.isclose(session.vpcf_results["pixel_to_nm"], 0.2)
    assert np.isclose(session.vpcf_results["config"]["r_max_nm"], 2.4)
    assert np.allclose(session.vpcf_results["example_vectors"], [[2.0, 0.0]])


def test_session_vpcf_requires_calibration_for_nm_output() -> None:
    session = AnalysisSession(name="uncalibrated_vpcf")
    session.set_stage("curated")
    session.curated_points = pd.DataFrame(
        {
            "atom_id": [0, 1],
            "x_px": [0.0, 10.0],
            "y_px": [0.0, 0.0],
            "keep": [True, True],
        }
    )

    with pytest.raises(ValueError, match="requires calibrated pixel size"):
        compute_session_vpcf(session, VPCFConfig(r_max_px=12.0))
