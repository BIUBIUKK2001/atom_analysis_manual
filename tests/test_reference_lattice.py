from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from em_atom_workbench import AnalysisSession, PixelCalibration, ReferenceLatticeConfig, build_reference_lattice
from em_atom_workbench.reference import _extract_xy_nm


def _points(count: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "atom_id": np.arange(count, dtype=int),
            "x_px": np.arange(count, dtype=float) * 2.0,
            "y_px": np.arange(count, dtype=float) * 3.0 + 1.0,
        }
    )


def test_manual_reference_lattice_defaults_to_pixel_coordinates() -> None:
    session = AnalysisSession(name="manual_px")
    session.curated_points = _points(5)
    config = ReferenceLatticeConfig(
        manual_basis=((2.0, 0.0), (0.0, 3.0)),
        min_points=3,
    )

    build_reference_lattice(session, config)

    lattice = session.reference_lattice["default"]
    assert lattice.unit == "px"
    assert np.allclose(lattice.basis, [[2.0, 0.0], [0.0, 3.0]])
    assert np.allclose(lattice.origin, [4.0, 7.0])
    assert lattice.metadata["n_points_used"] == 5
    assert lattice.metadata["coordinate_unit"] == "px"


def test_manual_reference_lattice_uses_calibrated_nm_when_explicit() -> None:
    session = AnalysisSession(name="manual_calibrated")
    session.pixel_calibration = PixelCalibration(size=0.2, unit="nm", source="test")
    session.curated_points = _points(5)
    config = ReferenceLatticeConfig(
        coordinate_unit="calibrated",
        manual_basis=((0.4, 0.0), (0.0, 0.6)),
        min_points=3,
    )

    build_reference_lattice(session, config)

    lattice = session.reference_lattice["default"]
    assert lattice.unit == "nm"
    assert np.allclose(lattice.basis, [[0.4, 0.0], [0.0, 0.6]])
    assert np.allclose(lattice.origin, [0.8, 1.4])
    assert np.allclose(lattice.basis_nm, lattice.basis)
    assert np.allclose(lattice.origin_nm, lattice.origin)


def test_calibrated_coordinates_require_nm_columns_or_valid_calibration() -> None:
    session = AnalysisSession(name="missing_calibration")
    session.curated_points = _points(5)
    config = ReferenceLatticeConfig(
        coordinate_unit="calibrated",
        manual_basis=((1.0, 0.0), (0.0, 1.0)),
        min_points=3,
    )

    with pytest.raises(ValueError, match="像素标定"):
        build_reference_lattice(session, config)


def test_role_filter_uses_column_role_and_atom_role_alias() -> None:
    session = AnalysisSession(name="role_filter")
    points = _points(6)
    points["column_role"] = ["Hf", "O", "Hf", "O", "Hf", "O"]
    session.curated_points = points
    config = ReferenceLatticeConfig(
        atom_role="Hf",
        manual_basis=((2.0, 0.0), (0.0, 3.0)),
        min_points=3,
    )

    build_reference_lattice(session, config)

    lattice = session.reference_lattice["default"]
    assert lattice.role_filter == "Hf"
    assert lattice.metadata["atom_role"] == "Hf"
    assert lattice.metadata["n_points_used"] == 3
    assert np.allclose(lattice.origin, [4.0, 7.0])


def test_keep_filter_raises_when_too_few_points_remain() -> None:
    session = AnalysisSession(name="too_few_after_keep")
    points = _points(5)
    points["keep"] = [True, False, True, False, False]
    session.curated_points = points
    config = ReferenceLatticeConfig(
        manual_basis=((2.0, 0.0), (0.0, 3.0)),
        min_points=3,
    )

    with pytest.raises(ValueError, match="至少 3"):
        build_reference_lattice(session, config)


def test_candidate_source_can_build_reference_lattice() -> None:
    session = AnalysisSession(name="candidate_source")
    session.candidate_points = _points(4)
    config = ReferenceLatticeConfig(
        source="candidate",
        manual_basis=((2.0, 0.0), (0.0, 3.0)),
        min_points=3,
    )

    build_reference_lattice(session, config, key="candidate")

    lattice = session.reference_lattice["candidate"]
    assert lattice.unit == "px"
    assert lattice.metadata["source"] == "candidate"
    assert lattice.metadata["n_points_used"] == 4


def test_global_median_uses_neighbor_graph_basis_table() -> None:
    session = AnalysisSession(name="global_median")
    session.curated_points = _points(5)
    session.neighbor_graph = {
        "basis_table": pd.DataFrame(
            {
                "atom_id": [0, 1, 2, 3, 4],
                "basis_a_x": [2.0, 2.0, 2.0, 2.0, 2.0],
                "basis_a_y": [0.0, 0.0, 0.0, 0.0, 0.0],
                "basis_b_x": [0.0, 0.0, 0.0, 0.0, 0.0],
                "basis_b_y": [3.0, 3.0, 3.0, 3.0, 3.0],
            }
        )
    }
    config = ReferenceLatticeConfig(mode="global_median", min_points=3)

    build_reference_lattice(session, config)

    lattice = session.reference_lattice["default"]
    assert lattice.mode == "median_local_basis"
    assert lattice.unit == "px"
    assert np.allclose(lattice.basis, [[2.0, 0.0], [0.0, 3.0]])


def test_extract_xy_nm_preserves_flag_columns_without_qc_flag() -> None:
    session = AnalysisSession(name="flag_semantics")
    table = pd.DataFrame(
        {
            "atom_id": [0, 1, 2],
            "x_nm": [0.0, 1.0, 2.0],
            "y_nm": [0.0, 1.5, 3.0],
            "column_role": ["Hf", "O", "Hf"],
            "keep": [True, True, False],
            "flag_edge": [False, True, False],
        }
    )

    extracted = _extract_xy_nm(session, table, source="curated")

    assert "flag_edge" in extracted.columns
    assert "keep" in extracted.columns
    assert "column_role" in extracted.columns
    assert "qc_flag" not in extracted.columns
