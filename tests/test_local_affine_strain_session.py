from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from em_atom_workbench import (
    AnalysisSession,
    LocalAffineStrainConfig,
    PixelCalibration,
    ReferenceLatticeConfig,
    build_reference_lattice,
    compute_local_affine_strain,
)


def _grid_points(
    nx: int = 7,
    ny: int = 7,
    F: np.ndarray | None = None,
    *,
    noise_sigma: float = 0.0,
    rng_seed: int = 0,
    missing_atom_ids: set[int] | None = None,
) -> pd.DataFrame:
    F = np.eye(2) if F is None else np.asarray(F, dtype=float)
    missing_atom_ids = set(missing_atom_ids or set())
    rng = np.random.default_rng(rng_seed)
    records = []
    for iy in range(ny):
        for ix in range(nx):
            atom_id = iy * nx + ix
            if atom_id in missing_atom_ids:
                continue
            xy = F @ np.array([float(ix), float(iy)])
            if noise_sigma > 0.0:
                xy = xy + rng.normal(scale=noise_sigma, size=2)
            records.append({"atom_id": atom_id, "x_px": xy[0], "y_px": xy[1]})
    return pd.DataFrame(records)


def _interior_atom_ids(nx: int, ny: int) -> list[int]:
    return [iy * nx + ix for iy in range(1, ny - 1) for ix in range(1, nx - 1)]


def _build_px_reference(session: AnalysisSession) -> None:
    build_reference_lattice(
        session,
        ReferenceLatticeConfig(
            manual_basis=((1.0, 0.0), (0.0, 1.0)),
            min_points=1,
        ),
    )


def _run_default_strain(session: AnalysisSession, **overrides: object) -> AnalysisSession:
    config_values = {
        "k_neighbors": 8,
        "min_pairs": 6,
        "neighbor_shells": 1,
        "pair_assignment_tolerance": 0.35,
        "max_condition_number": 50.0,
    }
    config_values.update(overrides)
    return compute_local_affine_strain(session, LocalAffineStrainConfig(**config_values))


def _ok_interior(session: AnalysisSession, nx: int = 7, ny: int = 7) -> pd.DataFrame:
    table = session.strain_table.set_index("atom_id")
    interior = table.loc[_interior_atom_ids(nx, ny)]
    return interior[interior["qc_flag"] == "ok"]


def test_local_affine_strain_identity_square_lattice_px() -> None:
    session = AnalysisSession(name="strain_identity")
    session.curated_points = _grid_points()
    _build_px_reference(session)

    _run_default_strain(session)

    table = session.strain_table
    interior = _ok_interior(session)
    assert len(interior) == 25
    assert {"x_px", "y_px", "x", "y", "coordinate_unit"}.issubset(table.columns)
    assert "x_nm" not in table.columns
    assert "local_a" not in table.columns
    assert "local_b" not in table.columns
    assert set(interior["coordinate_unit"]) == {"px"}
    assert np.allclose(interior["eps_xx"], 0.0, atol=1e-12)
    assert np.allclose(interior["eps_yy"], 0.0, atol=1e-12)
    assert np.allclose(interior["eps_xy"], 0.0, atol=1e-12)
    assert np.allclose(interior["rotation_rad"], 0.0, atol=1e-12)
    assert np.allclose(interior["F_xx"], 1.0, atol=1e-12)
    assert np.allclose(interior["F_yy"], 1.0, atol=1e-12)
    assert np.allclose(interior["F_xy"], 0.0, atol=1e-12)
    assert np.allclose(interior["F_yx"], 0.0, atol=1e-12)
    assert session.current_stage == "strain"
    assert session.provenance[-1]["parameters"]["k_neighbors"] == 8


def test_local_affine_strain_uniform_expansion() -> None:
    F = np.array([[1.02, 0.0], [0.0, 1.02]])
    session = AnalysisSession(name="strain_expansion")
    session.curated_points = _grid_points(F=F)
    _build_px_reference(session)

    _run_default_strain(session)

    interior = _ok_interior(session)
    assert np.allclose(interior["eps_xx"], 0.02, atol=1e-12)
    assert np.allclose(interior["eps_yy"], 0.02, atol=1e-12)
    assert np.allclose(interior["F_xx"], 1.02, atol=1e-12)
    assert np.allclose(interior["F_yy"], 1.02, atol=1e-12)
    assert np.allclose(interior["local_a_length"], 1.02, atol=1e-12)
    assert np.allclose(interior["local_b_length"], 1.02, atol=1e-12)


def test_local_affine_strain_anisotropic_shear() -> None:
    F = np.array([[1.02, 0.03], [-0.01, 0.98]])
    session = AnalysisSession(name="strain_shear")
    session.curated_points = _grid_points(F=F)
    _build_px_reference(session)

    _run_default_strain(session)

    interior = _ok_interior(session)
    assert np.allclose(interior["eps_xx"], 0.02, atol=1e-12)
    assert np.allclose(interior["eps_yy"], -0.02, atol=1e-12)
    assert np.allclose(interior["eps_xy"], 0.01, atol=1e-12)
    assert np.allclose(interior["rotation_rad"], -0.02, atol=1e-12)
    assert np.allclose(interior["F_xy"], 0.03, atol=1e-12)
    assert np.allclose(interior["F_yx"], -0.01, atol=1e-12)


def test_local_affine_strain_with_random_coordinate_noise_is_finite() -> None:
    session = AnalysisSession(name="strain_noise")
    session.curated_points = _grid_points(noise_sigma=0.004, rng_seed=7)
    _build_px_reference(session)

    _run_default_strain(session, pair_assignment_tolerance=0.45)

    ok_rows = session.strain_table[session.strain_table["qc_flag"] == "ok"]
    assert not ok_rows.empty
    assert np.isfinite(ok_rows["affine_residual"]).all()
    assert abs(float(ok_rows["eps_xx"].median())) < 0.03
    assert abs(float(ok_rows["eps_yy"].median())) < 0.03


def test_local_affine_strain_with_missing_atoms_keeps_failed_rows() -> None:
    nx = 7
    ny = 7
    center = 3 * nx + 3
    missing = {3 * nx + 2, 3 * nx + 4, 2 * nx + 3, 4 * nx + 3}
    session = AnalysisSession(name="strain_missing_atoms")
    session.curated_points = _grid_points(nx=nx, ny=ny, missing_atom_ids=missing)
    _build_px_reference(session)

    _run_default_strain(session)

    table = session.strain_table.set_index("atom_id")
    assert len(table) == len(session.curated_points)
    assert table.loc[center, "qc_flag"] == "too_few_pairs"
    assert table["qc_flag"].eq("ok").any()


def test_local_affine_strain_too_few_pairs_marks_nan_results() -> None:
    session = AnalysisSession(name="strain_too_few")
    session.curated_points = _grid_points(nx=3, ny=3)
    _build_px_reference(session)

    _run_default_strain(session, k_neighbors=3, min_pairs=6)

    table = session.strain_table
    assert set(table["qc_flag"]) == {"too_few_pairs"}
    assert table["eps_xx"].isna().all()
    assert table["local_a_length"].isna().all()
    assert table["n_pairs"].max() < 6


def test_local_affine_strain_requires_reference_lattice() -> None:
    session = AnalysisSession(name="strain_missing_reference")
    session.curated_points = _grid_points()

    with pytest.raises(ValueError, match="build_reference_lattice"):
        _run_default_strain(session)


def test_local_affine_strain_role_and_keep_filtering() -> None:
    session = AnalysisSession(name="strain_role_keep")
    points = _grid_points()
    points["column_role"] = "Hf"
    points.loc[points["atom_id"].isin([0, 1]), "column_role"] = "O"
    points["keep"] = True
    points.loc[points["atom_id"] == 2, "keep"] = False
    session.curated_points = points
    _build_px_reference(session)

    _run_default_strain(session, atom_role="Hf", min_pairs=4)

    table = session.strain_table
    assert set(table["role"]) == {"Hf"}
    assert not table["atom_id"].isin([0, 1, 2]).any()


def test_local_affine_strain_calibrated_nm_path_preserves_px_overlay_columns() -> None:
    session = AnalysisSession(name="strain_nm")
    session.pixel_calibration = PixelCalibration(size=0.5, unit="nm", source="test")
    session.curated_points = _grid_points()
    build_reference_lattice(
        session,
        ReferenceLatticeConfig(
            coordinate_unit="calibrated",
            manual_basis=((0.5, 0.0), (0.0, 0.5)),
            min_points=1,
        ),
    )

    _run_default_strain(session)

    table = session.strain_table
    interior = _ok_interior(session)
    assert {"x_px", "y_px", "x_nm", "y_nm", "local_a_length_nm", "local_b_length_nm", "affine_residual_nm"}.issubset(
        table.columns
    )
    assert set(interior["coordinate_unit"]) == {"nm"}
    assert np.allclose(interior["x_nm"], interior["x_px"] * 0.5)
    assert np.allclose(interior["eps_xx"], 0.0, atol=1e-12)
    assert np.allclose(interior["local_a_length_nm"], 0.5, atol=1e-12)
    assert np.allclose(interior["local_b_length_nm"], 0.5, atol=1e-12)
