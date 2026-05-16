from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from em_atom_workbench import (
    AnalysisSession,
    LocalAffineStrainConfig,
    ReferenceLattice,
    ReferenceLatticeConfig,
    assign_lattice_indices,
    build_complete_cells,
    build_reference_lattice,
    compute_cell_strain,
    compute_local_affine_strain,
    resolve_anchor_period_references,
)


def test_strain_api_imports_from_package() -> None:
    assert ReferenceLatticeConfig is not None
    assert ReferenceLattice is not None
    assert LocalAffineStrainConfig is not None
    assert callable(build_reference_lattice)
    assert callable(compute_local_affine_strain)


def test_reference_lattice_config_default_is_constructible() -> None:
    config = ReferenceLatticeConfig()

    assert config.mode == "manual_basis"
    assert config.manual_basis is None
    assert config.coordinate_unit == "px"


def test_reference_lattice_config_validates_invalid_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        ReferenceLatticeConfig(mode="roi_median")


def test_reference_lattice_config_validates_invalid_source() -> None:
    with pytest.raises(ValueError, match="source"):
        ReferenceLatticeConfig(source="atoms")


def test_reference_lattice_config_validates_invalid_coordinate_unit() -> None:
    with pytest.raises(ValueError, match="coordinate_unit"):
        ReferenceLatticeConfig(coordinate_unit="nm")


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (-1.0, 95.0),
        (5.0, 101.0),
        (95.0, 5.0),
        (5.0, 5.0),
        (np.nan, 95.0),
        (5.0, np.inf),
    ],
)
def test_reference_lattice_config_validates_percentile_ranges(low: float, high: float) -> None:
    with pytest.raises(ValueError, match="robust_percentile"):
        ReferenceLatticeConfig(robust_percentile_low=low, robust_percentile_high=high)


def test_reference_lattice_config_rejects_singular_manual_basis() -> None:
    with pytest.raises(ValueError, match="manual_basis"):
        ReferenceLatticeConfig(manual_basis=((1.0, 0.0), (2.0, 0.0)))


def test_reference_lattice_stores_basis_and_origin() -> None:
    lattice = ReferenceLattice(
        basis=np.array([[1.0, 0.2], [0.0, 2.0]]),
        origin=np.array([3.0, 4.0]),
        unit="px",
        role_filter="light_atom",
        mode="manual_basis",
        metadata={"source": "manual"},
    )

    assert lattice.basis.shape == (2, 2)
    assert lattice.origin.shape == (2,)
    assert np.allclose(lattice.basis, [[1.0, 0.2], [0.0, 2.0]])
    assert np.allclose(lattice.origin, [3.0, 4.0])
    assert lattice.unit == "px"


@pytest.mark.parametrize("strain_type", ["small", "green"])
def test_local_affine_strain_config_accepts_valid_strain_types(strain_type: str) -> None:
    assert LocalAffineStrainConfig(strain_type=strain_type).strain_type == strain_type


def test_local_affine_strain_config_validates_invalid_strain_type() -> None:
    with pytest.raises(ValueError, match="strain_type"):
        LocalAffineStrainConfig(strain_type="absolute")


def test_local_affine_strain_config_validates_invalid_output_frame() -> None:
    with pytest.raises(ValueError, match="output_frame"):
        LocalAffineStrainConfig(output_frame="lattice")


def test_local_affine_strain_config_validates_invalid_source() -> None:
    with pytest.raises(ValueError, match="source"):
        LocalAffineStrainConfig(source="atoms")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("neighbor_shells", 0),
        ("k_neighbors", 0),
        ("min_pairs", 0),
        ("pair_assignment_tolerance", 0.0),
        ("pair_assignment_tolerance", np.nan),
        ("max_condition_number", 0.0),
        ("max_condition_number", np.inf),
        ("weight_power", -1.0),
        ("weight_power", np.nan),
    ],
)
def test_local_affine_strain_config_validates_numeric_ranges(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        LocalAffineStrainConfig(**{field: value})


def test_build_reference_lattice_requires_manual_basis_for_manual_mode() -> None:
    session = AnalysisSession(name="missing_reference_basis")

    with pytest.raises(ValueError, match="manual_basis"):
        build_reference_lattice(session, ReferenceLatticeConfig())


def test_build_reference_lattice_manual_basis_stores_default_key() -> None:
    session = AnalysisSession(name="manual_reference_basis")
    session.curated_points = pd.DataFrame(
        {
            "atom_id": np.arange(10, dtype=int),
            "x_px": np.arange(10, dtype=float),
            "y_px": np.arange(10, dtype=float),
            "column_role": ["light_atom"] * 10,
        }
    )
    config = ReferenceLatticeConfig(
        manual_basis=((10.0, 0.0), (0.5, 9.0)),
        manual_origin=(2.0, 3.0),
        role_filter="light_atom",
    )

    result = build_reference_lattice(session, config)

    assert result is session
    assert "default" in session.reference_lattice
    lattice = session.reference_lattice["default"]
    assert isinstance(lattice, ReferenceLattice)
    assert np.allclose(lattice.basis, [[10.0, 0.5], [0.0, 9.0]])
    assert np.allclose(lattice.origin, [2.0, 3.0])
    assert lattice.role_filter == "light_atom"


def test_build_reference_lattice_median_mode_requires_existing_local_basis() -> None:
    session = AnalysisSession(name="median_reference_basis")
    session.curated_points = pd.DataFrame(
        {
            "atom_id": np.arange(10, dtype=int),
            "x_px": np.arange(10, dtype=float),
            "y_px": np.arange(10, dtype=float),
        }
    )
    config = ReferenceLatticeConfig(mode="median_local_basis")

    with pytest.raises(ValueError, match="compute_local_metrics"):
        build_reference_lattice(session, config)


def test_compute_local_affine_strain_requires_reference_lattice() -> None:
    session = AnalysisSession(name="strain_placeholder")

    with pytest.raises(ValueError, match="build_reference_lattice"):
        compute_local_affine_strain(session, LocalAffineStrainConfig())


def test_anchor_period_reference_requires_same_class_summary() -> None:
    summary = pd.DataFrame(
        {
            "roi_id": ["roi_0", "roi_0"],
            "direction": ["a", "b"],
            "class_selection": ["class_id:1", "class_id:1"],
            "length_median_px": [10.0, 12.0],
        }
    )

    resolved = resolve_anchor_period_references(summary, {"roi_0": 0})

    assert bool(resolved.iloc[0]["valid"]) is False
    assert resolved.iloc[0]["invalid_reason"] == "missing_anchor_period_reference"


def test_complete_cell_geometry_uses_cross_product_area() -> None:
    anchors = pd.DataFrame(
        {
            "point_id": ["p00", "p10", "p01", "p11"],
            "roi_id": ["global"] * 4,
            "roi_name": ["global"] * 4,
            "x_px": [0.0, 10.0, 2.0, 12.0],
            "y_px": [0.0, 0.0, 8.0, 8.0],
        }
    )

    indexed = assign_lattice_indices(
        anchors,
        a_ref_px=10.0,
        b_ref_px=np.hypot(2.0, 8.0),
        unit_a=(1.0, 0.0),
        unit_b=(2.0, 8.0),
        origin_point_id="p00",
        max_residual_fraction=0.1,
    )
    cells = build_complete_cells(indexed)
    strained, references = compute_cell_strain(cells)

    valid = strained.loc[strained["valid"].astype(bool)].iloc[0]
    assert valid["area_local"] == pytest.approx(80.0)
    assert references.iloc[0]["area_ref"] == pytest.approx(80.0)
