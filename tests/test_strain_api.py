from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from em_atom_workbench import (
    AnalysisSession,
    LocalAffineStrainConfig,
    ReferenceLattice,
    ReferenceLatticeConfig,
    build_reference_lattice,
    compute_local_affine_strain,
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
