from __future__ import annotations

import numpy as np
import pytest

from em_atom_workbench.strain import (
    _assign_reference_pair_vectors,
    _fit_affine_gradient,
    _local_lattice_from_F,
    _strain_from_deformation_gradient,
)


def _vectors_from_indices(basis: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return (basis @ np.asarray(indices, dtype=float).T).T


def _apply_F(F: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return (F @ vectors.T).T


def test_identity_assignment_fit_and_strain_are_zero() -> None:
    basis = np.eye(2)
    indices = np.array([[1, 0], [0, 1]], dtype=int)
    obs_vectors = _vectors_from_indices(basis, indices)

    ref_vectors, obs_vectors_kept, miller_like_indices, assignment_residuals = _assign_reference_pair_vectors(
        obs_vectors,
        basis,
        neighbor_shells=1,
        tolerance=0.05,
    )
    F, condition_number, residual_rms = _fit_affine_gradient(ref_vectors, obs_vectors_kept)
    epsilon, rotation_rad = _strain_from_deformation_gradient(F)

    assert miller_like_indices.shape == (2, 2)
    assert np.allclose(assignment_residuals, 0.0)
    assert np.allclose(F, np.eye(2))
    assert np.isclose(condition_number, 1.0)
    assert np.isclose(residual_rms, 0.0)
    assert np.allclose(epsilon, np.zeros((2, 2)))
    assert np.isclose(rotation_rad, 0.0)


def test_uniform_expansion_small_strain() -> None:
    ref_vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
    expected_F = np.array([[1.02, 0.0], [0.0, 1.02]])
    obs_vectors = _apply_F(expected_F, ref_vectors)

    F, _, residual_rms = _fit_affine_gradient(ref_vectors, obs_vectors)
    epsilon, rotation_rad = _strain_from_deformation_gradient(F)

    assert np.allclose(F, expected_F)
    assert np.isclose(residual_rms, 0.0)
    assert np.isclose(epsilon[0, 0], 0.02)
    assert np.isclose(epsilon[1, 1], 0.02)
    assert np.isclose(epsilon[0, 1], 0.0)
    assert np.isclose(rotation_rad, 0.0)


def test_anisotropic_strain_shear_and_rotation() -> None:
    F = np.array([[1.02, 0.03], [-0.01, 0.98]])

    epsilon, rotation_rad = _strain_from_deformation_gradient(F)

    assert np.isclose(epsilon[0, 0], 0.02)
    assert np.isclose(epsilon[1, 1], -0.02)
    assert np.isclose(epsilon[0, 1], 0.01)
    assert np.isclose(epsilon[1, 0], 0.01)
    assert np.isclose(rotation_rad, -0.02)


def test_pure_rotation_has_near_zero_small_strain_and_expected_rotation() -> None:
    theta = 1.0e-4
    F = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    epsilon, rotation_rad = _strain_from_deformation_gradient(F)

    assert np.allclose(epsilon, np.zeros((2, 2)), atol=1e-8)
    assert np.isclose(rotation_rad, theta, atol=1e-8)


def test_duplicate_assignment_keeps_smallest_residual() -> None:
    basis = np.eye(2)
    obs_vectors = np.array(
        [
            [1.20, 0.00],
            [1.01, 0.00],
            [0.00, 0.98],
            [0.00, 1.18],
        ]
    )

    ref_vectors, obs_vectors_kept, miller_like_indices, assignment_residuals = _assign_reference_pair_vectors(
        obs_vectors,
        basis,
        neighbor_shells=1,
        tolerance=0.25,
    )

    assert np.allclose(ref_vectors, [[0.0, 1.0], [1.0, 0.0]])
    assert np.array_equal(miller_like_indices, [[0, 1], [1, 0]])
    assert np.allclose(obs_vectors_kept, [[0.00, 0.98], [1.01, 0.00]])
    assert np.allclose(assignment_residuals, [0.02, 0.01])


def test_assignment_returns_empty_arrays_when_no_pairs_pass() -> None:
    ref_vectors, obs_vectors_kept, miller_like_indices, assignment_residuals = _assign_reference_pair_vectors(
        np.array([[0.01, 0.0], [3.0, 0.0]]),
        np.eye(2),
        neighbor_shells=1,
        tolerance=0.05,
    )

    assert ref_vectors.shape == (0, 2)
    assert obs_vectors_kept.shape == (0, 2)
    assert miller_like_indices.shape == (0, 2)
    assert assignment_residuals.shape == (0,)


def test_non_orthogonal_basis_recovers_known_gradient() -> None:
    basis = np.array([[1.0, 0.35], [0.1, 0.9]])
    expected_F = np.array([[1.03, 0.02], [-0.01, 0.97]])
    indices = np.array([[1, 0], [0, 1], [1, 1], [-1, 0], [0, -1], [1, -1]], dtype=int)
    ref_seed = _vectors_from_indices(basis, indices)
    obs_vectors = _apply_F(expected_F, ref_seed)

    ref_vectors, obs_vectors_kept, _, _ = _assign_reference_pair_vectors(
        obs_vectors,
        basis,
        neighbor_shells=1,
        tolerance=0.2,
    )
    F, _, residual_rms = _fit_affine_gradient(ref_vectors, obs_vectors_kept)

    assert np.allclose(F, expected_F, atol=1e-12)
    assert np.isclose(residual_rms, 0.0, atol=1e-12)


def test_green_strain_formula() -> None:
    F = np.array([[1.02, 0.0], [0.0, 0.98]])

    epsilon, rotation_rad = _strain_from_deformation_gradient(F, strain_type="green")

    assert np.allclose(epsilon, 0.5 * (F.T @ F - np.eye(2)))
    assert np.isclose(rotation_rad, 0.0)


def test_weighted_residual_rms_uses_weighted_definition() -> None:
    ref_vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
    obs_vectors = ref_vectors.copy()
    obs_vectors[2] += np.array([0.1, -0.2])
    weights = np.array([1.0, 2.0, 3.0, 4.0])

    F, _, residual_rms = _fit_affine_gradient(ref_vectors, obs_vectors, weights=weights)

    residual = obs_vectors - _apply_F(F, ref_vectors)
    expected = np.sqrt(np.sum(weights * np.sum(residual**2, axis=1)) / np.sum(weights))
    assert np.isclose(residual_rms, expected)


@pytest.mark.parametrize(
    "weights",
    [
        np.array([1.0, 2.0]),
        np.array([1.0, np.nan, 2.0]),
        np.array([1.0, -1.0, 2.0]),
        np.array([0.0, 0.0, 0.0]),
    ],
)
def test_fit_rejects_invalid_weights(weights: np.ndarray) -> None:
    ref_vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    obs_vectors = ref_vectors.copy()

    with pytest.raises(ValueError, match="weights"):
        _fit_affine_gradient(ref_vectors, obs_vectors, weights=weights)


@pytest.mark.parametrize(
    ("ref_vectors", "obs_vectors"),
    [
        (np.empty((0, 2)), np.empty((0, 2))),
        (np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]])),
        (np.array([[1.0, 0.0], [2.0, 0.0]]), np.array([[1.0, 0.0], [2.0, 0.0]])),
    ],
)
def test_fit_rejects_insufficient_or_collinear_vectors(ref_vectors: np.ndarray, obs_vectors: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _fit_affine_gradient(ref_vectors, obs_vectors)


@pytest.mark.parametrize("neighbor_shells", [0, 1.0, True])
def test_assignment_rejects_invalid_neighbor_shells(neighbor_shells: object) -> None:
    with pytest.raises(ValueError, match="neighbor_shells"):
        _assign_reference_pair_vectors(
            np.array([[1.0, 0.0]]),
            np.eye(2),
            neighbor_shells=neighbor_shells,
            tolerance=0.1,
        )


@pytest.mark.parametrize("tolerance", [0.0, -0.1, np.inf, np.nan])
def test_assignment_rejects_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        _assign_reference_pair_vectors(
            np.array([[1.0, 0.0]]),
            np.eye(2),
            neighbor_shells=1,
            tolerance=tolerance,
        )


def test_local_lattice_from_gradient() -> None:
    basis = np.array([[2.0, 0.5], [0.0, 1.5]])
    F = np.array([[1.02, 0.03], [-0.01, 0.98]])

    local_a, local_b, local_gamma_deg = _local_lattice_from_F(F, basis)
    local_basis = F @ basis
    expected_gamma = np.degrees(
        np.arccos(
            np.clip(
                np.dot(local_basis[:, 0], local_basis[:, 1])
                / (np.linalg.norm(local_basis[:, 0]) * np.linalg.norm(local_basis[:, 1])),
                -1.0,
                1.0,
            )
        )
    )

    assert np.allclose(local_a, local_basis[:, 0])
    assert np.allclose(local_b, local_basis[:, 1])
    assert np.isclose(local_gamma_deg, expected_gamma)
