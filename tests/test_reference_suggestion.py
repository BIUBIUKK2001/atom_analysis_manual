from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.quiver import Quiver
import numpy as np
import pandas as pd
import pytest

from em_atom_workbench import (
    AnalysisSession,
    ReferenceLatticeSuggestion,
    ReferenceLatticeSuggestionConfig,
    build_reference_lattice_from_suggestion,
    suggest_reference_lattices,
)
from em_atom_workbench.plotting import plot_reference_candidate_map


def _session_with_basis(points: pd.DataFrame, basis_rows: pd.DataFrame) -> AnalysisSession:
    session = AnalysisSession(name="reference_suggestion", raw_image=np.zeros((128, 256), dtype=float))
    session.curated_points = points.copy()
    session.neighbor_graph = {"basis_table": basis_rows.copy()}
    return session


def _basis_rows(atom_ids: list[int], basis: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "atom_id": atom_ids,
            "basis_a_x": float(basis[0, 0]),
            "basis_a_y": float(basis[1, 0]),
            "basis_b_x": float(basis[0, 1]),
            "basis_b_y": float(basis[1, 1]),
        }
    )


def _basis_rows_from_matrices(atom_ids: list[int], bases: list[np.ndarray]) -> pd.DataFrame:
    records = []
    for atom_id, basis in zip(atom_ids, bases):
        records.append(
            {
                "atom_id": atom_id,
                "basis_a_x": float(basis[0, 0]),
                "basis_a_y": float(basis[1, 0]),
                "basis_b_x": float(basis[0, 1]),
                "basis_b_y": float(basis[1, 1]),
            }
        )
    return pd.DataFrame(records)


def test_suggest_reference_lattices_single_domain_defaults_to_main_candidate() -> None:
    atom_ids = list(range(12))
    points = pd.DataFrame(
        {
            "atom_id": atom_ids,
            "x_px": np.linspace(0.0, 55.0, len(atom_ids)),
            "y_px": np.zeros(len(atom_ids)),
        }
    )
    basis = np.array([[10.0, 0.5], [0.0, 9.8]], dtype=float)
    session = _session_with_basis(points, _basis_rows(atom_ids, basis))

    suggestion = suggest_reference_lattices(
        session,
        ReferenceLatticeSuggestionConfig(n_candidates=3, min_points=4),
    )

    assert 1 <= len(suggestion.candidates) <= 3
    candidate = suggestion.candidates.iloc[0]
    recovered = np.array(
        [[candidate["basis_a_x"], candidate["basis_b_x"]], [candidate["basis_a_y"], candidate["basis_b_y"]]],
        dtype=float,
    )
    assert np.allclose(recovered, basis)
    assert suggestion.unit == "px"
    assert suggestion.metadata["basis_convention"] == "canonical_reduced_columns_are_a_b_vectors_in_xy"
    assert suggestion.metadata["cluster_feature_mode"] == "log_lengths_folded_gamma_orientation_axis"


def test_suggest_reference_lattices_folds_equivalent_basis_representations() -> None:
    atom_ids = list(range(16))
    points = pd.DataFrame(
        {
            "atom_id": atom_ids,
            "x_px": np.linspace(0.0, 75.0, len(atom_ids)),
            "y_px": np.zeros(len(atom_ids)),
        }
    )
    base = np.array([[10.0, 0.0], [0.0, 12.0]], dtype=float)
    variants = [
        base,
        -base,
        base[:, [1, 0]],
        np.column_stack([base[:, 0], base[:, 0] + base[:, 1]]),
    ]
    basis_rows = _basis_rows_from_matrices(atom_ids, [variants[index % len(variants)] for index in atom_ids])
    session = _session_with_basis(points, basis_rows)

    suggestion = suggest_reference_lattices(
        session,
        ReferenceLatticeSuggestionConfig(n_candidates=3, min_points=4),
    )

    assert len(suggestion.candidates) == 1
    candidate = suggestion.candidates.iloc[0]
    recovered = np.array(
        [[candidate["basis_a_x"], candidate["basis_b_x"]], [candidate["basis_a_y"], candidate["basis_b_y"]]],
        dtype=float,
    )
    assert np.allclose(recovered, base)
    assert set(suggestion.assignments["candidate_id"]) == {0}


def test_suggest_reference_lattices_separates_two_domains() -> None:
    atom_ids_a = list(range(18))
    atom_ids_b = list(range(18, 30))
    points = pd.DataFrame(
        {
            "atom_id": atom_ids_a + atom_ids_b,
            "x_px": np.r_[np.linspace(0.0, 80.0, len(atom_ids_a)), np.linspace(140.0, 220.0, len(atom_ids_b))],
            "y_px": np.zeros(30),
        }
    )
    basis_a = np.array([[10.0, 0.0], [0.0, 10.0]], dtype=float)
    basis_b = np.array([[13.0, 1.0], [0.5, 9.0]], dtype=float)
    basis_rows = pd.concat([_basis_rows(atom_ids_a, basis_a), _basis_rows(atom_ids_b, basis_b)], ignore_index=True)
    session = _session_with_basis(points, basis_rows)

    suggestion = suggest_reference_lattices(
        session,
        ReferenceLatticeSuggestionConfig(n_candidates=2, min_points=6),
    )

    assert len(suggestion.candidates) == 2
    assert suggestion.candidates.iloc[0]["n_points"] == len(atom_ids_a)
    lengths = sorted(np.round(suggestion.candidates["basis_a_length"].to_numpy(dtype=float), 1))
    assert lengths == [10.0, 13.0]
    assert set(suggestion.assignments["candidate_id"]) == {0, 1}


def test_build_reference_lattice_from_suggestion_stores_selected_candidate() -> None:
    atom_ids = list(range(8))
    points = pd.DataFrame({"atom_id": atom_ids, "x_px": np.arange(8.0), "y_px": np.arange(8.0)})
    basis = np.array([[9.0, 0.0], [1.0, 11.0]], dtype=float)
    session = _session_with_basis(points, _basis_rows(atom_ids, basis))
    suggestion = suggest_reference_lattices(session, ReferenceLatticeSuggestionConfig(n_candidates=1, min_points=4))

    build_reference_lattice_from_suggestion(session, suggestion, candidate_id=0)

    lattice = session.reference_lattice["default"]
    assert lattice.mode == "suggested_cluster"
    assert lattice.unit == "px"
    assert np.allclose(lattice.basis, basis)


def test_reference_suggestion_roi_uses_only_selected_area() -> None:
    atom_ids_a = list(range(8))
    atom_ids_b = list(range(8, 16))
    points = pd.DataFrame(
        {
            "atom_id": atom_ids_a + atom_ids_b,
            "x_px": np.r_[np.linspace(0.0, 70.0, 8), np.linspace(150.0, 220.0, 8)],
            "y_px": np.zeros(16),
        }
    )
    basis_a = np.array([[8.0, 0.0], [0.0, 8.0]], dtype=float)
    basis_b = np.array([[12.0, 1.0], [0.0, 9.0]], dtype=float)
    session = _session_with_basis(
        points,
        pd.concat([_basis_rows(atom_ids_a, basis_a), _basis_rows(atom_ids_b, basis_b)], ignore_index=True),
    )

    suggestion = suggest_reference_lattices(
        session,
        ReferenceLatticeSuggestionConfig(n_candidates=2, min_points=4, roi=(120.0, 240.0, -10.0, 10.0)),
    )

    candidate = suggestion.candidates.iloc[0]
    assert candidate["n_points"] == len(atom_ids_b)
    assert np.isclose(candidate["basis_a_length"], np.linalg.norm(basis_b[:, 0]))
    assert suggestion.assignments["x_px"].min() >= 120.0


def test_reference_suggestion_role_and_keep_filtering() -> None:
    points = pd.DataFrame(
        {
            "atom_id": [0, 1, 2, 3],
            "x_px": [0.0, 1.0, 2.0, 3.0],
            "y_px": [0.0, 0.0, 0.0, 0.0],
            "column_role": ["Hf", "Hf", "O", "Hf"],
            "keep": [True, True, True, False],
        }
    )
    basis = np.array([[10.0, 0.0], [0.0, 10.0]], dtype=float)
    session = _session_with_basis(points, _basis_rows([0, 1, 2, 3], basis))

    suggestion = suggest_reference_lattices(
        session,
        ReferenceLatticeSuggestionConfig(role_filter="Hf", use_keep=True, min_points=2),
    )

    assert suggestion.metadata["n_points_selected"] == 2
    assert set(suggestion.assignments["atom_id"]) == {0, 1}


def test_reference_suggestion_requires_local_basis_vectors() -> None:
    session = AnalysisSession(name="missing_basis")
    session.curated_points = pd.DataFrame({"atom_id": [0, 1], "x_px": [0.0, 1.0], "y_px": [0.0, 1.0]})

    with pytest.raises(ValueError, match="局域基矢量"):
        suggest_reference_lattices(session, ReferenceLatticeSuggestionConfig(min_points=2))


def test_plot_reference_candidate_map_returns_fig_ax() -> None:
    atom_ids = list(range(6))
    points = pd.DataFrame({"atom_id": atom_ids, "x_px": np.arange(6.0), "y_px": np.arange(6.0)})
    basis = np.array([[10.0, 0.0], [0.0, 10.0]], dtype=float)
    session = _session_with_basis(points, _basis_rows(atom_ids, basis))
    suggestion = suggest_reference_lattices(session, ReferenceLatticeSuggestionConfig(n_candidates=1, min_points=3))

    fig, ax = plot_reference_candidate_map(session, suggestion, selected_candidate_id=0)

    assert fig is ax.figure
    assert any(isinstance(collection, Quiver) for collection in ax.collections)
    plt.close(fig)


def test_plot_reference_candidate_map_rejects_empty_suggestion() -> None:
    session = AnalysisSession(name="empty_suggestion_plot")
    suggestion = ReferenceLatticeSuggestion(
        candidates=pd.DataFrame(),
        assignments=pd.DataFrame(),
        unit="px",
        coordinate_unit="px",
        source="curated",
    )

    with pytest.raises(ValueError, match="candidate"):
        plot_reference_candidate_map(session, suggestion)
