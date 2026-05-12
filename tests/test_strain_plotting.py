from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from em_atom_workbench.plotting import plot_strain_component_map, plot_strain_qc_map
from em_atom_workbench.session import AnalysisSession


def _synthetic_strain_session() -> AnalysisSession:
    session = AnalysisSession(name="strain_plotting", raw_image=np.zeros((32, 32), dtype=float))
    session.strain_table = pd.DataFrame(
        {
            "atom_id": [0, 1, 2, 3],
            "x_px": [8.0, 16.0, 8.0, 16.0],
            "y_px": [8.0, 8.0, 16.0, 16.0],
            "x": [8.0, 16.0, 8.0, 16.0],
            "y": [8.0, 8.0, 16.0, 16.0],
            "role": ["Hf", "Hf", "O", "O"],
            "qc_flag": ["ok", "ok", "too_few_pairs", "fit_failed"],
            "eps_xx": [0.01, 0.02, np.nan, np.nan],
            "eps_yy": [-0.01, -0.02, np.nan, np.nan],
            "eps_xy": [0.003, 0.004, np.nan, np.nan],
            "rotation_deg": [0.1, 0.2, np.nan, np.nan],
            "principal_eps_1": [0.02, 0.03, np.nan, np.nan],
            "principal_eps_2": [-0.02, -0.03, np.nan, np.nan],
            "dilatation": [0.0, 0.0, np.nan, np.nan],
            "shear_magnitude": [0.02, 0.04, np.nan, np.nan],
            "affine_residual": [0.05, 0.06, np.nan, np.nan],
            "local_a_length": [10.1, 10.2, np.nan, np.nan],
            "local_b_length": [9.9, 9.8, np.nan, np.nan],
            "local_gamma_deg": [90.0, 89.5, np.nan, np.nan],
        }
    )
    return session


def test_plot_strain_component_map_returns_fig_ax() -> None:
    session = _synthetic_strain_session()

    fig, ax = plot_strain_component_map(session, "eps_xx")

    assert fig is ax.figure
    assert ax.get_title()
    plt.close(fig)


def test_plot_strain_component_map_supports_qc_and_role_filtering() -> None:
    session = _synthetic_strain_session()

    fig, ax = plot_strain_component_map(session, "eps_yy", atom_role="Hf", qc_only=True, image_source=None)

    assert fig is ax.figure
    assert len(ax.collections) == 1
    plt.close(fig)


def test_plot_strain_qc_map_returns_fig_ax() -> None:
    session = _synthetic_strain_session()

    fig, ax = plot_strain_qc_map(session)

    assert fig is ax.figure
    assert ax.get_legend() is not None
    plt.close(fig)


def test_plot_strain_component_map_rejects_invalid_component() -> None:
    session = _synthetic_strain_session()

    with pytest.raises(ValueError, match="不支持"):
        plot_strain_component_map(session, "absolute_strain")


def test_plot_strain_component_map_falls_back_for_affine_residual_nm_alias() -> None:
    session = _synthetic_strain_session()

    fig, ax = plot_strain_component_map(session, "affine_residual_nm")

    assert fig is ax.figure
    assert ax.collections
    plt.close(fig)


def test_plot_strain_component_map_falls_back_for_local_length_aliases() -> None:
    session = _synthetic_strain_session()

    fig_a, ax_a = plot_strain_component_map(session, "local_a_nm")
    fig_b, ax_b = plot_strain_component_map(session, "local_b_nm")

    assert fig_a is ax_a.figure
    assert fig_b is ax_b.figure
    plt.close(fig_a)
    plt.close(fig_b)


def test_plot_strain_component_map_requires_role_column_when_filtering() -> None:
    session = _synthetic_strain_session()
    session.strain_table = session.strain_table.drop(columns=["role"])

    with pytest.raises(ValueError, match="role"):
        plot_strain_component_map(session, "eps_xx", atom_role="Hf")
