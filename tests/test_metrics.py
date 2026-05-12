import numpy as np
import pandas as pd

from em_atom_workbench.lattice import build_neighbor_graph
from em_atom_workbench.metrics import compute_local_metrics
from em_atom_workbench.session import AnalysisSession, LatticeConfig, MetricsConfig


def _make_regular_grid(nx=4, ny=4, spacing_x=10.2, spacing_y=9.8):
    records = []
    atom_id = 0
    for iy in range(ny):
        for ix in range(nx):
            records.append({"atom_id": atom_id, "x_px": ix * spacing_x, "y_px": iy * spacing_y})
            atom_id += 1
    return pd.DataFrame(records)


def test_compute_local_metrics_for_regular_grid():
    points = _make_regular_grid()
    session = AnalysisSession(name="synthetic_metrics", raw_image=np.zeros((64, 64), dtype=float))
    session.curated_points = points.copy()

    build_neighbor_graph(session, LatticeConfig(k_neighbors=6, min_basis_angle_deg=30.0))
    compute_local_metrics(
        session,
        MetricsConfig(reference_basis=((10.2, 0.0), (0.0, 9.8))),
    )

    metrics = session.local_metrics
    assert not metrics.empty
    assert np.isclose(metrics["nearest_neighbor_distance_px"].median(), 9.8, atol=0.5)
    assert np.isclose(metrics["basis_a_length_px"].median(), 10.2, atol=0.5) or np.isclose(metrics["basis_a_length_px"].median(), 9.8, atol=0.5)
    assert np.nanmax(np.abs(metrics["strain_exx"])) < 0.1
    assert np.nanmax(np.abs(metrics["strain_eyy"])) < 0.1


def test_compute_local_metrics_with_reference_displacement():
    reference = _make_regular_grid()
    displaced = reference.copy()
    displaced.loc[5, "x_px"] += 0.6
    displaced.loc[5, "y_px"] -= 0.4

    session = AnalysisSession(name="synthetic_displacement", raw_image=np.zeros((64, 64), dtype=float))
    session.curated_points = displaced
    build_neighbor_graph(session, LatticeConfig(k_neighbors=6))
    compute_local_metrics(
        session,
        MetricsConfig(reference_points=reference),
    )

    metrics = session.local_metrics.set_index("atom_id")
    assert np.isclose(metrics.loc[5, "reference_displacement_x_px"], 0.6, atol=1e-6)
    assert np.isclose(metrics.loc[5, "reference_displacement_y_px"], -0.4, atol=1e-6)

