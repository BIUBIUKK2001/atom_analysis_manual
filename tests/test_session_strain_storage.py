from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from em_atom_workbench import AnalysisSession, ReferenceLattice


def _reference_lattice(scale: float = 1.0) -> ReferenceLattice:
    return ReferenceLattice(
        basis=np.array([[scale, 0.0], [0.0, scale + 1.0]], dtype=float),
        origin=np.array([0.0, 0.0], dtype=float),
        unit="px",
        role_filter=None,
        mode="manual_basis",
        metadata={"source": "test"},
    )


@contextmanager
def _writable_temp_dir():
    root = Path(".runtime-tmp") / f"strain_storage_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_new_session_has_empty_strain_storage() -> None:
    session = AnalysisSession(name="empty_strain_storage")

    assert session.reference_lattice == {}
    assert session.strain_table.empty


def test_manifest_reports_strain_table_and_reference_lattice_keys() -> None:
    session = AnalysisSession(name="manifest_strain_storage")
    session.reference_lattice = {
        "z_reference": _reference_lattice(2.0),
        "a_reference": _reference_lattice(1.0),
    }
    session.strain_table = pd.DataFrame({"atom_id": [0, 1], "strain_exx": [0.01, -0.02]})

    manifest = session.to_manifest_dict()

    assert manifest["table_sizes"]["strain_table"] == 2
    assert manifest["has_strain_table"] is True
    assert manifest["reference_lattice_keys"] == ["a_reference", "z_reference"]


def test_manifest_reports_empty_strain_table_as_absent() -> None:
    session = AnalysisSession(name="empty_manifest_strain_storage")

    manifest = session.to_manifest_dict()

    assert manifest["table_sizes"]["strain_table"] == 0
    assert manifest["has_strain_table"] is False
    assert manifest["reference_lattice_keys"] == []


def test_pickle_roundtrip_preserves_strain_storage() -> None:
    session = AnalysisSession(name="pickle_strain_storage")
    session.reference_lattice = {"default": _reference_lattice(3.0)}
    session.strain_table = pd.DataFrame({"atom_id": [7], "strain_exx": [0.03]})

    with _writable_temp_dir() as tmp_path:
        path = session.save_pickle(tmp_path / "strain_storage.pkl")
        loaded = AnalysisSession.load_pickle(path)

    assert list(loaded.reference_lattice.keys()) == ["default"]
    assert np.allclose(loaded.reference_lattice["default"].basis, [[3.0, 0.0], [0.0, 4.0]])
    assert loaded.strain_table.equals(session.strain_table)


def test_legacy_pickle_backfills_strain_storage() -> None:
    session = AnalysisSession(name="legacy_strain_storage")
    delattr(session, "reference_lattice")
    delattr(session, "strain_table")

    with _writable_temp_dir() as tmp_path:
        path = session.save_pickle(tmp_path / "legacy_strain_storage.pkl")
        loaded = AnalysisSession.load_pickle(path)

    assert loaded.reference_lattice == {}
    assert loaded.strain_table.empty


def test_clear_downstream_results_clears_strain_storage_on_coordinate_invalidation() -> None:
    session = AnalysisSession(name="clear_strain_storage")
    session.curated_points = pd.DataFrame({"atom_id": [0], "x_px": [1.0], "y_px": [2.0]})
    session.neighbor_graph = {"edges": pd.DataFrame({"source_atom_id": [0], "target_atom_id": [0]})}
    session.local_metrics = pd.DataFrame({"atom_id": [0], "mean_nn_distance_px": [1.0]})
    session.reference_lattice = {"default": _reference_lattice()}
    session.strain_table = pd.DataFrame({"atom_id": [0], "strain_exx": [0.01]})

    session.clear_downstream_results("curate")

    assert not session.curated_points.empty
    assert session.neighbor_graph == {}
    assert session.local_metrics.empty
    assert session.reference_lattice == {}
    assert session.strain_table.empty


def test_clear_downstream_results_keeps_strain_storage_for_vpcf_invalidation() -> None:
    session = AnalysisSession(name="vpcf_does_not_clear_strain_storage")
    session.reference_lattice = {"default": _reference_lattice()}
    session.strain_table = pd.DataFrame({"atom_id": [0], "strain_exx": [0.01]})
    session.annotations = {"records": [{"label": "old"}]}
    session.vector_fields = {"demo": pd.DataFrame({"x_px": [0.0]})}

    session.clear_downstream_results("vpcf")

    assert "default" in session.reference_lattice
    assert not session.strain_table.empty
    assert session.annotations == {}
    assert session.vector_fields == {}
