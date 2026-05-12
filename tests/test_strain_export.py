from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from em_atom_workbench.export import export_results
from em_atom_workbench.session import AnalysisSession, ExportConfig


@contextmanager
def _writable_temp_dir():
    root = Path(".runtime-tmp") / f"strain_export_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _session_with_strain_table() -> AnalysisSession:
    session = AnalysisSession(name="strain_export_case", raw_image=np.zeros((8, 8), dtype=float))
    session.curated_points = pd.DataFrame({"atom_id": [0], "x_px": [1.0], "y_px": [2.0]})
    session.strain_table = pd.DataFrame(
        {
            "atom_id": [0, 1],
            "x_px": [1.0, 2.0],
            "y_px": [3.0, 4.0],
            "eps_xx": [0.01, -0.02],
            "qc_flag": ["ok", "too_few_pairs"],
        }
    )
    return session


def test_export_results_writes_strain_table_and_manifest() -> None:
    session = _session_with_strain_table()

    with _writable_temp_dir() as tmp_path:
        manifest_path = export_results(
            session,
            ExportConfig(
                output_dir=tmp_path,
                table_formats=("csv",),
                overwrite=True,
                save_fig_atom_overlay=False,
            ),
        )

        export_root = manifest_path.parent
        strain_path = export_root / "tables" / "strain_table.csv"
        assert strain_path.exists()

        exported = pd.read_csv(strain_path)
        assert list(exported["eps_xx"]) == [0.01, -0.02]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert any(path.endswith("tables\\strain_table.csv") or path.endswith("tables/strain_table.csv") for path in manifest["exported_files"])
        assert manifest["session"]["table_sizes"]["strain_table"] == 2
        assert manifest["session"]["has_strain_table"] is True


def test_export_results_skips_absent_or_empty_strain_table() -> None:
    with _writable_temp_dir() as tmp_path:
        for name, strain_table in (("absent_strain", None), ("empty_strain", pd.DataFrame())):
            session = AnalysisSession(name=name, raw_image=np.zeros((8, 8), dtype=float))
            if strain_table is None:
                delattr(session, "strain_table")
            else:
                session.strain_table = strain_table

            manifest_path = export_results(
                session,
                ExportConfig(
                    output_dir=tmp_path,
                    table_formats=("csv",),
                    overwrite=True,
                    save_atoms_table=False,
                    save_metrics_table=False,
                    save_fig_atom_overlay=False,
                ),
            )

            export_root = manifest_path.parent
            assert not (export_root / "tables" / "strain_table.csv").exists()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert not any("strain_table" in path for path in manifest["exported_files"])
