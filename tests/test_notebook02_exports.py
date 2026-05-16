from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from em_atom_workbench.notebook_workflows import export_notebook02_results, export_task1A_excel
from em_atom_workbench.session import AnalysisSession


def _output_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "output": root,
        "tables": root / "tables",
        "figures": root / "figures",
        "configs": root / "configs",
        "session": root / "session",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def test_task_excel_export_writes_empty_sheets(tmp_path: Path) -> None:
    dirs = _output_dirs(tmp_path)

    result = export_task1A_excel(
        dirs,
        period_segment_table=pd.DataFrame(),
        period_summary_table=pd.DataFrame({"roi_id": [], "n_segments": []}),
        task1A_config=pd.DataFrame({"parameter": ["class_group_mode"], "value": ["per_class"]}),
        roi_class_selection=pd.DataFrame(),
        basis_vectors=pd.DataFrame(),
    )

    path = Path(result["path"])
    assert path.exists()
    with zipfile.ZipFile(path) as archive:
        worksheet_names = [name for name in archive.namelist() if name.startswith("xl/worksheets/sheet")]
    assert len(worksheet_names) == 5


def test_export_notebook02_results_writes_manifest_tables_and_checkpoint(tmp_path: Path) -> None:
    dirs = _output_dirs(tmp_path)
    session = AnalysisSession(name="notebook02_export", raw_image=np.zeros((4, 4), dtype=float))
    table = pd.DataFrame({"value": [1, 2]})

    manifest = export_notebook02_results(
        session=session,
        output_dirs=dirs,
        tables={"example": table},
        figures={},
        configs={"config": {"ok": True}},
        excel_exports={"task1A": {"path": "task1A.xlsx"}},
    )

    manifest_path = Path(manifest["manifest"])
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["tables"]["example"].endswith("example.csv")
    assert payload["excel_exports"]["task1A"]["path"] == "task1A.xlsx"
    assert Path(manifest["session_checkpoint"]).exists()
