from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def test_build_04_disk_intensity_notebook_generates_compilable_notebook() -> None:
    subprocess.run([sys.executable, "scripts/build_04_disk_integrated_intensity_notebook.py"], check=True)
    notebook_path = Path("notebooks") / "04_Disk_integrated_intensity_mapping.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    joined = "\n".join(_cell_source(cell) for cell in notebook.get("cells", []))

    for parameter in (
        "OUTPUT_ROOT",
        "DATASET_ID",
        "ANALYSIS_ID",
        "SESSION_SOURCE",
        "SESSION_PATH",
        "USE_ACTIVE_SESSION",
        "COORDINATE_SOURCE",
        "candidate",
        "refined",
        "curated",
        "DISK_RADIUS_PX",
        "MAP_METRIC",
        "HIST_BINS",
        "SAVE_PREVIEW_FIGURES",
        "SAVE_FINAL_FIGURES",
    ):
        assert parameter in joined

    assert "initialize_disk_intensity_analysis" in joined
    assert "compute_disk_intensity_table" in joined
    assert "plot_disk_intensity_map" in joined
    assert "plot_disk_intensity_histogram" in joined
    assert "export_disk_intensity_analysis" in joined
    assert "04_intensity_mapping" in joined
    assert "vacancy threshold" in joined

    for cell in notebook.get("cells", []):
        assert cell.get("id")
        if cell.get("cell_type") == "code":
            ast.parse(_cell_source(cell), filename=f"04_Disk_integrated_intensity_mapping.ipynb:{cell['id']}")
