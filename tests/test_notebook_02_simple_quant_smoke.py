from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def test_build_02_simple_quant_notebook_generates_compilable_notebook() -> None:
    subprocess.run([sys.executable, "scripts/build_02_simple_quant_notebook.py"], check=True)
    notebook_path = Path("notebooks") / "02_Simple_quantitative_spacing_analysis.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    joined = "\n".join(_cell_source(cell) for cell in notebook.get("cells", []))
    source_by_id = {str(cell.get("id", "")): _cell_source(cell) for cell in notebook.get("cells", [])}

    for parameter in (
        "SOURCE_TABLE",
        "OPEN_ROI_PICKER",
        "OPEN_BASIS_VECTOR_PICKER",
        "BASIS_MODE",
        "BASIS_ROLES",
        "BASIS_VECTOR_SPECS",
        "GLOBAL_BASIS_FALLBACK",
        "FLIP_BASIS_NAMES",
        "TASK1A_CLASS_GROUP_MODE",
        "TASK2_PAIR_MODE",
        "TASK2_PROJECTION_VECTOR",
        "TASK3_ROIS",
        "TASK1B_ANCHOR_SELECTION",
        "period_segment_table",
        "period_summary_table",
        "pair_line_summary_table",
        "group_centroid_table",
        "cell_table",
    ):
        assert parameter in joined

    assert "plot_basis_check_on_image" in joined
    assert "all_analysis_points = analysis_points.copy()" in source_by_id["roi-selection"]
    assert "class_group_mode=TASK1A_CLASS_GROUP_MODE" in source_by_id["task1a-run"]
    assert "export_task1A_excel" in source_by_id["task1a-export"]
    assert "export_task1B_excel" in source_by_id["task1b-export"]
    assert "export_task2_excel" in source_by_id["task2-export"]
    assert "export_task3_excel" in source_by_id["task3-export"]
    assert "export_notebook02_results" in source_by_id["final-export"]
    assert "group_weights" not in joined

    for cell in notebook.get("cells", []):
        assert cell.get("id")
        if cell.get("cell_type") == "code":
            ast.parse(_cell_source(cell), filename=f"02_Simple_quantitative_spacing_analysis.ipynb:{cell['id']}")
