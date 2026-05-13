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

    for parameter in (
        "SOURCE_TABLE",
        "OPEN_ROI_PICKER",
        "OPEN_BASIS_VECTOR_PICKER",
        "BASIS_VECTOR_SPECS",
        "MEASUREMENT_TASKS",
        "PERIODIC_VECTOR_TASKS",
        "SEGMENT_COLOR_BY",
        "SEGMENT_LINEWIDTH",
        "SHOW_SIDE_PANEL",
    ):
        assert parameter in joined

    for cell in notebook.get("cells", []):
        assert cell.get("id")
        if cell.get("cell_type") == "code":
            ast.parse(_cell_source(cell), filename=f"02_Simple_quantitative_spacing_analysis.ipynb:{cell['id']}")
