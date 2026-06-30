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
        "INTENSITY_INPUT_MODE",
        "STACK_PATH",
        "STACK_AXIS",
        "STACK_SLICE_INDICES",
        "STACK_COORDINATE_MODE",
        "slice_refined",
        "fixed",
        "COMPUTE_FIXED_COORDINATE_CONTROL",
        "DISK_RADIUS_PX",
        "MAP_METRIC",
        "STACK_PROFILE_METRIC",
        "HIST_BINS",
        "REFINEMENT_CONFIG",
        "NN_CONTEXT_MODE",
        "CLASS_REFINEMENT_OVERRIDES",
        "OPEN_STACK_REFINEMENT_VIEWER",
        "NAPARI_REVIEW_SLICE_INDEX",
        "SAVE_PREVIEW_FIGURES",
        "SAVE_FINAL_FIGURES",
    ):
        assert parameter in joined

    for api_name in (
        "initialize_disk_intensity_analysis",
        "initialize_notebook04_intensity_context",
        "run_notebook04_intensity_analysis",
        "export_disk_intensity_analysis",
        "export_notebook04_intensity_results",
        "compute_disk_intensity_table",
        "compute_stack_disk_intensity_table",
        "compute_per_slice_disk_intensity_table",
        "summarize_stack_disk_intensity",
        "plot_disk_intensity_map",
        "plot_disk_intensity_histogram",
        "plot_stack_intensity_profiles",
        "plot_stack_slice_intensity_map",
        "plot_stack_intensity_histogram",
        "plot_stack_refinement_shift_profile",
        "launch_stack_refinement_napari_viewer",
    ):
        assert api_name in joined

    for table_name in (
        "stack_refined_points",
        "stack_refined_disk_intensity_table",
        "stack_refined_disk_intensity_summary",
        "stack_fixed_disk_intensity_table",
        "stack_fixed_disk_intensity_summary",
        "center_shift_rejected",
        "refinement_path",
    ):
        assert table_name in joined

    assert "04_intensity_mapping" in joined
    assert "vacancy threshold" in joined
    assert "stack registration" in joined

    for cell in notebook.get("cells", []):
        assert cell.get("id")
        if cell.get("cell_type") == "code":
            ast.parse(_cell_source(cell), filename=f"04_Disk_integrated_intensity_mapping.ipynb:{cell['id']}")
