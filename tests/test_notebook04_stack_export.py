from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from em_atom_workbench import AnalysisSession, export_notebook04_intensity_results, initialize_analysis_workspace


def test_export_notebook04_intensity_results_writes_stack_tables(tmp_path: Path) -> None:
    workspace = initialize_analysis_workspace(tmp_path, "dataset", "run")
    session = AnalysisSession(name="notebook04_stack", raw_image=np.zeros((4, 4), dtype=float))
    context = {
        "workspace": workspace,
        "session": session,
        "session_source": "01_final_curated",
        "session_path": None,
        "use_active_session": False,
        "session_load_mode": "stage_session",
        "resolved_session_path": str(workspace.sessions_dir / "01_final_curated.pkl"),
        "coordinate_source": "refined",
        "source_table": "refined_points",
        "image_channel": "primary",
        "image_key": "raw",
        "stack_path": "stack.npy",
        "stack_shape": (2, 4, 4),
        "slice_indices": [0, 1],
        "intensity_input_mode": "stack",
    }
    result = {
        "tables": {
            "stack_disk_intensity_table": pd.DataFrame({"slice_index": [0], "disk_intensity_mean": [1.0]}),
            "stack_disk_intensity_summary": pd.DataFrame({"slice_index": [0], "mean": [1.0]}),
        },
        "config_summary": {
            "intensity_input_mode": "stack",
            "stack_coordinate_mode": "fixed",
            "compute_fixed_coordinate_control": False,
            "disk_radius_px": 2.0,
        },
        "qc_summary_tables": {},
    }

    manifest = export_notebook04_intensity_results(
        context,
        result,
        config={"session_source": "01_final_curated", "disk_radius_px": 2.0},
        final_figures={},
    )

    assert "stack_disk_intensity_table" in manifest["tables"]
    assert "stack_disk_intensity_summary" in manifest["tables"]
    assert Path(manifest["tables"]["stack_disk_intensity_table"]).exists()
    assert manifest["intensity_input_mode"] == "stack"
    assert manifest["stack_coordinate_mode"] == "fixed"
    assert manifest["stack_shape"] == (2, 4, 4)
