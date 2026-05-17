from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from em_atom_workbench.session import AnalysisSession, PixelCalibration
from em_atom_workbench.workspace import (
    collect_project_manifest,
    export_stage_figure,
    export_stage_table,
    initialize_analysis_workspace,
    load_stage_session,
    save_stage_session,
    stage_session_path,
    write_stage_manifest,
)


def test_initialize_analysis_workspace_creates_expected_dirs(tmp_path: Path) -> None:
    workspace = initialize_analysis_workspace(tmp_path, "HZO_sample01_area03", "run_001")

    assert workspace.root.exists()
    assert (workspace.root / "state" / "sessions").is_dir()
    assert (workspace.root / "shared").is_dir()
    assert (workspace.root / "manifests").is_dir()
    for stage_name in ("01_findatom", "02_simple_quant", "03_group_centroid"):
        assert (workspace.root / stage_name).is_dir()
        assert (workspace.root / stage_name / "configs").is_dir()
        assert (workspace.root / stage_name / "tables").is_dir()
        assert (workspace.root / stage_name / "figures_preview").is_dir()
        assert (workspace.root / stage_name / "figures_final").is_dir()
    assert not (workspace.root / "03_cropped_group_centroid").exists()

    assert (workspace.shared_dir / "channel_summary.csv").exists()
    assert (workspace.shared_dir / "pixel_calibration.json").exists()
    assert (workspace.shared_dir / "input_metadata.json").exists()

    config = json.loads((workspace.root / "project_config.json").read_text(encoding="utf-8"))
    assert config["dataset_id"] == "HZO_sample01_area03"
    assert config["analysis_id"] == "run_001"
    assert config["workspace_schema_version"] == "1.0"


def test_save_and_load_stage_session(tmp_path: Path) -> None:
    workspace = initialize_analysis_workspace(tmp_path, "dataset", "run")
    session = AnalysisSession(name="test", raw_image=np.zeros((4, 4), dtype=float))
    session.set_stage("curated")

    saved = save_stage_session(session, workspace, "01_final_curated")
    loaded = load_stage_session(workspace, "01_final_curated")

    assert saved == stage_session_path(workspace, "01_final_curated")
    assert loaded.name == "test"
    assert loaded.current_stage == "curated"


def test_active_session_updated_and_history_appended(tmp_path: Path) -> None:
    workspace = initialize_analysis_workspace(tmp_path, "dataset", "run")
    session = AnalysisSession(name="active", raw_image=np.zeros((4, 4), dtype=float))

    save_stage_session(session, workspace, "01_loaded", update_active=True)
    session.set_stage("curated")
    save_stage_session(session, workspace, "01_final_curated", update_active=True)

    assert (workspace.state_dir / "active_session.pkl").exists()
    active_payload = json.loads((workspace.state_dir / "active_session.json").read_text(encoding="utf-8"))
    assert active_payload["stage_name"] == "01_final_curated"
    latest_payload = json.loads((workspace.state_dir / "latest_session.json").read_text(encoding="utf-8"))
    assert latest_payload["latest_stage_name"] == "01_final_curated"
    assert [row["stage_name"] for row in latest_payload["stage_history"]] == ["01_loaded", "01_final_curated"]


def test_stage_output_dirs_and_exports(tmp_path: Path) -> None:
    workspace = initialize_analysis_workspace(tmp_path, "dataset", "run")

    table_paths = export_stage_table(
        workspace,
        "02_simple_quant",
        "analysis_points",
        pd.DataFrame({"x": [1.0], "y": [2.0]}),
    )
    assert table_paths[0] == workspace.root / "02_simple_quant" / "tables" / "analysis_points.csv"
    assert table_paths[0].exists()

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    figure_paths = export_stage_figure(
        workspace,
        "02_simple_quant",
        "demo",
        fig,
        final=True,
        formats=("png",),
    )
    plt.close(fig)

    assert figure_paths[0] == workspace.root / "02_simple_quant" / "figures_final" / "demo.png"
    assert figure_paths[0].exists()


def test_write_stage_and_project_manifest(tmp_path: Path) -> None:
    workspace = initialize_analysis_workspace(tmp_path, "dataset", "run")
    manifest_paths = write_stage_manifest(
        workspace,
        "03_group_centroid",
        {
            "notebook_name": "03_Cropped_group_centroid_analysis.ipynb",
            "session_source": "01_final_curated",
            "tables": {"summary": "summary.csv"},
            "figures": {"final": ["final.png"]},
            "configs": {},
            "session_paths": {"workspace_stage": str(stage_session_path(workspace, "03_group_centroid"))},
        },
    )
    project_manifest = collect_project_manifest(workspace)

    assert manifest_paths["stage_manifest"].exists()
    payload = json.loads(manifest_paths["stage_manifest"].read_text(encoding="utf-8"))
    assert payload["dataset_id"] == "dataset"
    assert payload["analysis_id"] == "run"
    assert payload["stage_name"] == "03_group_centroid"
    assert payload["workspace_schema_version"] == "1.0"
    assert project_manifest.exists()


def test_save_stage_session_updates_shared_files(tmp_path: Path) -> None:
    workspace = initialize_analysis_workspace(tmp_path, "dataset", "run")
    session = AnalysisSession(
        name="shared",
        raw_image=np.zeros((4, 4), dtype=float),
        pixel_calibration=PixelCalibration(size=0.02, unit="nm", source="manual"),
    )
    save_stage_session(session, workspace, "01_loaded")

    channel_summary = pd.read_csv(workspace.shared_dir / "channel_summary.csv")
    pixel_payload = json.loads((workspace.shared_dir / "pixel_calibration.json").read_text(encoding="utf-8"))

    assert "channel" in channel_summary.columns
    assert pixel_payload["size"] == 0.02
    assert pixel_payload["unit"] == "nm"
