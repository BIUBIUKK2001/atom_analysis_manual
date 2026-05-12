from __future__ import annotations

from pathlib import Path

import numpy as np

from em_atom_workbench.session import AnalysisSession


def test_set_workflow_updates_manifest_and_provenance() -> None:
    session = AnalysisSession(name="workflow_session", raw_image=np.zeros((8, 8), dtype=float))

    session.set_workflow(
        "hfo2_multichannel",
        {
            "primary_channel": "idpc",
            "heavy_channel": "haadf",
            "light_channel": "idpc",
            "confirm_channel": "abf",
        },
    )

    manifest = session.to_manifest_dict()

    assert session.workflow_mode == "hfo2_multichannel"
    assert session.workflow_settings["primary_channel"] == "idpc"
    assert manifest["workflow_mode"] == "hfo2_multichannel"
    assert manifest["workflow_settings"]["heavy_channel"] == "haadf"
    assert session.provenance[-1]["step"] == "set_workflow"


def test_load_pickle_backfills_legacy_workflow_fields(tmp_path: Path) -> None:
    session = AnalysisSession(name="legacy_pickle", raw_image=np.zeros((8, 8), dtype=float))
    delattr(session, "workflow_mode")
    delattr(session, "workflow_settings")

    path = session.save_pickle(tmp_path / "legacy.pkl")
    loaded = AnalysisSession.load_pickle(path)

    assert loaded.workflow_mode == "single_channel"
    assert loaded.workflow_settings == {}
