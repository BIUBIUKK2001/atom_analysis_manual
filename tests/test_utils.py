from pathlib import Path

import numpy as np

from em_atom_workbench.session import AnalysisSession
from em_atom_workbench.utils import (
    extract_patch,
    load_or_connect_session,
    resolve_session_snapshot,
    save_active_session,
    save_checkpoint,
    save_session_snapshot,
)


def test_save_active_session_and_load_or_connect_session(tmp_path: Path) -> None:
    session = AnalysisSession(name="real_session", raw_image=np.zeros((8, 8), dtype=np.uint16))
    session.set_stage("detected")

    active_path = save_active_session(session, tmp_path)
    loaded = load_or_connect_session(tmp_path, required_stage="loaded")

    assert active_path.exists()
    assert loaded.name == session.name
    assert loaded.current_stage == "detected"


def test_save_checkpoint_and_legacy_snapshot_resolution(tmp_path: Path) -> None:
    session = AnalysisSession(name="real_session", raw_image=np.zeros((8, 8), dtype=float))

    checkpoint_path = save_checkpoint(session, tmp_path, "00_session_initialized.pkl")
    legacy_path = save_session_snapshot(session, tmp_path, "01_detection_ready.pkl")

    assert checkpoint_path.exists()
    assert legacy_path.exists()
    assert resolve_session_snapshot(tmp_path, "01_detection_ready.pkl") == legacy_path


def test_extract_patch_accepts_float_half_window() -> None:
    image = np.arange(100).reshape(10, 10)

    patch, bounds = extract_patch(image, center_xy_global=(5.0, 5.0), half_window=2.0)

    np.testing.assert_array_equal(patch, image[3:8, 3:8])
    assert bounds == (3, 8, 3, 8)
