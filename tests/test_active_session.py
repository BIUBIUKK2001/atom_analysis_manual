from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import shutil
import uuid

import numpy as np
import pytest

from em_atom_workbench.session import AnalysisSession
from em_atom_workbench.utils import ACTIVE_SESSION_INFO, load_or_connect_session, save_active_session


@contextmanager
def _writable_temp_dir():
    root = Path(".runtime-tmp") / f"active_session_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_load_or_connect_session_reads_matching_active_session() -> None:
    session = AnalysisSession(
        name="real_session",
        input_path="I:/demo/sample.dm4",
        dataset_index=1,
        raw_image=np.zeros((8, 8), dtype=np.uint16),
    )
    session.set_stage("detected")

    with _writable_temp_dir() as tmp_path:
        save_active_session(session, tmp_path)
        loaded = load_or_connect_session(tmp_path, required_stage="loaded")

    assert loaded.name == "real_session"
    assert loaded.input_path == "I:/demo/sample.dm4"
    assert loaded.dataset_index == 1
    assert loaded.current_stage == "detected"


def test_load_or_connect_session_rejects_stage_below_requirement() -> None:
    session = AnalysisSession(name="too_early", raw_image=np.zeros((8, 8), dtype=np.uint16))
    session.set_stage("loaded")

    with _writable_temp_dir() as tmp_path:
        save_active_session(session, tmp_path)
        with pytest.raises(ValueError, match="required stage 'detected'"):
            load_or_connect_session(tmp_path, required_stage="detected")


def test_load_or_connect_session_accepts_strain_stage_for_curated_requirement() -> None:
    session = AnalysisSession(name="strain_ready", raw_image=np.zeros((8, 8), dtype=np.uint16))
    session.set_stage("strain")

    with _writable_temp_dir() as tmp_path:
        save_active_session(session, tmp_path)
        loaded = load_or_connect_session(tmp_path, required_stage="curated")

    assert loaded.name == "strain_ready"
    assert loaded.current_stage == "strain"


def test_load_or_connect_session_rejects_active_manifest_mismatch() -> None:
    session = AnalysisSession(
        name="real_session",
        input_path="I:/demo/sample.dm4",
        dataset_index=1,
        raw_image=np.zeros((8, 8), dtype=np.uint16),
    )
    session.set_stage("metrics")

    with _writable_temp_dir() as tmp_path:
        save_active_session(session, tmp_path)
        info_path = tmp_path / ACTIVE_SESSION_INFO
        payload = json.loads(info_path.read_text(encoding="utf-8"))
        payload["session_name"] = "different_session"
        info_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        with pytest.raises(ValueError, match="does not match"):
            load_or_connect_session(tmp_path, required_stage="loaded")


def test_load_or_connect_session_accepts_explicit_session_path() -> None:
    session = AnalysisSession(name="manual_session", raw_image=np.zeros((8, 8), dtype=np.uint16))
    session.set_stage("curated")

    with _writable_temp_dir() as tmp_path:
        manual_path = session.save_pickle(tmp_path / "manual_checkpoint.pkl")
        loaded = load_or_connect_session(tmp_path, required_stage="detected", session_path=manual_path)

    assert loaded.name == "manual_session"
    assert loaded.current_stage == "curated"


def test_load_or_connect_session_requires_existing_active_session() -> None:
    with _writable_temp_dir() as tmp_path:
        with pytest.raises(FileNotFoundError, match="Active session info was not found"):
            load_or_connect_session(tmp_path, required_stage="loaded")
