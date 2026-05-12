from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

import em_atom_workbench.curate as curate_module
from em_atom_workbench.curate import apply_candidate_edits_from_viewer, launch_napari_candidate_editor
from em_atom_workbench.notebook_workflows import run_generic_refinement
from em_atom_workbench.session import AnalysisSession, RefinementConfig


class _FakeLayer:
    def __init__(self, data: np.ndarray, metadata: dict) -> None:
        self.data = data
        self.metadata = metadata


class _FakeViewer:
    def __init__(self, data: np.ndarray, metadata: dict) -> None:
        self.layers = {"atom_points": _FakeLayer(data, metadata)}


class _FakeNapariLayer:
    def __init__(self, data, **kwargs) -> None:
        self.data = np.asarray(data)
        self.kwargs = dict(kwargs)
        self.metadata = {}
        self.editable = False


class _FakeNapariViewer:
    def __init__(self, title: str) -> None:
        self.title = title
        self.layers: dict[str, _FakeNapariLayer] = {}

    def add_image(self, data, *, name: str, **kwargs):
        layer = _FakeNapariLayer(data, name=name, **kwargs)
        self.layers[name] = layer
        return layer

    def add_points(self, data, *, name: str, **kwargs):
        layer = _FakeNapariLayer(data, name=name, **kwargs)
        self.layers[name] = layer
        return layer


def _review_demo_session() -> AnalysisSession:
    image = np.zeros((64, 64), dtype=float)
    session = AnalysisSession(name="candidate_review", raw_image=image, primary_channel="signal")
    session.set_channel_state("signal", raw_image=image, contrast_mode="bright_peak")
    session.set_primary_channel("signal")
    session.set_workflow("atom_column_classification", {"primary_channel": "signal", "channel_names": ["signal"]})
    session.candidate_points = pd.DataFrame(
        {
            "candidate_id": [0, 1],
            "x_px": [12.0, 28.0],
            "y_px": [18.0, 18.0],
            "column_role": ["atom_column", "atom_column"],
        }
    )
    session.refined_points = pd.DataFrame({"atom_id": [0], "x_px": [12.1], "y_px": [18.1]})
    session.curated_points = pd.DataFrame({"atom_id": [0], "x_px": [12.1], "y_px": [18.1], "keep": [True]})
    session.classification_features = pd.DataFrame({"atom_id": [0], "signal__center_intensity": [1.0]})
    session.classification_summary = {"chosen_n_classes": 1}
    session.local_metrics = pd.DataFrame({"atom_id": [0], "mean_nn_distance_px": [8.0]})
    session.set_stage("detected")
    return session


def test_launch_napari_candidate_editor_sets_new_point_defaults(monkeypatch) -> None:
    monkeypatch.setattr(curate_module, "_require_napari", lambda: types.SimpleNamespace(Viewer=_FakeNapariViewer))
    session = _review_demo_session()

    viewer = launch_napari_candidate_editor(session, point_size=6.0)
    layer = viewer.layers["atom_points"]

    assert layer.kwargs["symbol"] == "disc"
    assert layer.kwargs["canvas_size_limits"] == (4, 8)
    assert layer.current_symbol == "disc"
    assert layer.current_size == 6.0
    assert layer.current_face_color == "#00d7ff"
    assert layer.current_border_width == 0.0


def test_apply_candidate_edits_from_viewer_writes_generic_reviewed_candidates() -> None:
    session = _review_demo_session()
    viewer = _FakeViewer(
        np.asarray([[20.0, 11.0], [31.0, 33.0], [45.0, 49.0]], dtype=float),
        {
            "origin": {"x": 2, "y": 3},
            "channel_name": "signal",
            "image_key": "processed",
            "point_size": 6.0,
        },
    )

    apply_candidate_edits_from_viewer(session, viewer)

    assert session.current_stage == "candidate_reviewed"
    assert len(session.candidate_points) == 3
    assert session.candidate_points["candidate_id"].tolist() == [0, 1, 2]
    assert session.candidate_points["x_px"].tolist() == [13.0, 35.0, 51.0]
    assert session.candidate_points["y_px"].tolist() == [23.0, 34.0, 48.0]
    assert set(session.candidate_points["column_role"]) == {"atom_column"}
    assert set(session.candidate_points["contrast_mode_used"]) == {"manual_edit"}
    assert session.candidate_points["class_id"].isna().all()
    assert session.candidate_points["class_name"].isna().all()
    assert session.candidate_points["class_color"].isna().all()
    assert session.candidate_points["class_confidence"].isna().all()
    assert session.candidate_points["class_source"].isna().all()
    assert not session.candidate_points["class_reviewed"].any()
    assert session.refined_points.empty
    assert session.curated_points.empty
    assert session.classification_features.empty
    assert session.classification_summary == {}
    assert session.local_metrics.empty
    assert session.provenance[-1]["step"] == "apply_candidate_edits_from_viewer"
    assert session.provenance[-1]["notes"]["previous_candidate_count"] == 2
    assert session.provenance[-1]["notes"]["candidate_count"] == 3


def test_generic_refinement_requires_candidate_review(tmp_path) -> None:
    session = _review_demo_session()
    session.refined_points = pd.DataFrame()
    session.curated_points = pd.DataFrame()
    session.classification_features = pd.DataFrame()
    session.classification_summary = {}

    with pytest.raises(RuntimeError, match="reviewed in napari"):
        run_generic_refinement(
            session,
            result_root=tmp_path,
            refinement_config=RefinementConfig(mode="adaptive_atomap", gaussian_retry_count=0),
        )

    apply_candidate_edits_from_viewer(
        session,
        _FakeViewer(
            np.asarray([[18.0, 12.0], [18.0, 28.0]], dtype=float),
            {"origin": {"x": 0, "y": 0}, "channel_name": "signal", "image_key": "processed"},
        ),
    )

    with pytest.raises(RuntimeError, match="classified"):
        run_generic_refinement(
            session,
            result_root=tmp_path,
            refinement_config=RefinementConfig(mode="adaptive_atomap", gaussian_retry_count=0),
        )

    session.candidate_points["class_id"] = [0, 1]
    session.candidate_points["class_name"] = ["class_0", "class_1"]
    session.candidate_points["class_color"] = ["#111111", "#222222"]
    session.candidate_points["class_confidence"] = [1.0, 1.0]
    session.candidate_points["class_source"] = ["manual_review", "manual_review"]
    session.candidate_points["class_reviewed"] = [True, True]
    session.set_stage("classified")

    result = run_generic_refinement(
        session,
        result_root=tmp_path,
        refinement_config=RefinementConfig(mode="adaptive_atomap", gaussian_retry_count=0),
    )

    assert result.session is session
    assert not session.refined_points.empty
