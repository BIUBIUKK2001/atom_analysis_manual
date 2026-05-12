from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd

from em_atom_workbench.classification import (
    AtomColumnClassificationConfig,
    apply_class_name_map,
    apply_class_review_from_viewer,
    classify_atom_columns,
    extract_atom_column_features,
    launch_class_review_napari,
)
from em_atom_workbench.curate import curate_points
from em_atom_workbench.detect import detect_multichannel_candidates
from em_atom_workbench.session import AnalysisSession, CurationConfig, DetectionConfig
from em_atom_workbench.utils import synthetic_gaussian_image


def _classified_demo_session() -> AnalysisSession:
    coords = []
    peaks_a = []
    peaks_b = []
    for class_id, amplitude in enumerate((0.55, 1.0, 1.55)):
        for idx in range(5):
            x = 16.0 + idx * 14.0
            y = 18.0 + class_id * 22.0
            coords.append({"candidate_id": len(coords), "x_px": x, "y_px": y, "truth_class_id": class_id})
            peaks_a.append({"x": x, "y": y, "amplitude": amplitude, "sigma_x": 1.1, "sigma_y": 1.1})
            peaks_b.append({"x": x, "y": y, "amplitude": 1.7 - amplitude, "sigma_x": 1.1, "sigma_y": 1.1})
    image_a = synthetic_gaussian_image((96, 96), peaks_a, background=0.05, noise_sigma=0.0)
    image_b = synthetic_gaussian_image((96, 96), peaks_b, background=0.05, noise_sigma=0.0)
    session = AnalysisSession(name="generic_classes", raw_image=image_a, primary_channel="signal_a")
    session.set_channel_state("signal_a", raw_image=image_a, contrast_mode="bright_peak")
    session.set_channel_state("signal_b", raw_image=image_b, contrast_mode="bright_peak")
    session.set_primary_channel("signal_a")
    session.set_workflow(
        "atom_column_classification",
        {"primary_channel": "signal_a", "channel_names": ["signal_a", "signal_b"]},
    )
    session.candidate_points = pd.DataFrame(coords).drop(columns=["truth_class_id"])
    session.set_stage("detected")
    return session


def test_classify_atom_columns_adds_generic_class_fields() -> None:
    session = _classified_demo_session()

    classify_atom_columns(
        session,
        AtomColumnClassificationConfig(
            source_table="candidate",
            feature_channels=("signal_a",),
            features_enabled=("center_intensity", "prominence", "local_snr", "integrated_intensity"),
            n_classes=3,
            cluster_method="kmeans",
            feature_scaling="standard",
            random_state=3,
        ),
    )
    apply_class_name_map(session, {0: "类别_低", 1: "类别_中", 2: "类别_高"}, {0: "#3366cc"})

    assert session.current_stage == "classified"
    assert {"class_id", "class_name", "class_color", "class_confidence", "class_source", "class_reviewed"}.issubset(
        session.candidate_points.columns
    )
    assert session.candidate_points["class_id"].nunique() == 3
    assert "signal_a__center_intensity" in session.classification_features.columns
    assert set(session.candidate_points["class_name"]) == {"类别_低", "类别_中", "类别_高"}
    assert session.classification_summary["chosen_n_classes"] == 3


def test_multichannel_feature_extraction_uses_arbitrary_channel_names() -> None:
    session = _classified_demo_session()
    features = extract_atom_column_features(
        session,
        AtomColumnClassificationConfig(
            source_table="candidate",
            feature_channels=("signal_a", "signal_b"),
            features_enabled=("center_intensity", "prominence", "mean"),
            feature_channel_weights={"signal_b": 2.0},
        ),
    )

    assert "signal_a__center_intensity" in features.columns
    assert "signal_b__center_intensity" in features.columns
    assert "haadf__center_intensity" not in features.columns
    assert "idpc__center_intensity" not in features.columns


def test_classification_parameter_variants_run() -> None:
    for method, scaling in (("gaussian_mixture", "robust"), ("agglomerative", "minmax")):
        session = _classified_demo_session()
        classify_atom_columns(
            session,
            AtomColumnClassificationConfig(
                source_table="candidate",
                feature_channels=("signal_a", "signal_b"),
                n_classes=3,
                cluster_method=method,
                feature_scaling=scaling,
                confidence_threshold=0.0,
                random_state=5,
            ),
        )
        assert session.candidate_points["class_id"].nunique() == 3


def test_detect_multichannel_candidates_uses_generic_atom_column_role() -> None:
    session = _classified_demo_session()
    session.candidate_points = pd.DataFrame()

    detect_multichannel_candidates(
        session,
        {
            "signal_a": DetectionConfig(
                gaussian_sigma=0.0,
                min_distance=8,
                threshold_rel=0.08,
                min_prominence=0.05,
                min_snr=0.0,
                edge_margin=4,
                patch_radius=5,
                dedupe_radius_px=4.0,
            ),
            "signal_b": DetectionConfig(
                gaussian_sigma=0.0,
                min_distance=8,
                threshold_rel=0.08,
                min_prominence=0.05,
                min_snr=0.0,
                edge_margin=4,
                patch_radius=5,
                dedupe_radius_px=4.0,
            ),
        },
        dedupe_radius_px=4.0,
    )

    assert session.current_stage == "detected"
    assert set(session.candidate_points["column_role"]) == {"atom_column"}
    assert "detected_from_channels" in session.candidate_points.columns
    assert len(session.candidate_points) >= 10


class _FakeLayer:
    def __init__(self, data: np.ndarray, metadata: dict) -> None:
        self.data = data
        self.metadata = metadata


class _FakeViewer:
    def __init__(self) -> None:
        self.layers = [
            _FakeLayer(
                np.asarray([[18.0, 16.0], [18.0, 30.0]], dtype=float),
                {"class_id": 0, "class_name": "reviewed_a", "class_color": "#111111", "origin": {"x": 0, "y": 0}},
            ),
            _FakeLayer(
                np.asarray([[40.0, 16.0]], dtype=float),
                {"class_id": 1, "class_name": "reviewed_b", "class_color": "#222222", "origin": {"x": 0, "y": 0}},
            ),
        ]


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


def test_launch_class_review_napari_uses_bounded_canvas_point_size(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "napari", types.SimpleNamespace(Viewer=_FakeNapariViewer))
    session = _classified_demo_session()
    classify_atom_columns(
        session,
        AtomColumnClassificationConfig(source_table="candidate", feature_channels=("signal_a",), n_classes=3),
    )

    viewer = launch_class_review_napari(session, point_size=6.0)

    point_layers = [layer for name, layer in viewer.layers.items() if name.startswith("class_")]
    assert point_layers
    assert all(layer.kwargs["canvas_size_limits"] == (4, 8) for layer in point_layers)
    assert all(layer.metadata["point_size"] == 6.0 for layer in point_layers)
    assert all(layer.current_symbol == "disc" for layer in point_layers)
    assert all(layer.current_size == 6.0 for layer in point_layers)
    assert all(layer.current_border_width == 0.0 for layer in point_layers)
    assert all(layer.editable for layer in point_layers)


def test_launch_class_review_napari_can_review_candidate_table_before_refinement(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "napari", types.SimpleNamespace(Viewer=_FakeNapariViewer))
    session = _classified_demo_session()
    classify_atom_columns(
        session,
        AtomColumnClassificationConfig(source_table="candidate", feature_channels=("signal_a",), n_classes=3),
    )
    session.refined_points = session.candidate_points.iloc[:1].copy()

    viewer = launch_class_review_napari(session, point_size=6.0, source_table="candidate")

    reviewed_count = sum(
        len(layer.data)
        for name, layer in viewer.layers.items()
        if name.startswith("class_")
    )
    assert reviewed_count == len(session.candidate_points)


def test_apply_class_review_from_viewer_updates_classes_and_curation_keeps_fields() -> None:
    session = _classified_demo_session()
    classify_atom_columns(
        session,
        AtomColumnClassificationConfig(source_table="candidate", feature_channels=("signal_a",), n_classes=3),
    )

    apply_class_review_from_viewer(session, _FakeViewer(), source_table="candidate")
    curate_points(session, CurationConfig(min_quality_score=0.0, max_fit_residual=1.0, edge_margin=0))

    assert session.current_stage == "curated"
    assert len(session.curated_points) == 3
    assert set(session.curated_points["class_name"]) == {"reviewed_a", "reviewed_b"}
    assert bool(session.curated_points["class_reviewed"].all())
