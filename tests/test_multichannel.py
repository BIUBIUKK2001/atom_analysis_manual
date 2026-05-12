from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree

from em_atom_workbench.curate import (
    apply_hfo2_heavy_candidate_edits_from_viewer,
    apply_hfo2_light_candidate_edits_from_viewer,
)
from em_atom_workbench.detect import detect_candidates, detect_hfo2_multichannel_candidates
from em_atom_workbench.detect import detect_hfo2_heavy_candidates, detect_hfo2_light_candidates
from em_atom_workbench.io import load_image_bundle
from em_atom_workbench.preprocess import preprocess_channels, preprocess_image
from em_atom_workbench.refine import refine_points
from em_atom_workbench.session import (
    AnalysisSession,
    DetectionConfig,
    HfO2MultichannelDetectionConfig,
    PreprocessConfig,
    RefinementConfig,
)
from em_atom_workbench.utils import synthetic_hfo2_multichannel_bundle


class _FakeLayer:
    def __init__(self, data: np.ndarray, origin_x: int = 0, origin_y: int = 0) -> None:
        self.data = np.asarray(data, dtype=float)
        self.metadata = {"origin": {"x": origin_x, "y": origin_y}}


class _FakeViewer:
    def __init__(self, data: np.ndarray, origin_x: int = 0, origin_y: int = 0) -> None:
        self.layers = {"atom_points": _FakeLayer(data, origin_x=origin_x, origin_y=origin_y)}


def _recall_with_tolerance(reference: np.ndarray, predicted: np.ndarray, tolerance: float) -> float:
    if len(reference) == 0 or len(predicted) == 0:
        return 0.0
    tree = cKDTree(predicted)
    distances, _ = tree.query(reference, k=1)
    return float(np.mean(distances <= tolerance))


def _build_multichannel_session(images: dict[str, np.ndarray]) -> AnalysisSession:
    session = AnalysisSession(
        name="synthetic_multichannel",
        raw_image=images["idpc"],
        primary_channel="idpc",
    )
    for channel_name, image in images.items():
        contrast_mode = "dark_dip" if channel_name == "abf" else "bright_peak"
        session.set_channel_state(
            channel_name,
            input_path=f"{channel_name}.tif",
            raw_image=image,
            contrast_mode=contrast_mode,
        )
    session.set_primary_channel("idpc")
    return session


def test_load_image_bundle_and_preprocess_channels_keep_primary_alias(tmp_path: Path) -> None:
    paths: dict[str, Path] = {}
    for channel_name, value in {
        "idpc": np.arange(64, dtype=np.float32).reshape(8, 8),
        "haadf": np.flipud(np.arange(64, dtype=np.float32).reshape(8, 8)),
        "abf": np.rot90(np.arange(64, dtype=np.float32).reshape(8, 8)),
    }.items():
        path = tmp_path / f"{channel_name}.tif"
        tifffile.imwrite(path, value)
        paths[channel_name] = path

    session = load_image_bundle(
        paths,
        primary_channel="idpc",
        manual_calibration={"size": 0.1, "unit": "nm"},
        contrast_modes={"idpc": "bright_peak", "haadf": "bright_peak", "abf": "dark_dip"},
    )

    assert session.primary_channel == "idpc"
    assert set(session.list_channels()) == {"idpc", "haadf", "abf"}
    assert np.array_equal(session.raw_image, tifffile.imread(paths["idpc"]))
    assert session.pixel_calibration.size is not None

    preprocess_channels(
        session,
        {
            "idpc": PreprocessConfig(contrast_mode="bright_peak", denoise_method="none", edge_mask_width=0),
            "haadf": PreprocessConfig(contrast_mode="bright_peak", denoise_method="none", edge_mask_width=0),
            "abf": PreprocessConfig(contrast_mode="dark_dip", denoise_method="none", edge_mask_width=0),
        },
    )

    assert session.get_processed_image("idpc").shape == (8, 8)
    assert session.get_processed_origin("idpc") == session.get_processed_origin("haadf")
    assert np.array_equal(session.preprocess_result["processed_image"], session.get_processed_image("idpc"))


def test_detect_hfo2_multichannel_candidates_recovers_heavy_and_light_columns() -> None:
    images, truth = synthetic_hfo2_multichannel_bundle(rng_seed=4)
    session = _build_multichannel_session(images)
    preprocess_channels(
        session,
        {
            "idpc": PreprocessConfig(contrast_mode="bright_peak", denoise_method="none", edge_mask_width=0),
            "haadf": PreprocessConfig(contrast_mode="bright_peak", denoise_method="none", edge_mask_width=0),
            "abf": PreprocessConfig(contrast_mode="dark_dip", denoise_method="none", edge_mask_width=0),
        },
    )

    config = HfO2MultichannelDetectionConfig()
    detect_hfo2_multichannel_candidates(session, config)

    heavy_pred = session.candidate_points.loc[session.candidate_points["column_role"] == "heavy_atom", ["x_px", "y_px"]].to_numpy()
    light_pred = session.candidate_points.loc[session.candidate_points["column_role"] == "light_atom", ["x_px", "y_px"]].to_numpy()
    heavy_recall = _recall_with_tolerance(truth["heavy"][["x_px", "y_px"]].to_numpy(), heavy_pred, tolerance=2.0)
    light_recall = _recall_with_tolerance(truth["light"][["x_px", "y_px"]].to_numpy(), light_pred, tolerance=2.0)

    assert heavy_recall >= 0.95
    assert light_recall >= 0.80
    assert {"column_role", "seed_channel", "confirm_channel", "parent_heavy_id"}.issubset(session.candidate_points.columns)
    assert set(session.candidate_points["seed_channel"]) >= {"haadf", "idpc"}


def test_hfo2_multichannel_detection_without_abf_degrades_gracefully() -> None:
    images, truth = synthetic_hfo2_multichannel_bundle(include_abf=False, rng_seed=7)
    session = _build_multichannel_session(images)
    preprocess_channels(
        session,
        {
            "idpc": PreprocessConfig(contrast_mode="bright_peak", denoise_method="none", edge_mask_width=0),
            "haadf": PreprocessConfig(contrast_mode="bright_peak", denoise_method="none", edge_mask_width=0),
        },
    )

    config = HfO2MultichannelDetectionConfig(confirm_channel="abf")
    detect_hfo2_multichannel_candidates(session, config)

    light_pred = session.candidate_points.loc[session.candidate_points["column_role"] == "light_atom", ["x_px", "y_px"]].to_numpy()
    light_recall = _recall_with_tolerance(truth["light"][["x_px", "y_px"]].to_numpy(), light_pred, tolerance=2.0)

    assert light_recall >= 0.75


def test_hfo2_multichannel_improves_light_recall_over_single_channel_baseline() -> None:
    images, truth = synthetic_hfo2_multichannel_bundle(rng_seed=11)

    baseline_session = AnalysisSession(name="baseline_idpc", raw_image=images["idpc"])
    preprocess_image(
        baseline_session,
        PreprocessConfig(contrast_mode="bright_peak", denoise_method="none", edge_mask_width=0),
    )
    detect_candidates(
        baseline_session,
        DetectionConfig(
            contrast_mode="bright_peak",
            gaussian_sigma=1.2,
            min_distance=12,
            threshold_rel=0.22,
            min_prominence=0.07,
            min_snr=1.6,
            edge_margin=4,
            patch_radius=11,
            dedupe_radius_px=15.0,
        ),
    )
    baseline_recall = _recall_with_tolerance(
        truth["light"][["x_px", "y_px"]].to_numpy(),
        baseline_session.candidate_points[["x_px", "y_px"]].to_numpy(),
        tolerance=2.0,
    )

    session = _build_multichannel_session(images)
    preprocess_channels(
        session,
        {
            "idpc": PreprocessConfig(contrast_mode="bright_peak", denoise_method="none", edge_mask_width=0),
            "haadf": PreprocessConfig(contrast_mode="bright_peak", denoise_method="none", edge_mask_width=0),
            "abf": PreprocessConfig(contrast_mode="dark_dip", denoise_method="none", edge_mask_width=0),
        },
    )
    detect_hfo2_multichannel_candidates(session, HfO2MultichannelDetectionConfig())
    multichannel_recall = _recall_with_tolerance(
        truth["light"][["x_px", "y_px"]].to_numpy(),
        session.candidate_points.loc[session.candidate_points["column_role"] == "light_atom", ["x_px", "y_px"]].to_numpy(),
        tolerance=2.0,
    )

    assert multichannel_recall - baseline_recall >= 0.15


def test_detect_hfo2_heavy_then_light_candidates_updates_stage_progressively() -> None:
    images, truth = synthetic_hfo2_multichannel_bundle(rng_seed=9)
    session = _build_multichannel_session(images)
    preprocess_channels(
        session,
        {
            "haadf": PreprocessConfig(contrast_mode="bright_peak", denoise_method="none", edge_mask_width=0),
        },
    )

    config = HfO2MultichannelDetectionConfig()
    detect_hfo2_heavy_candidates(session, config)

    heavy_pred = session.candidate_points.loc[session.candidate_points["column_role"] == "heavy_atom", ["x_px", "y_px"]].to_numpy()
    heavy_recall = _recall_with_tolerance(truth["heavy"][["x_px", "y_px"]].to_numpy(), heavy_pred, tolerance=2.0)
    assert session.current_stage == "heavy_reviewed"
    assert heavy_recall >= 0.95

    heavy_points = session.candidate_points.copy()
    preprocess_channels(
        session,
        {
            "idpc": PreprocessConfig(contrast_mode="bright_peak", denoise_method="none", edge_mask_width=0),
            "abf": PreprocessConfig(contrast_mode="dark_dip", denoise_method="none", edge_mask_width=0),
        },
    )
    detect_hfo2_light_candidates(session, config, heavy_points=heavy_points)

    light_pred = session.candidate_points.loc[session.candidate_points["column_role"] == "light_atom", ["x_px", "y_px"]].to_numpy()
    light_recall = _recall_with_tolerance(truth["light"][["x_px", "y_px"]].to_numpy(), light_pred, tolerance=2.0)
    assert session.current_stage == "detected"
    assert light_recall >= 0.80


def test_hfo2_heavy_manual_edit_preserves_heavy_metadata() -> None:
    session = _build_multichannel_session({"idpc": np.zeros((32, 32), dtype=float), "haadf": np.zeros((32, 32), dtype=float)})
    session.candidate_points = pd.DataFrame(
        {
            "candidate_id": [0, 1],
            "x_px": [10.0, 20.0],
            "y_px": [11.0, 21.0],
            "column_role": ["heavy_atom", "heavy_atom"],
            "seed_channel": ["haadf", "haadf"],
        }
    )
    viewer = _FakeViewer(np.asarray([[5.0, 6.0], [15.0, 16.0]], dtype=float))

    apply_hfo2_heavy_candidate_edits_from_viewer(session, viewer, heavy_channel="haadf")

    assert session.current_stage == "heavy_reviewed"
    assert set(session.candidate_points["column_role"]) == {"heavy_atom"}
    assert set(session.candidate_points["seed_channel"]) == {"haadf"}
    assert set(session.candidate_points["contrast_mode_used"]) == {"manual_edit"}


def test_hfo2_light_manual_edit_preserves_heavy_points_and_reassigns_parents() -> None:
    session = _build_multichannel_session({"idpc": np.zeros((32, 32), dtype=float), "haadf": np.zeros((32, 32), dtype=float)})
    heavy_points = pd.DataFrame(
        {
            "candidate_id": [0, 1],
            "x_px": [8.0, 24.0],
            "y_px": [8.0, 24.0],
            "x_local_px": [8.0, 24.0],
            "y_local_px": [8.0, 24.0],
            "contrast_mode_used": ["manual_edit", "manual_edit"],
            "column_role": ["heavy_atom", "heavy_atom"],
            "seed_channel": ["haadf", "haadf"],
            "confirm_channel": [pd.NA, pd.NA],
            "parent_heavy_id": [pd.NA, pd.NA],
            "support_score": [np.nan, np.nan],
            "confirm_score": [np.nan, np.nan],
        }
    )
    session.candidate_points = heavy_points.copy()
    viewer = _FakeViewer(np.asarray([[7.5, 8.5], [24.5, 23.0]], dtype=float))

    apply_hfo2_light_candidate_edits_from_viewer(
        session,
        viewer,
        heavy_points=heavy_points,
        light_channel="idpc",
    )

    light_points = session.candidate_points.loc[session.candidate_points["column_role"] == "light_atom"].reset_index(drop=True)
    assert session.current_stage == "detected"
    assert int((session.candidate_points["column_role"] == "heavy_atom").sum()) == 2
    assert int((session.candidate_points["column_role"] == "light_atom").sum()) == 2
    assert set(light_points["seed_channel"]) == {"idpc"}
    assert list(light_points["parent_heavy_id"]) == [0, 1]


def test_overlap_refinement_preserves_close_light_pair_metadata() -> None:
    images, truth = synthetic_hfo2_multichannel_bundle(rng_seed=13)
    session = _build_multichannel_session(images)
    preprocess_channels(
        session,
        {
            "idpc": PreprocessConfig(contrast_mode="bright_peak", denoise_method="none", edge_mask_width=0),
            "haadf": PreprocessConfig(contrast_mode="bright_peak", denoise_method="none", edge_mask_width=0),
            "abf": PreprocessConfig(contrast_mode="dark_dip", denoise_method="none", edge_mask_width=0),
        },
    )
    truth_light_coords = truth["light"][["x_px", "y_px"]].to_numpy(dtype=float)
    pair_distances, pair_indices = cKDTree(truth_light_coords).query(truth_light_coords, k=2)
    pair_anchor = int(np.argmin(pair_distances[:, 1]))
    close_pair = truth["light"].iloc[[pair_anchor, int(pair_indices[pair_anchor, 1])]].copy().reset_index(drop=True)
    session.candidate_points = pd.DataFrame(
        {
            "candidate_id": [0, 1],
            "x_px": close_pair["x_px"],
            "y_px": close_pair["y_px"],
            "column_role": ["light_atom", "light_atom"],
            "seed_channel": ["idpc", "idpc"],
            "confirm_channel": [pd.NA, pd.NA],
            "parent_heavy_id": [pd.NA, pd.NA],
            "contrast_mode_used": ["bright_peak", "bright_peak"],
        }
    )

    refine_points(
        session,
        RefinementConfig(
            mode="adaptive_atomap",
            fit_half_window=5,
            com_half_window=4,
            min_patch_radius_px=4,
            max_patch_radius_px=10,
            gaussian_retry_count=0,
            overlap_trigger_px=3.2,
        ),
    )

    refined_light = session.refined_points.loc[session.refined_points["column_role"] == "light_atom"]
    assert "overlap_shared_gaussian" in set(refined_light["refinement_path"])
    assert {"column_role", "seed_channel", "confirm_channel", "parent_heavy_id"}.issubset(session.refined_points.columns)
