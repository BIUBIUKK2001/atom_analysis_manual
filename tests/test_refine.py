import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

import em_atom_workbench.refine as refine_module
from em_atom_workbench.preprocess import preprocess_image
from em_atom_workbench.refine import refine_points, refine_points_by_class
from em_atom_workbench.session import AnalysisSession, PreprocessConfig, RefinementConfig
from em_atom_workbench.utils import synthetic_gaussian_image


def test_refine_points_legacy_recovers_subpixel_peak_positions():
    truth_positions = [(22.4, 19.7), (41.2, 38.6)]
    image = synthetic_gaussian_image(
        shape=(64, 64),
        peaks=[
            {"x": truth_positions[0][0], "y": truth_positions[0][1], "amplitude": 1.2, "sigma_x": 1.1, "sigma_y": 1.4, "theta": 0.3},
            {"x": truth_positions[1][0], "y": truth_positions[1][1], "amplitude": 1.0, "sigma_x": 1.3, "sigma_y": 1.0, "theta": -0.2},
        ],
        background=0.08,
        noise_sigma=0.01,
        rng_seed=7,
    )
    session = AnalysisSession(name="synthetic_refine", raw_image=image)
    preprocess_image(
        session,
        PreprocessConfig(
            contrast_mode="bright_peak",
            denoise_method="none",
            denoise_sigma=0.0,
            background_sigma=0.0,
        ),
    )
    session.candidate_points = pd.DataFrame(
        {
            "candidate_id": [0, 1],
            "x_px": [21.8, 40.5],
            "y_px": [20.4, 39.2],
        }
    )

    refine_points(
        session,
        RefinementConfig(
            mode="legacy",
            fit_half_window=6,
            initial_sigma_px=1.2,
            max_sigma_px=3.0,
            max_center_shift_px=3.0,
        ),
    )

    refined = session.refined_points.sort_values("atom_id").reset_index(drop=True)
    assert refined["fit_success"].sum() >= 1
    assert set(refined["refinement_method"]).issubset({"gaussian", "quadratic", "com"})
    assert set(refined["refinement_path"]).issubset({"legacy_gaussian", "quadratic_fallback", "com_fallback"})
    assert np.all(refined["gaussian_image_source"] == "processed")

    recovered = refined[["x_px", "y_px"]].to_numpy()
    truth = np.asarray(truth_positions, dtype=float)
    error = np.sqrt(np.sum((recovered - truth) ** 2, axis=1))
    assert np.max(error) < 0.5


def test_adaptive_refine_clamps_nn_radius_and_records_diagnostics():
    image = synthetic_gaussian_image(
        shape=(128, 128),
        peaks=[
            {"x": 15.2, "y": 40.0, "amplitude": 1.1, "sigma_x": 2.0, "sigma_y": 2.0, "theta": 0.0},
            {"x": 35.4, "y": 40.0, "amplitude": 1.0, "sigma_x": 2.0, "sigma_y": 2.0, "theta": 0.0},
            {"x": 82.1, "y": 40.0, "amplitude": 1.0, "sigma_x": 2.0, "sigma_y": 2.0, "theta": 0.0},
        ],
        background=0.02,
        noise_sigma=0.0,
        rng_seed=1,
    )
    session = AnalysisSession(name="adaptive_nn", raw_image=image)
    preprocess_image(
        session,
        PreprocessConfig(
            contrast_mode="bright_peak",
            denoise_method="none",
            background_sigma=0.0,
            edge_mask_width=0,
        ),
    )
    session.candidate_points = pd.DataFrame(
        {
            "candidate_id": [0, 1, 2],
            "x_px": [15.0, 35.0, 82.0],
            "y_px": [40.0, 40.0, 40.0],
        }
    )

    refine_points(
        session,
        RefinementConfig(
            mode="adaptive_atomap",
            fit_half_window=5,
            com_half_window=4,
            nn_radius_fraction=0.5,
            min_patch_radius_px=6,
            max_patch_radius_px=14,
            initial_sigma_px=2.0,
            max_sigma_px=5.0,
            gaussian_retry_count=0,
        ),
    )

    refined = session.refined_points.sort_values("atom_id").reset_index(drop=True)
    assert np.allclose(refined["nn_distance_px"].to_numpy()[:2], [20.0, 20.0], atol=0.5)
    assert refined.loc[0, "adaptive_half_window_px"] == 10
    assert refined.loc[1, "adaptive_half_window_px"] == 10
    assert refined.loc[2, "adaptive_half_window_px"] == 14
    assert np.all(refined["gaussian_attempt_count"] >= 1)
    assert set(refined["refinement_path"]).issubset({"adaptive_atomap", "quadratic_fallback", "com_fallback"})


def test_refine_points_by_class_uses_overrides_and_global_nn_context():
    image = synthetic_gaussian_image(
        shape=(110, 110),
        peaks=[
            {"x": 20.0, "y": 45.0, "amplitude": 1.0, "sigma_x": 1.4, "sigma_y": 1.4, "theta": 0.0},
            {"x": 26.0, "y": 45.0, "amplitude": 1.2, "sigma_x": 1.4, "sigma_y": 1.4, "theta": 0.0},
            {"x": 80.0, "y": 45.0, "amplitude": 0.9, "sigma_x": 1.4, "sigma_y": 1.4, "theta": 0.0},
        ],
        background=0.02,
        noise_sigma=0.0,
        rng_seed=12,
    )
    session = AnalysisSession(name="class_refine", raw_image=image)
    preprocess_image(session, PreprocessConfig(contrast_mode="bright_peak", denoise_method="none", edge_mask_width=0))
    session.candidate_points = pd.DataFrame(
        {
            "candidate_id": [0, 1, 2],
            "x_px": [20.0, 26.0, 80.0],
            "y_px": [45.0, 45.0, 45.0],
            "class_id": [0, 1, 2],
            "class_name": ["class_0", "class_1", "class_2"],
            "class_color": ["#111111", "#222222", "#333333"],
            "class_confidence": [0.9, 0.8, 0.7],
            "class_source": ["manual_review", "manual_review", "manual_review"],
            "class_reviewed": [True, True, True],
        }
    )

    refine_points_by_class(
        session,
        RefinementConfig(
            mode="adaptive_atomap",
            fit_half_window=5,
            com_half_window=4,
            nn_radius_fraction=0.5,
            min_patch_radius_px=2,
            max_patch_radius_px=20,
            initial_sigma_px=1.3,
            max_sigma_px=3.0,
            gaussian_retry_count=0,
            gaussian_image_source="raw",
        ),
        {1: {"nn_radius_fraction": 1.0}},
    )

    refined = session.refined_points.sort_values("candidate_id").reset_index(drop=True)
    assert {"class_id", "class_name", "class_color", "class_reviewed"}.issubset(refined.columns)
    assert refined["class_id"].tolist() == [0, 1, 2]
    assert refined["refinement_class_id"].tolist() == [0, 1, 2]
    assert refined["refinement_config_source"].tolist() == ["default", "class_1", "default"]
    assert refined["nn_context_mode"].tolist() == ["all", "all", "all"]
    assert np.allclose(refined["nn_distance_px"].to_numpy()[:2], [6.0, 6.0], atol=0.1)
    assert refined.loc[0, "adaptive_half_window_px"] == 3
    assert refined.loc[1, "adaptive_half_window_px"] == 6
    assert refined.loc[2, "adaptive_half_window_px"] == 20


def test_refine_points_by_class_same_class_context_ignores_cross_class_neighbors():
    image = synthetic_gaussian_image(
        shape=(110, 110),
        peaks=[
            {"x": 20.0, "y": 45.0, "amplitude": 1.0, "sigma_x": 1.4, "sigma_y": 1.4, "theta": 0.0},
            {"x": 26.0, "y": 45.0, "amplitude": 1.1, "sigma_x": 1.4, "sigma_y": 1.4, "theta": 0.0},
            {"x": 80.0, "y": 45.0, "amplitude": 0.9, "sigma_x": 1.4, "sigma_y": 1.4, "theta": 0.0},
        ],
        background=0.02,
        noise_sigma=0.0,
        rng_seed=13,
    )
    session = AnalysisSession(name="same_class_refine", raw_image=image)
    preprocess_image(session, PreprocessConfig(contrast_mode="bright_peak", denoise_method="none", edge_mask_width=0))
    session.candidate_points = pd.DataFrame(
        {
            "candidate_id": [0, 1, 2],
            "x_px": [20.0, 26.0, 80.0],
            "y_px": [45.0, 45.0, 45.0],
            "class_id": [0, 1, 0],
            "class_name": ["class_0", "class_1", "class_0"],
            "class_color": ["#111111", "#222222", "#111111"],
            "class_confidence": [0.9, 0.8, 0.7],
            "class_source": ["manual_review", "manual_review", "manual_review"],
            "class_reviewed": [True, True, True],
        }
    )

    refine_points_by_class(
        session,
        RefinementConfig(
            mode="adaptive_atomap",
            fit_half_window=5,
            com_half_window=4,
            nn_radius_fraction=0.5,
            min_patch_radius_px=2,
            max_patch_radius_px=40,
            initial_sigma_px=1.3,
            max_sigma_px=3.0,
            gaussian_retry_count=0,
            gaussian_image_source="raw",
        ),
        nn_context_mode="same_class",
    )

    refined = session.refined_points.sort_values("candidate_id").reset_index(drop=True)
    assert refined["nn_context_mode"].tolist() == ["same_class", "same_class", "same_class"]
    assert np.allclose(refined.loc[[0, 2], "nn_distance_px"].to_numpy(), [60.0, 60.0], atol=0.1)
    assert pd.isna(refined.loc[1, "nn_distance_px"])
    assert refined.loc[0, "adaptive_half_window_px"] == 30
    assert refined.loc[1, "adaptive_half_window_px"] == 5
    assert refined.loc[2, "adaptive_half_window_px"] == 30


def test_refine_points_by_class_rejects_invalid_nn_context_mode():
    session = AnalysisSession(name="bad_nn_context", raw_image=np.zeros((24, 24), dtype=float))
    preprocess_image(session, PreprocessConfig(contrast_mode="bright_peak", denoise_method="none", edge_mask_width=0))
    session.candidate_points = pd.DataFrame(
        {
            "candidate_id": [0],
            "x_px": [12.0],
            "y_px": [12.0],
            "class_id": [0],
        }
    )

    with pytest.raises(ValueError, match="nn_context_mode"):
        refine_points_by_class(session, RefinementConfig(), nn_context_mode="nearest_magic")


def test_refine_points_reverts_to_candidate_when_center_shift_exceeds_limit(monkeypatch):
    session = AnalysisSession(name="shift_guard", raw_image=np.zeros((40, 40), dtype=float))
    session.preprocess_result = {"processed_image": np.zeros((40, 40), dtype=float), "origin_x": 0, "origin_y": 0}
    session.candidate_points = pd.DataFrame({"candidate_id": [0], "x_px": [10.0], "y_px": [10.0]})

    def shifted_legacy_fit(*args, **kwargs):
        fitted = {
            "fit_success": True,
            "refinement_method": "gaussian",
            "x_patch": 18.0,
            "y_patch": 10.0,
            "amplitude": 1.0,
            "sigma_x": 1.0,
            "sigma_y": 1.0,
            "theta": 0.0,
            "local_background": 0.0,
            "fit_residual": 0.1,
        }
        return fitted, (0, 40, 0, 40), 1

    monkeypatch.setattr(refine_module, "_fit_patch_legacy", shifted_legacy_fit)

    refine_points(
        session,
        RefinementConfig(mode="legacy", fit_half_window=6, max_center_shift_px=2.0),
    )

    row = session.refined_points.iloc[0]
    assert row["x_px"] == pytest.approx(10.0)
    assert row["y_px"] == pytest.approx(10.0)
    assert row["x_fit_px"] == pytest.approx(18.0)
    assert row["y_fit_px"] == pytest.approx(10.0)
    assert row["attempted_center_shift_px"] == pytest.approx(8.0)
    assert row["center_shift_px"] == pytest.approx(0.0)
    assert bool(row["center_shift_rejected"]) is True
    assert row["position_source"] == "candidate_shift_guard"
    assert row["refinement_method"] == "gaussian"
    assert row["refinement_path"] == "legacy_gaussian"


def test_refine_points_by_class_applies_shift_guard_per_class(monkeypatch):
    session = AnalysisSession(name="class_shift_guard", raw_image=np.zeros((64, 64), dtype=float))
    session.preprocess_result = {"processed_image": np.zeros((64, 64), dtype=float), "origin_x": 0, "origin_y": 0}
    session.candidate_points = pd.DataFrame(
        {
            "candidate_id": [0, 1],
            "x_px": [10.0, 30.0],
            "y_px": [20.0, 20.0],
            "class_id": [0, 1],
            "class_name": ["class_0", "class_1"],
            "class_color": ["#111111", "#222222"],
            "class_confidence": [0.9, 0.8],
            "class_source": ["manual_review", "manual_review"],
            "class_reviewed": [True, True],
        }
    )

    def shifted_legacy_fit(image, center_xy, fit_half_window, origin_xy, config):
        fitted = {
            "fit_success": True,
            "refinement_method": "gaussian",
            "x_patch": float(center_xy[0]) + 2.0,
            "y_patch": float(center_xy[1]),
            "amplitude": 1.0,
            "sigma_x": 1.0,
            "sigma_y": 1.0,
            "theta": 0.0,
            "local_background": 0.0,
            "fit_residual": 0.1,
        }
        return fitted, (0, image.shape[1], 0, image.shape[0]), 1

    monkeypatch.setattr(refine_module, "_fit_patch_legacy", shifted_legacy_fit)

    refine_points_by_class(
        session,
        RefinementConfig(mode="legacy", fit_half_window=6, max_center_shift_px=1.0),
        {1: {"max_center_shift_px": 3.0}},
    )

    refined = session.refined_points.sort_values("candidate_id").reset_index(drop=True)
    assert refined["class_id"].tolist() == [0, 1]
    assert refined["refinement_config_source"].tolist() == ["default", "class_1"]
    assert refined["attempted_center_shift_px"].tolist() == pytest.approx([2.0, 2.0])
    assert refined["center_shift_px"].tolist() == pytest.approx([0.0, 2.0])
    assert refined["x_px"].tolist() == pytest.approx([10.0, 32.0])
    assert refined["position_source"].tolist() == ["candidate_shift_guard", "refined"]
    assert refined["center_shift_rejected"].tolist() == [True, False]


def test_adaptive_refine_is_more_accurate_for_wide_peaks_with_offset_seed():
    truth_positions = [(30.4, 32.7), (84.2, 90.3)]
    raw_image = synthetic_gaussian_image(
        shape=(128, 128),
        peaks=[
            {"x": truth_positions[0][0], "y": truth_positions[0][1], "amplitude": 1.2, "sigma_x": 4.3, "sigma_y": 4.0, "theta": 0.15},
            {"x": truth_positions[1][0], "y": truth_positions[1][1], "amplitude": 1.0, "sigma_x": 4.5, "sigma_y": 4.1, "theta": -0.2},
        ],
        background=0.05,
        noise_sigma=0.01,
        rng_seed=9,
    )
    processed_image = gaussian_filter(raw_image, sigma=2.5)
    candidate_points = pd.DataFrame(
        {
            "candidate_id": [0, 1],
            "x_px": [27.2, 87.4],
            "y_px": [29.5, 93.1],
        }
    )

    legacy_session = AnalysisSession(name="legacy_wide", raw_image=raw_image)
    legacy_session.preprocess_result = {"processed_image": processed_image, "origin_x": 0, "origin_y": 0}
    legacy_session.candidate_points = candidate_points.copy()

    adaptive_session = AnalysisSession(name="adaptive_wide", raw_image=raw_image)
    adaptive_session.preprocess_result = {"processed_image": processed_image, "origin_x": 0, "origin_y": 0}
    adaptive_session.candidate_points = candidate_points.copy()

    legacy_config = RefinementConfig(
        mode="legacy",
        fit_half_window=4,
        initial_sigma_px=1.2,
        max_sigma_px=8.0,
        max_center_shift_px=6.0,
    )
    adaptive_config = RefinementConfig(
        mode="adaptive_atomap",
        fit_half_window=4,
        com_half_window=4,
        nn_radius_fraction=0.35,
        min_patch_radius_px=6,
        max_patch_radius_px=14,
        initial_sigma_px=1.2,
        max_sigma_px=8.0,
        max_center_shift_px=6.0,
        gaussian_retry_count=0,
        gaussian_image_source="raw",
    )

    refine_points(legacy_session, legacy_config)
    refine_points(adaptive_session, adaptive_config)

    truth = np.asarray(truth_positions, dtype=float)
    legacy_error = np.sqrt(np.sum((legacy_session.refined_points[["x_px", "y_px"]].to_numpy() - truth) ** 2, axis=1))
    adaptive_error = np.sqrt(np.sum((adaptive_session.refined_points[["x_px", "y_px"]].to_numpy() - truth) ** 2, axis=1))

    assert np.max(adaptive_error) <= np.max(legacy_error)
    assert np.max(adaptive_error) < 1.0


def test_adaptive_refine_retries_then_falls_back_to_quadratic(monkeypatch):
    image = synthetic_gaussian_image(
        shape=(48, 48),
        peaks=[{"x": 24.0, "y": 24.0, "amplitude": 1.0, "sigma_x": 1.5, "sigma_y": 1.5, "theta": 0.0}],
        background=0.01,
        noise_sigma=0.0,
        rng_seed=2,
    )
    session = AnalysisSession(name="retry_fallback", raw_image=image)
    preprocess_image(session, PreprocessConfig(contrast_mode="bright_peak", denoise_method="none", edge_mask_width=0))
    session.candidate_points = pd.DataFrame({"candidate_id": [0], "x_px": [24.0], "y_px": [24.0]})

    def always_fail(*args, **kwargs):
        raise RuntimeError("forced gaussian failure")

    monkeypatch.setattr(refine_module, "curve_fit", always_fail)

    refine_points(
        session,
        RefinementConfig(
            mode="adaptive_atomap",
            fit_half_window=8,
            com_half_window=4,
            min_patch_radius_px=6,
            gaussian_retry_count=2,
            gaussian_retry_shrink_factor=0.9,
            fallback_to_quadratic=True,
            fallback_to_com=False,
        ),
    )

    row = session.refined_points.iloc[0]
    assert row["gaussian_attempt_count"] == 3
    assert row["refinement_method"] == "quadratic"
    assert row["refinement_path"] == "quadratic_fallback"
    assert row["adaptive_half_window_px"] == 6


def test_adaptive_refine_uses_processed_image_when_raw_missing():
    processed_image = synthetic_gaussian_image(
        shape=(64, 64),
        peaks=[{"x": 28.5, "y": 30.2, "amplitude": 1.0, "sigma_x": 2.2, "sigma_y": 2.0, "theta": 0.0}],
        background=0.02,
        noise_sigma=0.0,
        rng_seed=4,
    )
    session = AnalysisSession(name="processed_only", raw_image=None)
    session.preprocess_result = {"processed_image": processed_image, "origin_x": 0, "origin_y": 0}
    session.candidate_points = pd.DataFrame({"candidate_id": [0], "x_px": [28.0], "y_px": [30.0]})

    refine_points(
        session,
        RefinementConfig(
            mode="adaptive_atomap",
            fit_half_window=7,
            com_half_window=4,
            min_patch_radius_px=6,
            gaussian_retry_count=0,
        ),
    )

    row = session.refined_points.iloc[0]
    assert pd.isna(row["nn_distance_px"])
    assert row["gaussian_image_source"] == "processed"
    assert row["adaptive_half_window_px"] == 7
