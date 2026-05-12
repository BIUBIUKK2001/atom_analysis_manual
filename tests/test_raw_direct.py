from __future__ import annotations

import numpy as np

from em_atom_workbench.detect import (
    detect_candidates,
    detect_hfo2_heavy_candidates,
    detect_hfo2_light_candidates,
)
from em_atom_workbench.refine import refine_points
from em_atom_workbench.session import (
    AnalysisSession,
    DetectionConfig,
    HfO2MultichannelDetectionConfig,
    RefinementConfig,
)
from em_atom_workbench.utils import synthetic_gaussian_image, synthetic_hfo2_multichannel_bundle


def _build_multichannel_session(images: dict[str, np.ndarray]) -> AnalysisSession:
    session = AnalysisSession(
        name="raw_direct_multichannel",
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
    session.set_workflow(
        "hfo2_multichannel",
        {
            "primary_channel": "idpc",
            "heavy_channel": "haadf",
            "light_channel": "idpc",
            "confirm_channel": "abf" if "abf" in images else None,
        },
    )
    return session


def test_single_channel_raw_direct_detection_then_refinement() -> None:
    image = synthetic_gaussian_image(
        shape=(96, 96),
        peaks=[
            {"x": 24.2, "y": 28.5, "amplitude": 1.2, "sigma_x": 1.3, "sigma_y": 1.3, "theta": 0.0},
            {"x": 49.8, "y": 50.1, "amplitude": 1.1, "sigma_x": 1.4, "sigma_y": 1.2, "theta": 0.2},
            {"x": 71.0, "y": 66.4, "amplitude": 1.0, "sigma_x": 1.2, "sigma_y": 1.2, "theta": -0.1},
        ],
        background=0.05,
        noise_sigma=0.01,
        rng_seed=3,
    )
    session = AnalysisSession(name="raw_direct_single", raw_image=image)

    detect_candidates(
        session,
        DetectionConfig(
            contrast_mode="bright_peak",
            gaussian_sigma=1.0,
            min_distance=10,
            threshold_rel=0.15,
            min_prominence=0.03,
            min_snr=0.8,
            edge_margin=4,
            patch_radius=8,
            dedupe_radius_px=10.0,
        ),
    )

    assert session.current_stage == "detected"
    assert session.preprocess_result == {}
    assert len(session.candidate_points) >= 3

    refine_points(
        session,
        RefinementConfig(
            mode="adaptive_atomap",
            fit_half_window=6,
            com_half_window=4,
            min_patch_radius_px=5,
            max_patch_radius_px=12,
            gaussian_retry_count=0,
        ),
    )

    assert session.current_stage == "detected"
    assert len(session.refined_points) == len(session.candidate_points)
    assert not session.refined_points.empty
    assert set(session.refined_points["gaussian_image_source"]) == {"raw"}


def test_multichannel_raw_direct_detection_progresses_without_preprocess() -> None:
    images, _ = synthetic_hfo2_multichannel_bundle(rng_seed=6)
    session = _build_multichannel_session(images)
    config = HfO2MultichannelDetectionConfig()

    assert all(not session.get_channel_state(name).preprocess_result for name in session.list_channels())

    detect_hfo2_heavy_candidates(session, config)

    heavy_points = session.candidate_points.copy()
    assert session.current_stage == "heavy_reviewed"
    assert not heavy_points.empty
    assert set(heavy_points["column_role"]) == {"heavy_atom"}

    detect_hfo2_light_candidates(session, config, heavy_points=heavy_points)

    roles = set(session.candidate_points["column_role"])
    assert session.current_stage == "detected"
    assert {"heavy_atom", "light_atom"}.issubset(roles)
    assert all(not session.get_channel_state(name).preprocess_result for name in session.list_channels())
