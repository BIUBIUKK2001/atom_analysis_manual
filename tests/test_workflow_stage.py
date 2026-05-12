from __future__ import annotations

import numpy as np
import pandas as pd

from em_atom_workbench.annotation import save_annotations
from em_atom_workbench.curate import curate_points
from em_atom_workbench.detect import detect_candidates
from em_atom_workbench.metrics import compute_local_metrics
from em_atom_workbench.polarization import attach_vector_field
from em_atom_workbench.preprocess import preprocess_image
from em_atom_workbench.refine import refine_points
from em_atom_workbench.session import (
    AnalysisSession,
    CurationConfig,
    DetectionConfig,
    MetricsConfig,
    PreprocessConfig,
    RefinementConfig,
)
from em_atom_workbench.utils import synthetic_lattice_image


def test_preprocess_and_detect_update_session_stage() -> None:
    image, _ = synthetic_lattice_image(shape=(96, 96), spacing=16.0, noise_sigma=0.0, rng_seed=3)
    session = AnalysisSession(name="detect_stage", raw_image=image)
    session.set_stage("metrics")

    session = preprocess_image(
        session,
        PreprocessConfig(
            contrast_mode="bright_peak",
            denoise_method="none",
            edge_mask_width=0,
        ),
    )
    assert session.current_stage == "loaded"

    session = detect_candidates(
        session,
        DetectionConfig(
            contrast_mode="bright_peak",
            gaussian_sigma=0.0,
            min_distance=6,
            threshold_rel=0.05,
            min_prominence=0.0,
            min_snr=0.0,
            edge_margin=0,
            patch_radius=4,
        ),
    )

    assert session.current_stage == "detected"
    assert not session.candidate_points.empty


def test_curate_points_updates_stage_and_clears_downstream_results() -> None:
    session = AnalysisSession(name="curate_stage", raw_image=np.zeros((32, 32), dtype=float))
    session.candidate_points = pd.DataFrame(
        {
            "candidate_id": [0, 1],
            "x_px": [10.0, 20.0],
            "y_px": [10.0, 20.0],
            "quality_score": [0.9, 0.8],
            "fit_residual": [0.1, 0.1],
        }
    )
    session.neighbor_graph = {"edges": [(0, 1)]}
    session.local_metrics = pd.DataFrame({"atom_id": [0], "mean_nn_distance_px": [10.0]})
    session.vpcf_results = {"global_average_H": np.ones((3, 3))}
    session.annotations = {"records": [{"label": "old"}]}
    session.vector_fields = {
        "demo": pd.DataFrame({"x_px": [0.0], "y_px": [0.0], "u_px": [1.0], "v_px": [1.0], "magnitude_px": [1.4]})
    }

    session = curate_points(
        session,
        CurationConfig(
            duplicate_radius_px=0.5,
            min_quality_score=0.0,
            max_fit_residual=1.0,
        ),
    )

    assert session.current_stage == "curated"
    assert session.local_metrics.empty
    assert session.vpcf_results == {}
    assert session.annotations == {}
    assert session.vector_fields == {}


def test_refine_points_adaptive_mode_preserves_detected_stage() -> None:
    image = np.zeros((48, 48), dtype=float)
    session = AnalysisSession(name="refine_stage", raw_image=image)
    session.current_stage = "detected"
    session.preprocess_result = {"processed_image": image, "origin_x": 0, "origin_y": 0}
    session.candidate_points = pd.DataFrame(
        {
            "candidate_id": [0, 1],
            "x_px": [14.0, 30.0],
            "y_px": [16.0, 16.0],
        }
    )

    session = refine_points(
        session,
        RefinementConfig(
            mode="adaptive_atomap",
            fit_half_window=6,
            com_half_window=4,
            gaussian_retry_count=0,
        ),
    )

    assert session.current_stage == "detected"
    assert not session.refined_points.empty


def test_compute_local_metrics_updates_stage() -> None:
    session = AnalysisSession(name="metrics_stage", raw_image=np.zeros((40, 40), dtype=float))
    session.curated_points = pd.DataFrame(
        {
            "atom_id": [0, 1, 2],
            "x_px": [8.0, 16.0, 24.0],
            "y_px": [12.0, 12.0, 12.0],
        }
    )
    session.neighbor_graph = {
        "basis_table": pd.DataFrame(
            {
                "atom_id": [0, 1, 2],
                "basis_a_x": [8.0, 8.0, 8.0],
                "basis_a_y": [0.0, 0.0, 0.0],
                "basis_b_x": [0.0, 0.0, 0.0],
                "basis_b_y": [8.0, 8.0, 8.0],
                "basis_a_length_px": [8.0, 8.0, 8.0],
                "basis_b_length_px": [8.0, 8.0, 8.0],
                "basis_angle_deg": [90.0, 90.0, 90.0],
                "local_orientation_deg": [0.0, 0.0, 0.0],
            }
        ),
        "directed_neighbors": {0: [1], 1: [0, 2], 2: [1]},
    }

    session = compute_local_metrics(session, MetricsConfig())

    assert session.current_stage == "metrics"
    assert not session.local_metrics.empty


def test_annotation_and_vector_field_update_session_stage() -> None:
    session = AnalysisSession(name="annotation_stage", raw_image=np.zeros((32, 32), dtype=float))
    session.curated_points = pd.DataFrame(
        {
            "atom_id": [0, 1],
            "x_px": [8.0, 16.0],
            "y_px": [8.0, 16.0],
        }
    )
    session.local_metrics = pd.DataFrame(
        {
            "atom_id": [0, 1],
            "mean_nn_distance_px": [7.8, 8.1],
        }
    )

    session = save_annotations(
        session,
        polygons=[[[0.0, 0.0], [24.0, 0.0], [24.0, 24.0], [0.0, 24.0]]],
        labels=["domain_a"],
    )
    assert session.current_stage == "annotated"

    vector_field = pd.DataFrame(
        {
            "x_px": [8.0, 16.0],
            "y_px": [8.0, 16.0],
            "u_px": [0.5, 0.2],
            "v_px": [0.1, -0.1],
            "magnitude_px": [0.51, 0.22],
            "unit": ["px", "px"],
        }
    )
    session = attach_vector_field(session, "demo", vector_field)

    assert session.current_stage == "vector_field"
    assert "demo" in session.vector_fields
