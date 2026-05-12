import numpy as np
from scipy.spatial import cKDTree

from em_atom_workbench.detect import detect_candidates
from em_atom_workbench.preprocess import preprocess_image
from em_atom_workbench.session import AnalysisSession, DetectionConfig, PreprocessConfig
from em_atom_workbench.utils import synthetic_lattice_image


def _recall_with_tolerance(reference, predicted, tolerance):
    if len(reference) == 0 or len(predicted) == 0:
        return 0.0
    tree = cKDTree(predicted)
    distances, _ = tree.query(reference, k=1)
    return float(np.mean(distances <= tolerance))


def test_detect_candidates_on_synthetic_lattice():
    image, truth = synthetic_lattice_image(spacing=14.0, noise_sigma=0.02, background=0.15, rng_seed=3)
    session = AnalysisSession(name="synthetic_detect", raw_image=image)
    preprocess_image(
        session,
        PreprocessConfig(
            contrast_mode="bright_peak",
            denoise_method="gaussian",
            denoise_sigma=0.4,
            background_sigma=0.0,
        ),
    )
    detect_candidates(
        session,
        DetectionConfig(
            contrast_mode="bright_peak",
            gaussian_sigma=0.8,
            min_distance=8,
            threshold_rel=0.15,
            min_prominence=0.05,
            min_snr=1.0,
            edge_margin=4,
        ),
    )

    assert len(session.candidate_points) >= int(0.8 * len(truth))
    recall = _recall_with_tolerance(
        truth[["x_px", "y_px"]].to_numpy(),
        session.candidate_points[["x_px", "y_px"]].to_numpy(),
        tolerance=2.0,
    )
    assert recall >= 0.8


def test_detect_candidates_supports_dark_dip_mode():
    image, truth = synthetic_lattice_image(spacing=15.0, noise_sigma=0.01, background=0.85, rng_seed=4)
    image = 1.0 - image
    session = AnalysisSession(name="synthetic_dark", raw_image=image)
    preprocess_image(
        session,
        PreprocessConfig(
            contrast_mode="dark_dip",
            denoise_method="gaussian",
            denoise_sigma=0.3,
            background_sigma=0.0,
        ),
    )
    detect_candidates(
        session,
        DetectionConfig(
            contrast_mode="dark_dip",
            gaussian_sigma=0.7,
            min_distance=9,
            threshold_rel=0.12,
            min_prominence=0.04,
            min_snr=0.8,
            edge_margin=4,
        ),
    )

    assert len(session.candidate_points) >= int(0.75 * len(truth))
