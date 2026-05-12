import numpy as np
import pandas as pd

from em_atom_workbench.preprocess import build_gaussian_psf, preprocess_image, summarize_wiener_filter
from em_atom_workbench.session import AnalysisSession, PreprocessConfig
from em_atom_workbench.utils import synthetic_lattice_image


def test_build_gaussian_psf_is_normalized():
    psf = build_gaussian_psf(size=8, sigma=1.1)

    assert psf.shape == (9, 9)
    assert np.isclose(np.sum(psf), 1.0)
    assert float(psf[4, 4]) == float(np.max(psf))


def test_preprocess_wiener_stores_diagnostics_and_clears_downstream():
    image, _ = synthetic_lattice_image(shape=(96, 96), spacing=14.0, noise_sigma=0.01, rng_seed=5)
    session = AnalysisSession(name="synthetic_wiener", raw_image=image)
    session.candidate_points = pd.DataFrame({"x_px": [10.0], "y_px": [10.0]})

    preprocess_image(
        session,
        PreprocessConfig(
            contrast_mode="bright_peak",
            denoise_method="wiener",
            wiener_psf_size=9,
            wiener_psf_sigma=1.0,
            wiener_balance=0.05,
            background_sigma=0.0,
            edge_mask_width=2,
        ),
    )

    result = session.preprocess_result
    assert result["processed_image"].shape == image.shape
    assert result["filter_method"] == "wiener"
    assert result["wiener_balance"] == 0.05
    assert result["wiener_psf_size"] == 9
    assert np.isclose(result["wiener_psf_sigma"], 1.0)
    assert session.candidate_points.empty


def test_summarize_wiener_filter_reports_physical_scale():
    image, _ = synthetic_lattice_image(shape=(64, 64), spacing=12.0, noise_sigma=0.0, rng_seed=0)
    session = AnalysisSession(name="summary_case", raw_image=image)
    session.pixel_calibration.size = 0.2
    session.pixel_calibration.unit = "A"

    summary = summarize_wiener_filter(
        session,
        PreprocessConfig(wiener_psf_size=9, wiener_psf_sigma=1.1, wiener_balance=0.04),
    )

    assert list(summary["parameter"]) == ["wiener_psf_size", "wiener_psf_sigma", "wiener_balance"]
    assert np.isclose(float(summary.loc[0, "equivalent_physical"]), 1.8)
