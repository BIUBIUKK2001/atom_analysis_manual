from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, median_filter
from skimage import exposure
from skimage.restoration import denoise_bilateral, wiener

from .session import AnalysisSession, PreprocessConfig


def normalize_image(image: np.ndarray, quantiles: tuple[float, float] = (0.01, 0.99)) -> np.ndarray:
    working = np.asarray(image, dtype=np.float32)
    lower, upper = np.quantile(working, quantiles)
    if np.isclose(upper, lower):
        return np.zeros_like(working, dtype=np.float32)
    clipped = np.clip(working, lower, upper)
    return np.asarray((clipped - lower) / (upper - lower), dtype=np.float32)


def crop_image(image: np.ndarray, roi: tuple[int, int, int, int] | None) -> tuple[np.ndarray, tuple[int, int]]:
    if roi is None:
        return image, (0, 0)
    y0, y1, x0, x1 = roi
    return image[y0:y1, x0:x1], (x0, y0)


def edge_mask(shape: tuple[int, int], width: int) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    if width <= 0:
        return mask
    mask[:width, :] = False
    mask[-width:, :] = False
    mask[:, :width] = False
    mask[:, -width:] = False
    return mask


def build_gaussian_psf(size: int, sigma: float) -> np.ndarray:
    size = max(int(size), 3)
    if size % 2 == 0:
        size += 1
    sigma = max(float(sigma), 1e-3)

    radius = size // 2
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    psf = np.exp(-0.5 * (xx**2 + yy**2) / (sigma**2))
    psf_sum = float(np.sum(psf))
    if psf_sum <= 0:
        raise ValueError("Gaussian PSF sum must be positive.")
    return psf / psf_sum


def resolve_psf(config: PreprocessConfig) -> np.ndarray:
    if config.psf_kernel is not None:
        psf = np.asarray(config.psf_kernel, dtype=float)
        if psf.ndim != 2:
            raise ValueError("psf_kernel must be a 2D array.")
        total = float(np.sum(psf))
        if not np.isfinite(total) or np.isclose(total, 0.0):
            raise ValueError("psf_kernel must have a non-zero finite sum.")
        return psf / total
    return build_gaussian_psf(config.wiener_psf_size, config.wiener_psf_sigma)


def apply_wiener_filter(image: np.ndarray, config: PreprocessConfig) -> tuple[np.ndarray, np.ndarray]:
    psf = resolve_psf(config)
    filtered = wiener(
        image,
        psf=psf,
        balance=max(float(config.wiener_balance), 1e-8),
        clip=bool(config.wiener_clip),
    )
    filtered = np.asarray(filtered, dtype=np.float32)
    return filtered, psf


def _apply_filter(image: np.ndarray, config: PreprocessConfig) -> tuple[np.ndarray, dict[str, np.ndarray | str | float | int | bool]]:
    method = config.denoise_method.lower()
    diagnostics: dict[str, np.ndarray | str | float | int | bool] = {"filter_method": method}

    if method == "none":
        return image.copy(), diagnostics
    if method == "wiener":
        filtered, psf = apply_wiener_filter(image, config)
        diagnostics.update(
            {
                "wiener_psf": psf,
                "wiener_balance": float(config.wiener_balance),
                "wiener_psf_size": int(psf.shape[0]),
                "wiener_psf_sigma": float(config.wiener_psf_sigma),
                "wiener_clip": bool(config.wiener_clip),
            }
        )
        return filtered, diagnostics
    if method == "gaussian":
        if config.denoise_sigma <= 0:
            return image.copy(), diagnostics
        diagnostics["gaussian_sigma"] = float(config.denoise_sigma)
        return gaussian_filter(image, sigma=config.denoise_sigma), diagnostics
    if method == "median":
        size = max(int(config.median_size), 1)
        if size % 2 == 0:
            size += 1
        diagnostics["median_size"] = size
        return median_filter(image, size=size), diagnostics
    if method == "bilateral":
        diagnostics.update(
            {
                "bilateral_sigma_color": float(config.bilateral_sigma_color),
                "bilateral_sigma_spatial": float(config.bilateral_sigma_spatial),
            }
        )
        return (
            denoise_bilateral(
                image,
                sigma_color=max(float(config.bilateral_sigma_color), 1e-6),
                sigma_spatial=max(float(config.bilateral_sigma_spatial), 1e-6),
                channel_axis=None,
            ),
            diagnostics,
        )
    raise ValueError(f"Unsupported denoise_method: {config.denoise_method}")


def summarize_wiener_filter(session: AnalysisSession, config: PreprocessConfig) -> pd.DataFrame:
    pixel_size = session.pixel_calibration.size
    unit = session.pixel_calibration.unit
    size_phys = config.wiener_psf_size * pixel_size if pixel_size is not None else np.nan
    sigma_phys = config.wiener_psf_sigma * pixel_size if pixel_size is not None else np.nan
    phys_unit = unit if pixel_size is not None else "n/a"

    records = [
        {
            "parameter": "wiener_psf_size",
            "value": int(config.wiener_psf_size),
            "equivalent_physical": size_phys,
            "physical_unit": phys_unit,
            "meaning_cn": "\u9ad8\u65af PSF \u5377\u79ef\u6838\u7684\u8fb9\u957f\u50cf\u7d20\u6570",
            "guide_cn": "\u901a\u5e38\u53d6 7\u201311 px \u8f83\u7a33\u59a5\uff0c\u592a\u5927\u5bb9\u6613\u8fc7\u5ea6\u5e73\u6ed1\u6216\u5f15\u5165\u4e0d\u5fc5\u8981\u7684\u53cd\u5377\u79ef\u5047\u8c61\u3002",
        },
        {
            "parameter": "wiener_psf_sigma",
            "value": float(config.wiener_psf_sigma),
            "equivalent_physical": sigma_phys,
            "physical_unit": phys_unit,
            "meaning_cn": "\u9ad8\u65af PSF \u7684 sigma\uff0c\u53cd\u6620\u4f60\u5047\u8bbe\u7684\u6210\u50cf\u6a21\u7cca\u5bbd\u5ea6",
            "guide_cn": "\u503c\u8d8a\u5927\u8868\u793a\u5047\u5b9a\u7684\u6a21\u7cca\u8d8a\u91cd\uff1b\u771f\u5b9e\u4e0d\u786e\u5b9a\u65f6\uff0c\u5148\u4ece 0.8\u20131.2 px \u5f00\u59cb\u3002",
        },
        {
            "parameter": "wiener_balance",
            "value": float(config.wiener_balance),
            "equivalent_physical": np.nan,
            "physical_unit": "n/a",
            "meaning_cn": "Wiener \u6b63\u5219 / \u566a\u58f0\u5e73\u8861\u53c2\u6570",
            "guide_cn": "\u503c\u8d8a\u5c0f\u8d8a\u9510\uff0c\u4f46\u4e5f\u66f4\u5bb9\u6613\u653e\u5927\u566a\u58f0\u548c\u632f\u94c3\uff1b\u5efa\u8bae\u4ece 0.03\u20130.08 \u8fd9\u7c7b\u4fdd\u5b88\u533a\u95f4\u5c0f\u6b65\u8c03\u6574\u3002",
        },
    ]
    return pd.DataFrame(records)


def _preprocess_array(
    image: np.ndarray,
    config: PreprocessConfig,
) -> tuple[dict[str, np.ndarray | float | int | bool | tuple[int, int, int, int] | None | str], dict[str, np.ndarray | str | float | int | bool]]:
    cropped, (origin_x, origin_y) = crop_image(image, config.roi)
    normalized = normalize_image(cropped, config.normalization_quantiles)

    filter_input = normalized.copy()
    if config.contrast_mode == "dark_dip" and config.invert_for_dark_dip:
        filter_input = 1.0 - filter_input

    filtered, filter_diagnostics = _apply_filter(filter_input, config)
    processed = normalize_image(filtered, (0.01, 0.99))

    background = np.zeros_like(processed, dtype=np.float32)
    if config.background_sigma > 0:
        background = gaussian_filter(processed, sigma=config.background_sigma)
        processed = processed - background
        processed = normalize_image(processed, (0.01, 0.99))

    if config.local_contrast:
        processed = exposure.equalize_adapthist(processed, clip_limit=config.clahe_clip_limit)

    mask = edge_mask(processed.shape, config.edge_mask_width)
    processed_masked = np.asarray(processed.copy(), dtype=np.float32)
    processed_masked[~mask] = np.float32(np.nanmedian(processed))

    preprocess_result: dict[str, np.ndarray | float | int | bool | tuple[int, int, int, int] | None | str] = {
        "processed_image": processed_masked,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "roi": config.roi,
        "filter_method": filter_diagnostics.get("filter_method"),
        "wiener_balance": filter_diagnostics.get("wiener_balance"),
        "wiener_psf_size": filter_diagnostics.get("wiener_psf_size"),
        "wiener_psf_sigma": filter_diagnostics.get("wiener_psf_sigma"),
        "background_sigma": float(config.background_sigma),
        "local_contrast": bool(config.local_contrast),
    }
    if config.background_sigma > 0:
        preprocess_result["background_applied"] = True
    return preprocess_result, filter_diagnostics


def preprocess_image(
    session: AnalysisSession,
    config: PreprocessConfig,
    channel_name: str | None = None,
) -> AnalysisSession:
    channel_key = str(channel_name or session.primary_channel)
    raw_image = session.get_channel_state(channel_key).raw_image
    if raw_image is None:
        raise ValueError(f"Session does not contain raw_image for channel '{channel_key}'.")

    session.clear_downstream_results("preprocess")
    preprocess_result, filter_diagnostics = _preprocess_array(raw_image, config)
    session.set_channel_state(
        channel_key,
        preprocess_result=preprocess_result,
        contrast_mode=config.contrast_mode,
    )
    session.set_stage("loaded")
    session.record_step(
        "preprocess_image",
        parameters={"channel_name": channel_key, "config": config},
        notes={
            "origin_x": preprocess_result["origin_x"],
            "origin_y": preprocess_result["origin_y"],
            "processed_shape": np.asarray(preprocess_result["processed_image"]).shape,
            "filter_method": filter_diagnostics.get("filter_method"),
            "wiener_balance": filter_diagnostics.get("wiener_balance"),
        },
    )
    return session


def preprocess_channels(
    session: AnalysisSession,
    configs_by_channel: dict[str, PreprocessConfig],
) -> AnalysisSession:
    if not configs_by_channel:
        raise ValueError("configs_by_channel must contain at least one channel.")

    session.clear_downstream_results("preprocess")
    processed_results: dict[str, dict[str, np.ndarray | float | int | bool | tuple[int, int, int, int] | None | str]] = {}
    diagnostics_by_channel: dict[str, dict[str, np.ndarray | str | float | int | bool]] = {}
    reference_origin: tuple[int, int] | None = None
    reference_shape: tuple[int, int] | None = None

    for channel_name, config in configs_by_channel.items():
        raw_image = session.get_channel_state(channel_name).raw_image
        if raw_image is None:
            raise ValueError(f"Session does not contain raw_image for channel '{channel_name}'.")
        preprocess_result, diagnostics = _preprocess_array(raw_image, config)
        processed_shape = np.asarray(preprocess_result["processed_image"]).shape
        origin = (int(preprocess_result["origin_x"]), int(preprocess_result["origin_y"]))
        if reference_origin is None:
            reference_origin = origin
            reference_shape = processed_shape
        elif origin != reference_origin or processed_shape != reference_shape:
            raise ValueError("All channels must share the same ROI and processed image shape in preprocess_channels.")
        processed_results[channel_name] = preprocess_result
        diagnostics_by_channel[channel_name] = diagnostics

    for channel_name, preprocess_result in processed_results.items():
        session.set_channel_state(
            channel_name,
            preprocess_result=preprocess_result,
            contrast_mode=configs_by_channel[channel_name].contrast_mode,
        )

    session.set_stage("loaded")
    session.record_step(
        "preprocess_channels",
        parameters={name: config for name, config in configs_by_channel.items()},
        notes={
            "channel_names": list(configs_by_channel.keys()),
            "shared_origin": reference_origin,
            "processed_shape": reference_shape,
            "filter_methods": {
                name: diagnostics.get("filter_method")
                for name, diagnostics in diagnostics_by_channel.items()
            },
        },
    )
    return session
