from __future__ import annotations

from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
import tifffile
from skimage import io as skio

from .session import AnalysisSession, PixelCalibration
from .utils import slugify


def _coerce_to_2d_image(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image)
    data = np.squeeze(data)
    if data.ndim == 2:
        return data
    if data.ndim == 3 and data.shape[-1] in {3, 4}:
        return data[..., :3].mean(axis=-1, dtype=np.float32)
    raise ValueError(f"Only 2D images are supported in v1. Received shape {data.shape}.")


def _safe_metadata_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return value.decode(errors="ignore")
    if isinstance(value, dict):
        return {str(key): _safe_metadata_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_metadata_value(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        array = np.asarray(value)
    except Exception:
        array = None

    candidate = value
    if array is not None:
        if array.size == 0:
            return None
        candidate = array.reshape(-1)[0]

    try:
        result = float(candidate)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(result) or np.isclose(result, 0.0):
        return None
    return result


def _mrc_header_to_dict(header: Any) -> dict[str, Any]:
    field_names = getattr(getattr(header, "dtype", None), "names", None)
    if not field_names:
        return {"raw_header": _safe_metadata_value(header)}
    return {str(name): _safe_metadata_value(header[name]) for name in field_names}


def _manual_calibration_to_object(manual_calibration: PixelCalibration | dict[str, Any] | float | None) -> PixelCalibration:
    if manual_calibration is None:
        return PixelCalibration()
    if isinstance(manual_calibration, PixelCalibration):
        return PixelCalibration(
            size=manual_calibration.size,
            unit=manual_calibration.unit,
            source="manual_override",
        )
    if isinstance(manual_calibration, (int, float)):
        return PixelCalibration(size=float(manual_calibration), unit="arb.", source="manual_override")
    if isinstance(manual_calibration, dict):
        return PixelCalibration(
            size=manual_calibration.get("size"),
            unit=manual_calibration.get("unit", "arb."),
            source="manual_override",
        )
    raise TypeError("manual_calibration must be None, PixelCalibration, dict, or float.")


def _select_hyperspy_signal(loaded: Any, dataset_index: int | None) -> tuple[Any, int | None]:
    if not isinstance(loaded, (list, tuple)):
        return loaded, dataset_index

    if dataset_index is not None:
        if dataset_index < 0 or dataset_index >= len(loaded):
            raise IndexError(f"dataset_index {dataset_index} is out of range for {len(loaded)} loaded datasets.")
        return loaded[int(dataset_index)], int(dataset_index)

    candidates = [
        (idx, signal)
        for idx, signal in enumerate(loaded)
        if getattr(getattr(signal, "axes_manager", None), "signal_dimension", None) == 2
        and np.squeeze(np.asarray(getattr(signal, "data", np.empty((0, 0))))).ndim == 2
    ]
    if len(candidates) == 1:
        idx, signal = candidates[0]
        return signal, int(idx)

    descriptions = []
    for idx, signal in enumerate(loaded):
        shape = getattr(getattr(signal, "data", None), "shape", None)
        signal_dimension = getattr(getattr(signal, "axes_manager", None), "signal_dimension", None)
        descriptions.append(f"{idx}: shape={shape}, signal_dimension={signal_dimension}")
    raise ValueError(
        "Multiple datasets were found in the DM file. "
        "Please specify dataset_index in the notebook config. "
        f"Available datasets: {'; '.join(descriptions)}"
    )


def _calibration_from_signal_axes(signal: Any) -> PixelCalibration | None:
    axes = list(getattr(getattr(signal, "axes_manager", None), "signal_axes", []) or [])
    if len(axes) < 2:
        return None
    x_axis = axes[-1]
    y_axis = axes[-2]
    x_scale = _coerce_optional_float(getattr(x_axis, "scale", None))
    y_scale = _coerce_optional_float(getattr(y_axis, "scale", None))
    x_unit = str(getattr(x_axis, "units", None) or "").strip()
    y_unit = str(getattr(y_axis, "units", None) or "").strip()
    if x_scale is None or y_scale is None:
        return None
    if not np.isclose(x_scale, y_scale):
        return None
    if x_unit and y_unit and x_unit != y_unit:
        return None
    return PixelCalibration(
        size=float(x_scale),
        unit=x_unit or y_unit or "arb.",
        source="hyperspy_axes",
    )


def _calibration_from_dm_original_metadata(original_metadata: dict[str, Any]) -> PixelCalibration | None:
    image_list = original_metadata.get("ImageList", {}) if isinstance(original_metadata, dict) else {}
    if not isinstance(image_list, dict):
        return None

    for image_group in image_list.values():
        if not isinstance(image_group, dict):
            continue
        dimension_group = (
            image_group.get("ImageData", {})
            .get("Calibrations", {})
            .get("Dimension", {})
        )
        if not isinstance(dimension_group, dict):
            continue

        scales: list[float] = []
        units: list[str] = []
        for entry in dimension_group.values():
            if not isinstance(entry, dict):
                continue
            scale = _coerce_optional_float(entry.get("Scale"))
            unit = str(entry.get("Units", "") or "").strip()
            if scale is None:
                continue
            scales.append(scale)
            units.append(unit or "arb.")

        if not scales:
            continue
        if len(scales) >= 2 and not np.isclose(scales[0], scales[1]):
            continue
        if len(units) >= 2 and units[0] != units[1]:
            continue

        return PixelCalibration(
            size=float(scales[0]),
            unit=units[0] if units else "arb.",
            source="dm_original_metadata_calibration",
        )
    return None


def _extract_hyperspy(path: Path, dataset_index: int | None = None) -> tuple[np.ndarray, dict[str, Any], PixelCalibration, int | None]:
    try:
        import hyperspy.api as hs
    except ImportError as exc:
        raise ImportError("DM3/DM4 loading requires hyperspy and rosettasciio.") from exc

    loaded = hs.load(str(path))
    signal, selected_dataset_index = _select_hyperspy_signal(loaded, dataset_index)
    data = _coerce_to_2d_image(signal.data)
    metadata_dict = _safe_metadata_value(signal.metadata.as_dictionary())
    original_metadata_dict = _safe_metadata_value(signal.original_metadata.as_dictionary())
    metadata = {
        "metadata": metadata_dict,
        "original_metadata": original_metadata_dict,
        "dataset_index": selected_dataset_index,
    }

    calibration = _calibration_from_signal_axes(signal)
    if calibration is None:
        calibration = _calibration_from_dm_original_metadata(original_metadata_dict)
    if calibration is None:
        calibration = PixelCalibration(size=None, unit="px", source="metadata_missing")
    return data, metadata, calibration, selected_dataset_index


def _extract_tiff(path: Path) -> tuple[np.ndarray, dict[str, Any], PixelCalibration]:
    with tifffile.TiffFile(path) as tif:
        image = _coerce_to_2d_image(tif.asarray())
        metadata = {
            "tiff_pages": len(tif.pages),
            "tiff_tags": {
                tag.name: _safe_metadata_value(tag.value)
                for tag in tif.pages[0].tags.values()
            },
            "ome_metadata": _safe_metadata_value(getattr(tif, "ome_metadata", None)),
        }
        calibration = PixelCalibration()
        x_resolution = tif.pages[0].tags.get("XResolution")
        resolution_unit = tif.pages[0].tags.get("ResolutionUnit")
        if x_resolution is not None and resolution_unit is not None:
            try:
                numerator, denominator = x_resolution.value
                pixels_per_unit = numerator / denominator
                if pixels_per_unit > 0:
                    calibration = PixelCalibration(
                        size=1.0 / pixels_per_unit,
                        unit=str(resolution_unit.value),
                        source="tiff_resolution",
                    )
            except Exception:
                calibration = PixelCalibration()
    return image, metadata, calibration


def _extract_mrc(path: Path) -> tuple[np.ndarray, dict[str, Any], PixelCalibration]:
    with mrcfile.open(path, permissive=True) as handle:
        image = _coerce_to_2d_image(handle.data)
        voxel_size = getattr(handle, "voxel_size", None)
        voxel_x = getattr(voxel_size, "x", None)
        voxel_y = getattr(voxel_size, "y", None)
        voxel_z = getattr(voxel_size, "z", None)
        pixel_size = _coerce_optional_float(voxel_x)
        metadata = {
            "header": _mrc_header_to_dict(handle.header),
            "voxel_size": _safe_metadata_value(
                {
                    "x": voxel_x,
                    "y": voxel_y,
                    "z": voxel_z,
                }
            ),
        }
        calibration = PixelCalibration(size=pixel_size, unit="A", source="mrc_voxel_size" if pixel_size else "metadata_missing")
    return image, metadata, calibration


def _extract_generic(path: Path) -> tuple[np.ndarray, dict[str, Any], PixelCalibration]:
    image = _coerce_to_2d_image(skio.imread(path))
    metadata = {"reader": "skimage.io.imread", "suffix": path.suffix.lower()}
    return image, metadata, PixelCalibration()


def _load_image_components(path: str | Path, dataset_index: int | None = None) -> tuple[np.ndarray, dict[str, Any], PixelCalibration, int | None]:
    image_path = Path(path)
    suffix = image_path.suffix.lower()
    selected_dataset_index = dataset_index

    if suffix in {".dm3", ".dm4"}:
        image, metadata, calibration, selected_dataset_index = _extract_hyperspy(image_path, dataset_index=dataset_index)
    elif suffix in {".tif", ".tiff"}:
        image, metadata, calibration = _extract_tiff(image_path)
    elif suffix == ".mrc":
        image, metadata, calibration = _extract_mrc(image_path)
    else:
        image, metadata, calibration = _extract_generic(image_path)
    return image, metadata, calibration, selected_dataset_index


def _merge_bundle_calibration(
    calibrations: dict[str, PixelCalibration],
    manual_calibration: PixelCalibration | dict[str, Any] | float | None,
) -> PixelCalibration:
    valid = {
        name: calibration
        for name, calibration in calibrations.items()
        if calibration.is_calibrated and calibration.size is not None
    }
    if valid:
        first_name, first_calibration = next(iter(valid.items()))
        for name, calibration in valid.items():
            if calibration.unit != first_calibration.unit or not np.isclose(float(calibration.size), float(first_calibration.size)):
                raise ValueError(
                    "All bundle channels must share the same pixel calibration. "
                    f"Mismatch between '{first_name}' and '{name}'."
                )
        return PixelCalibration(
            size=float(first_calibration.size),
            unit=first_calibration.unit,
            source=first_calibration.source,
        )

    manual = _manual_calibration_to_object(manual_calibration)
    if manual.size is not None:
        return manual
    return PixelCalibration(size=None, unit="px", source="metadata_missing")


def load_image(
    path: str | Path,
    manual_calibration: PixelCalibration | dict[str, Any] | float | None = None,
    session_name: str | None = None,
    contrast_mode: str = "bright_peak",
    dataset_index: int | None = None,
) -> AnalysisSession:
    image_path = Path(path)
    image, metadata, calibration, selected_dataset_index = _load_image_components(image_path, dataset_index=dataset_index)

    manual = _manual_calibration_to_object(manual_calibration)
    if not calibration.is_calibrated and manual.size is not None:
        calibration = manual

    session = AnalysisSession(
        name=session_name or slugify(image_path.stem),
        input_path=str(image_path),
        dataset_index=selected_dataset_index,
        raw_image=image,
        raw_metadata=metadata,
        pixel_calibration=calibration,
        contrast_mode=contrast_mode,
    )
    session.set_stage("loaded")
    session.record_step(
        "load_image",
        parameters={
            "path": str(image_path),
            "contrast_mode": contrast_mode,
            "dataset_index": selected_dataset_index,
        },
        notes={"image_shape": image.shape, "calibration": calibration, "calibration_source": calibration.source},
    )
    return session


def load_image_bundle(
    paths: dict[str, str | Path],
    *,
    primary_channel: str = "idpc",
    manual_calibration: PixelCalibration | dict[str, Any] | float | None = None,
    session_name: str | None = None,
    contrast_modes: dict[str, str] | None = None,
    dataset_indices: dict[str, int | None] | None = None,
) -> AnalysisSession:
    if not paths:
        raise ValueError("paths must contain at least one channel.")
    if primary_channel not in paths:
        raise KeyError(f"primary_channel '{primary_channel}' was not found in paths.")

    contrast_modes = dict(contrast_modes or {})
    dataset_indices = dict(dataset_indices or {})

    channel_payloads: dict[str, dict[str, Any]] = {}
    calibrations: dict[str, PixelCalibration] = {}
    reference_shape: tuple[int, int] | None = None

    for channel_name, channel_path in paths.items():
        image, metadata, calibration, selected_dataset_index = _load_image_components(
            channel_path,
            dataset_index=dataset_indices.get(channel_name),
        )
        if reference_shape is None:
            reference_shape = image.shape
        elif image.shape != reference_shape:
            raise ValueError(
                "All bundle channels must share the same image shape. "
                f"Expected {reference_shape}, got {image.shape} for '{channel_name}'."
            )
        channel_payloads[channel_name] = {
            "input_path": str(Path(channel_path)),
            "dataset_index": selected_dataset_index,
            "raw_image": image,
            "raw_metadata": metadata,
            "contrast_mode": contrast_modes.get(channel_name, "bright_peak"),
        }
        calibrations[channel_name] = calibration

    shared_calibration = _merge_bundle_calibration(calibrations, manual_calibration)
    primary_payload = channel_payloads[primary_channel]
    session = AnalysisSession(
        name=session_name or slugify(Path(primary_payload["input_path"]).stem),
        input_path=primary_payload["input_path"],
        dataset_index=primary_payload["dataset_index"],
        raw_image=primary_payload["raw_image"],
        raw_metadata=primary_payload["raw_metadata"],
        pixel_calibration=shared_calibration,
        contrast_mode=primary_payload["contrast_mode"],
        primary_channel=primary_channel,
    )
    for channel_name, payload in channel_payloads.items():
        session.set_channel_state(
            channel_name,
            input_path=payload["input_path"],
            dataset_index=payload["dataset_index"],
            raw_image=payload["raw_image"],
            raw_metadata=payload["raw_metadata"],
            contrast_mode=payload["contrast_mode"],
        )
    session.set_primary_channel(primary_channel)
    session.set_stage("loaded")
    session.record_step(
        "load_image_bundle",
        parameters={
            "paths": {key: str(Path(value)) for key, value in paths.items()},
            "primary_channel": primary_channel,
            "contrast_modes": {key: contrast_modes.get(key, "bright_peak") for key in paths},
            "dataset_indices": dataset_indices,
        },
        notes={
            "channel_names": list(paths.keys()),
            "image_shape": reference_shape,
            "calibration": shared_calibration,
            "calibration_source": shared_calibration.source,
        },
    )
    return session
