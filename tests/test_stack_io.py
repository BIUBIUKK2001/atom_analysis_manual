from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from em_atom_workbench.io import load_image_stack


def test_load_npy_stack(tmp_path: Path) -> None:
    path = tmp_path / "stack.npy"
    stack = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    np.save(path, stack)

    loaded, metadata, calibration = load_image_stack(path)

    assert loaded.shape == (3, 4, 5)
    assert np.array_equal(loaded, stack)
    assert metadata["original_shape"] == (3, 4, 5)
    assert metadata["returned_shape"] == (3, 4, 5)
    assert metadata["reader"] == "numpy.load"
    assert calibration.size is None
    assert calibration.unit == "px"
    assert calibration.source == "metadata_missing"


def test_load_tif_stack(tmp_path: Path) -> None:
    path = tmp_path / "stack.tif"
    stack = np.arange(2 * 5 * 6, dtype=np.uint16).reshape(2, 5, 6)
    tifffile.imwrite(path, stack)

    loaded, metadata, _ = load_image_stack(path)

    assert loaded.shape == (2, 5, 6)
    assert np.array_equal(loaded, stack)
    assert metadata["reader"] == "tifffile.TiffFile.asarray"
    assert metadata["tiff_pages"] >= 1


def test_stack_axis_normalization(tmp_path: Path) -> None:
    path = tmp_path / "stack_axis.npy"
    data = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    np.save(path, data)

    loaded, metadata, _ = load_image_stack(path, stack_axis=2)

    assert loaded.shape == (3, 4, 5)
    assert np.array_equal(loaded[0], data[:, :, 0])
    assert metadata["stack_axis"] == 2


def test_invalid_2d_input_raises(tmp_path: Path) -> None:
    path = tmp_path / "image.npy"
    np.save(path, np.zeros((5, 5), dtype=np.float32))

    with pytest.raises(ValueError, match="3D"):
        load_image_stack(path)


def test_manual_calibration_fallback(tmp_path: Path) -> None:
    path = tmp_path / "stack.npy"
    np.save(path, np.zeros((2, 4, 4), dtype=np.float32))

    _, _, calibration = load_image_stack(path, manual_calibration={"size": 0.25, "unit": "nm"})

    assert calibration.size == pytest.approx(0.25)
    assert calibration.unit == "nm"
    assert calibration.source == "manual_override"
