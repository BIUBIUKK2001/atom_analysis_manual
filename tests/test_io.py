from pathlib import Path

import mrcfile
import numpy as np

from em_atom_workbench.io import load_image


def test_load_image_reads_mrc_voxel_size_from_numpy_scalar_wrapper(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.mrc"
    data = np.zeros((16, 16), dtype=np.float32)
    data[8, 8] = 1.0

    with mrcfile.new(image_path, overwrite=True) as handle:
        handle.set_data(data)
        handle.voxel_size = (0.25, 0.25, 1.0)

    session = load_image(image_path)

    assert session.raw_image.shape == (16, 16)
    assert session.pixel_calibration.size == 0.25
    assert session.pixel_calibration.unit == "A"
    assert session.pixel_calibration.source == "mrc_voxel_size"


def test_load_image_reads_mrc_without_voxel_size_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "no_calibration.mrc"
    data = np.ones((8, 8), dtype=np.float32)

    with mrcfile.new(image_path, overwrite=True) as handle:
        handle.set_data(data)

    session = load_image(image_path)

    assert session.raw_image.shape == (8, 8)
    assert session.pixel_calibration.size is None
    assert session.pixel_calibration.source == "metadata_missing"
