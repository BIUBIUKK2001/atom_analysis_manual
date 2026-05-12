from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pytest

from em_atom_workbench.io import load_image


class _FakeMetadata:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def as_dictionary(self) -> dict:
        return self._payload


class _FakeAxis:
    def __init__(self, scale: float | None = None, units: str | None = None) -> None:
        self.scale = scale
        self.units = units


class _FakeAxesManager:
    def __init__(self, signal_axes: list[_FakeAxis], signal_dimension: int = 2) -> None:
        self.signal_axes = signal_axes
        self.signal_dimension = signal_dimension


class _FakeSignal:
    def __init__(
        self,
        data: np.ndarray,
        signal_axes: list[_FakeAxis],
        *,
        metadata: dict | None = None,
        original_metadata: dict | None = None,
        signal_dimension: int = 2,
    ) -> None:
        self.data = data
        self.axes_manager = _FakeAxesManager(signal_axes, signal_dimension=signal_dimension)
        self.metadata = _FakeMetadata(metadata or {})
        self.original_metadata = _FakeMetadata(original_metadata or {})


def _install_fake_hyperspy(monkeypatch: pytest.MonkeyPatch, loaded: object) -> None:
    hyperspy_module = types.ModuleType("hyperspy")
    api_module = types.ModuleType("hyperspy.api")

    def _fake_load(path: str) -> object:
        return loaded

    api_module.load = _fake_load  # type: ignore[attr-defined]
    hyperspy_module.api = api_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hyperspy", hyperspy_module)
    monkeypatch.setitem(sys.modules, "hyperspy.api", api_module)


def _dm_metadata(scale: float, unit: str = "nm") -> dict:
    return {
        "ImageList": {
            "TagGroup0": {
                "ImageData": {
                    "Calibrations": {
                        "Dimension": {
                            "0": {"Scale": scale, "Units": unit},
                            "1": {"Scale": scale, "Units": unit},
                        }
                    }
                }
            }
        }
    }


def test_load_image_prefers_signal_axes_over_metadata_and_manual(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    signal = _FakeSignal(
        np.ones((8, 8), dtype=np.float32),
        [_FakeAxis(0.12, "nm"), _FakeAxis(0.12, "nm")],
        original_metadata=_dm_metadata(0.3, unit="A"),
    )
    _install_fake_hyperspy(monkeypatch, signal)

    session = load_image(
        tmp_path / "sample.dm4",
        manual_calibration={"size": 0.5, "unit": "A"},
    )

    assert session.pixel_calibration.size == pytest.approx(0.12)
    assert session.pixel_calibration.unit == "nm"
    assert session.pixel_calibration.source == "hyperspy_axes"


def test_load_image_uses_original_metadata_when_axes_are_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    signal = _FakeSignal(
        np.ones((8, 8), dtype=np.float32),
        [_FakeAxis(None, "nm"), _FakeAxis(None, "nm")],
        original_metadata=_dm_metadata(0.25, unit="nm"),
    )
    _install_fake_hyperspy(monkeypatch, signal)

    session = load_image(tmp_path / "sample.dm4")

    assert session.pixel_calibration.size == pytest.approx(0.25)
    assert session.pixel_calibration.unit == "nm"
    assert session.pixel_calibration.source == "dm_original_metadata_calibration"


def test_load_image_uses_manual_override_only_after_metadata_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    signal = _FakeSignal(
        np.ones((8, 8), dtype=np.float32),
        [_FakeAxis(None, None), _FakeAxis(None, None)],
        original_metadata={},
    )
    _install_fake_hyperspy(monkeypatch, signal)

    session = load_image(tmp_path / "sample.dm4", manual_calibration=0.42)

    assert session.pixel_calibration.size == pytest.approx(0.42)
    assert session.pixel_calibration.unit == "arb."
    assert session.pixel_calibration.source == "manual_override"


def test_load_image_marks_metadata_missing_when_no_calibration_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    signal = _FakeSignal(
        np.ones((8, 8), dtype=np.float32),
        [_FakeAxis(None, None), _FakeAxis(None, None)],
        original_metadata={},
    )
    _install_fake_hyperspy(monkeypatch, signal)

    session = load_image(tmp_path / "sample.dm4")

    assert session.pixel_calibration.size is None
    assert session.pixel_calibration.unit == "px"
    assert session.pixel_calibration.source == "metadata_missing"


def test_load_image_auto_selects_the_only_2d_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    one_dimensional = _FakeSignal(
        np.ones((8,), dtype=np.float32),
        [_FakeAxis(None, None)],
        signal_dimension=1,
    )
    image_signal = _FakeSignal(
        np.ones((8, 8), dtype=np.float32),
        [_FakeAxis(0.2, "nm"), _FakeAxis(0.2, "nm")],
    )
    _install_fake_hyperspy(monkeypatch, [one_dimensional, image_signal])

    session = load_image(tmp_path / "sample.dm4")

    assert session.dataset_index == 1
    assert session.raw_image.shape == (8, 8)


def test_load_image_rejects_dataset_index_out_of_range(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    signal = _FakeSignal(
        np.ones((8, 8), dtype=np.float32),
        [_FakeAxis(0.2, "nm"), _FakeAxis(0.2, "nm")],
    )
    _install_fake_hyperspy(monkeypatch, [signal])

    with pytest.raises(IndexError):
        load_image(tmp_path / "sample.dm4", dataset_index=3)


def test_load_image_requires_dataset_index_when_multiple_2d_candidates_exist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    signal_a = _FakeSignal(
        np.ones((8, 8), dtype=np.float32),
        [_FakeAxis(0.2, "nm"), _FakeAxis(0.2, "nm")],
    )
    signal_b = _FakeSignal(
        np.ones((8, 8), dtype=np.float32),
        [_FakeAxis(0.3, "nm"), _FakeAxis(0.3, "nm")],
    )
    _install_fake_hyperspy(monkeypatch, [signal_a, signal_b])

    with pytest.raises(ValueError, match="Please specify dataset_index"):
        load_image(tmp_path / "sample.dm4")
