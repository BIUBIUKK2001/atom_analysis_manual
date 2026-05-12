from __future__ import annotations

import types

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import em_atom_workbench.plotting as plotting_module
from em_atom_workbench.plotting import launch_detection_napari_viewer, plot_class_overlay
from em_atom_workbench.session import AnalysisSession


class _FakeLayer:
    def __init__(self, data, **kwargs) -> None:
        self.data = np.asarray(data)
        self.kwargs = dict(kwargs)
        self.visible = bool(kwargs.get("visible", True))
        self.editable = True
        self.translate = kwargs.get("translate")


class _FakeViewer:
    def __init__(self, title: str) -> None:
        self.title = title
        self.layers: dict[str, _FakeLayer] = {}

    def add_image(self, data, *, name: str, **kwargs):
        layer = _FakeLayer(data, name=name, **kwargs)
        self.layers[name] = layer
        return layer

    def add_points(self, data, *, name: str, **kwargs):
        layer = _FakeLayer(data, name=name, **kwargs)
        self.layers[name] = layer
        return layer


def _install_fake_napari(monkeypatch) -> None:
    fake_napari = types.SimpleNamespace(Viewer=_FakeViewer)
    monkeypatch.setattr(plotting_module, "_require_napari", lambda: fake_napari)


def test_launch_detection_napari_viewer_single_channel_uses_read_only_candidate_layer(monkeypatch) -> None:
    _install_fake_napari(monkeypatch)

    session = AnalysisSession(name="single_detection", raw_image=np.ones((8, 8), dtype=float))
    session.preprocess_result = {"processed_image": np.full((4, 4), 2.0), "origin_x": 2, "origin_y": 3}
    session.candidate_points = pd.DataFrame({"candidate_id": [0], "x_px": [4.0], "y_px": [5.0]})

    viewer = launch_detection_napari_viewer(session)

    assert list(viewer.layers) == ["raw_image", "processed_image", "candidate_points"]
    assert viewer.layers["raw_image"].visible is False
    assert viewer.layers["processed_image"].visible is True
    assert viewer.layers["processed_image"].translate == (3, 2)
    assert viewer.layers["candidate_points"].visible is True
    assert viewer.layers["candidate_points"].editable is False


def test_launch_detection_napari_viewer_multichannel_separates_heavy_and_light_layers(monkeypatch) -> None:
    _install_fake_napari(monkeypatch)

    session = AnalysisSession(
        name="multichannel_detection",
        raw_image=np.zeros((12, 12), dtype=float),
        primary_channel="idpc",
    )
    session.set_channel_state(
        "idpc",
        raw_image=np.full((12, 12), 1.0),
        preprocess_result={"processed_image": np.full((6, 6), 2.0), "origin_x": 1, "origin_y": 2},
        contrast_mode="bright_peak",
    )
    session.set_channel_state(
        "haadf",
        raw_image=np.full((12, 12), 3.0),
        preprocess_result={"processed_image": np.full((6, 6), 4.0), "origin_x": 0, "origin_y": 0},
        contrast_mode="bright_peak",
    )
    session.set_channel_state(
        "abf",
        raw_image=np.full((12, 12), 5.0),
        preprocess_result={"processed_image": np.full((6, 6), 6.0), "origin_x": 1, "origin_y": 2},
        contrast_mode="dark_dip",
    )
    session.set_workflow(
        "hfo2_multichannel",
        {
            "primary_channel": "idpc",
            "heavy_channel": "haadf",
            "light_channel": "idpc",
            "confirm_channel": "abf",
        },
    )
    session.candidate_points = pd.DataFrame(
        {
            "candidate_id": [0, 1, 2],
            "x_px": [4.0, 6.5, 8.5],
            "y_px": [4.0, 6.0, 8.0],
            "column_role": ["heavy_atom", "light_atom", "light_atom"],
        }
    )

    viewer = launch_detection_napari_viewer(session)

    assert list(viewer.layers) == [
        "raw_primary_image",
        "processed_idpc",
        "processed_haadf",
        "processed_abf",
        "heavy_atom_points",
        "light_atom_points",
    ]
    assert viewer.layers["raw_primary_image"].visible is False
    assert viewer.layers["processed_idpc"].visible is True
    assert viewer.layers["processed_haadf"].visible is False
    assert viewer.layers["processed_abf"].visible is False
    assert viewer.layers["heavy_atom_points"].editable is False
    assert viewer.layers["light_atom_points"].editable is False
    assert viewer.layers["heavy_atom_points"].data.shape == (1, 2)
    assert viewer.layers["light_atom_points"].data.shape == (2, 2)
    assert "candidate_points" not in viewer.layers


def test_plot_class_overlay_places_legend_outside_axes() -> None:
    image = np.zeros((10, 10), dtype=float)
    points = pd.DataFrame(
        {
            "x_px": [2.0, 6.0],
            "y_px": [3.0, 7.0],
            "class_id": [0, 1],
            "class_name": ["class_0", "class_1"],
            "class_color": ["#00a5cf", "#f18f01"],
        }
    )

    fig, ax = plot_class_overlay(image, points)
    legend = ax.get_legend()

    assert legend is not None
    assert legend.get_title().get_text() == "class_name"
    assert legend.get_bbox_to_anchor()._bbox.x0 > 1.0
    plt.close(fig)


def test_plot_atom_overlay_uses_nm_axis_labels_when_calibrated() -> None:
    image = np.zeros((100, 100), dtype=float)
    points = pd.DataFrame({"x_px": [5.0], "y_px": [5.0]})

    fig, ax = plotting_module.plot_atom_overlay(
        image,
        points,
        pixel_size=0.05,
        unit="nm",
        target_unit="nm",
    )

    assert ax.get_xlabel() == "x (nm)"
    assert ax.get_ylabel() == "y (nm)"
    assert ax.xaxis.get_major_formatter()(0, 0) == "0"
    assert ax.yaxis.get_major_formatter()(0, 0) == "0"
    assert ax.xaxis.get_major_formatter()(20, 0) == "1"

    x_limits = ax.get_xlim()
    x_ticks = ax.xaxis.get_major_locator().tick_values(*x_limits)
    visible_x_labels = [
        ax.xaxis.get_major_formatter()(tick, index)
        for index, tick in enumerate(x_ticks)
        if min(x_limits) <= tick <= max(x_limits)
    ]
    assert "0" in visible_x_labels
    assert visible_x_labels
    assert all(label == str(int(label)) for label in visible_x_labels)

    y_limits = ax.get_ylim()
    y_ticks = ax.yaxis.get_major_locator().tick_values(*y_limits)
    visible_y_labels = [
        ax.yaxis.get_major_formatter()(tick, index)
        for index, tick in enumerate(y_ticks)
        if min(y_limits) <= tick <= max(y_limits)
    ]
    assert "0" in visible_y_labels
    assert visible_y_labels
    assert all(label == str(int(label)) for label in visible_y_labels)
    plt.close(fig)
