from __future__ import annotations

from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .simple_quant import AnalysisROI, BasisVectorSpec, DirectionSpec


def _resolve_image(session: Any, image_channel: str | None, image_key: str) -> tuple[np.ndarray, tuple[float, float], str]:
    channel_name = image_channel or session.primary_channel
    key = str(image_key).lower()
    if key == "processed":
        return session.get_processed_image(channel_name), session.get_processed_origin(channel_name), channel_name
    if key == "raw":
        image = session.get_channel_state(channel_name).raw_image
        if image is None:
            raise ValueError(f"Raw image is not available for channel {channel_name!r}.")
        return image, (0.0, 0.0), channel_name
    raise ValueError("image_key must be 'raw' or 'processed'.")


def _points_to_layer_data(points: pd.DataFrame, origin_xy: tuple[float, float]) -> np.ndarray:
    origin_x, origin_y = origin_xy
    if points.empty:
        return np.empty((0, 2), dtype=float)
    return np.column_stack(
        (
            points["y_px"].to_numpy(dtype=float) - float(origin_y),
            points["x_px"].to_numpy(dtype=float) - float(origin_x),
        )
    )


def _features(points: pd.DataFrame) -> pd.DataFrame:
    columns = ["point_id", "atom_id", "class_id", "class_name", "point_set", "roi_id", "x_px", "y_px"]
    result = pd.DataFrame(index=points.index)
    for column in columns:
        result[column] = points[column] if column in points.columns else pd.NA
    return result


def _class_label(row: pd.Series) -> str:
    name = row.get("class_name", pd.NA)
    if pd.notna(name) and str(name):
        return str(name)
    class_id = row.get("class_id", pd.NA)
    return f"class_{int(class_id)}" if pd.notna(class_id) else "unclassified"


def _point_colors(points: pd.DataFrame, fallback_palette: str = "tab10") -> list[str]:
    if points.empty:
        return []
    labels = [_class_label(row) for _, row in points.iterrows()]
    palette = plt.get_cmap(fallback_palette)
    color_map: dict[str, str] = {}
    for label, (_, row) in zip(labels, points.iterrows(), strict=False):
        color = row.get("class_color", pd.NA)
        if isinstance(color, str) and color and mcolors.is_color_like(color):
            color_map.setdefault(label, color)
    for index, label in enumerate(sorted(set(labels))):
        color_map.setdefault(label, "#8a8a8a" if label == "unclassified" else mcolors.to_hex(palette(index % palette.N)))
    return [color_map[label] for label in labels]


def _snap_xy(points: pd.DataFrame, xy: tuple[float, float]) -> tuple[tuple[float, float], int, str]:
    tree = cKDTree(points[["x_px", "y_px"]].to_numpy(dtype=float))
    _, index = tree.query(np.asarray(xy, dtype=float), k=1)
    row = points.iloc[int(index)]
    atom_id = int(row["atom_id"]) if "atom_id" in row.index and pd.notna(row["atom_id"]) else -1
    point_id = str(row["point_id"]) if "point_id" in row.index and pd.notna(row["point_id"]) else f"atom:{atom_id}"
    return (float(row["x_px"]), float(row["y_px"])), atom_id, point_id


def pick_direction_vectors_with_napari(
    session: Any,
    quant_points: pd.DataFrame,
    direction_names: tuple[str, ...] = ("u", "v"),
    image_channel: str | None = None,
    image_key: str = "raw",
    snap_to_nearest_atoms: bool = True,
    point_size: float = 5.0,
) -> list[DirectionSpec]:
    try:
        import napari
    except ImportError as exc:
        raise ImportError("napari is required for interactive direction picking.") from exc

    image, origin_xy, channel_name = _resolve_image(session, image_channel, image_key)
    origin_x, origin_y = origin_xy
    viewer = napari.Viewer(title=f"Simple quant direction picker - {channel_name}")
    viewer.add_image(image, name=f"{image_key}_{channel_name}", colormap="gray")
    points_layer = viewer.add_points(
        _points_to_layer_data(quant_points, origin_xy),
        name="quant_points",
        features=_features(quant_points),
        size=point_size,
        face_color=_point_colors(quant_points),
        border_color="black",
        border_width=0.15,
        opacity=0.85,
    )
    points_layer.editable = False
    for name in direction_names:
        layer = viewer.add_points(
            np.empty((0, 2), dtype=float),
            name=f"direction_{name}",
            size=point_size * 1.6,
            face_color="#f18f01",
            border_color="white",
            border_width=0.25,
            opacity=0.95,
            symbol="cross",
        )
        layer.editable = True
        layer.metadata["direction_name"] = str(name)

    if hasattr(viewer, "show"):
        viewer.show(block=True)

    specs: list[DirectionSpec] = []
    for name in direction_names:
        layer = viewer.layers[f"direction_{name}"]
        data = np.asarray(layer.data, dtype=float)
        if data.shape != (2, 2):
            raise ValueError(f"Direction {name!r} requires exactly two picked points; got shape {data.shape}.")
        xy_1 = (float(data[0, 1] + origin_x), float(data[0, 0] + origin_y))
        xy_2 = (float(data[1, 1] + origin_x), float(data[1, 0] + origin_y))
        if snap_to_nearest_atoms:
            xy_1, _atom_1, _point_1 = _snap_xy(quant_points, xy_1)
            xy_2, _atom_2, _point_2 = _snap_xy(quant_points, xy_2)
        specs.append(
            DirectionSpec(
                name=str(name),
                from_xy_px=(xy_1, xy_2),
                snap_to_nearest_atoms=bool(snap_to_nearest_atoms),
            )
        )
    return specs


def pick_rois_with_napari(
    session: Any,
    points: pd.DataFrame,
    image_channel: str | None = None,
    image_key: str = "raw",
    default_roi_prefix: str = "ROI",
) -> list[AnalysisROI]:
    try:
        import napari
    except ImportError as exc:
        raise ImportError("napari is required for interactive ROI picking.") from exc

    image, origin_xy, channel_name = _resolve_image(session, image_channel, image_key)
    origin_x, origin_y = origin_xy
    viewer = napari.Viewer(title=f"Simple quant ROI picker - {channel_name}")
    viewer.add_image(image, name=f"{image_key}_{channel_name}", colormap="gray")
    points_layer = viewer.add_points(
        _points_to_layer_data(points, origin_xy),
        name="analysis_points",
        features=_features(points),
        size=5.0,
        face_color=_point_colors(points),
        border_color="black",
        border_width=0.15,
        opacity=0.85,
    )
    points_layer.editable = False
    shapes = viewer.add_shapes(
        name="analysis_rois",
        shape_type="polygon",
        edge_color="#ff9f1c",
        face_color=[1.0, 0.62, 0.11, 0.12],
        edge_width=2,
    )
    shapes.editable = True

    if hasattr(viewer, "show"):
        viewer.show(block=True)

    palette = plt.get_cmap("tab10")
    rois: list[AnalysisROI] = []
    for index, shape in enumerate(shapes.data):
        data = np.asarray(shape, dtype=float)
        if data.ndim != 2 or data.shape[0] < 3:
            continue
        polygon_xy = tuple((float(row[1] + origin_x), float(row[0] + origin_y)) for row in data)
        roi_id = f"roi_{index}"
        rois.append(
            AnalysisROI(
                roi_id=roi_id,
                roi_name=f"{default_roi_prefix}_{index}",
                polygon_xy_px=polygon_xy,
                color=mcolors.to_hex(palette(index % palette.N)),
            )
        )
    if not rois:
        return [AnalysisROI(roi_id="global", roi_name="global", polygon_xy_px=None, color="#ff9f1c")]
    return rois


def pick_basis_vectors_with_napari(
    session: Any,
    points: pd.DataFrame,
    basis_names: tuple[str, ...] = ("a", "b"),
    image_channel: str | None = None,
    image_key: str = "raw",
    snap_to_nearest_points: bool = True,
    point_size: float = 5.0,
) -> list[BasisVectorSpec]:
    try:
        import napari
    except ImportError as exc:
        raise ImportError("napari is required for interactive basis-vector picking.") from exc

    image, origin_xy, channel_name = _resolve_image(session, image_channel, image_key)
    origin_x, origin_y = origin_xy
    viewer = napari.Viewer(title=f"Simple quant basis picker - {channel_name}")
    viewer.add_image(image, name=f"{image_key}_{channel_name}", colormap="gray")
    points_layer = viewer.add_points(
        _points_to_layer_data(points, origin_xy),
        name="analysis_points",
        features=_features(points),
        size=point_size,
        face_color=_point_colors(points),
        border_color="black",
        border_width=0.15,
        opacity=0.85,
    )
    points_layer.editable = False
    palette = plt.get_cmap("tab10")
    for index, name in enumerate(basis_names):
        layer = viewer.add_points(
            np.empty((0, 2), dtype=float),
            name=f"basis_{name}",
            size=point_size * 1.7,
            face_color=mcolors.to_hex(palette(index % palette.N)),
            border_color="white",
            border_width=0.25,
            opacity=0.95,
            symbol="cross",
        )
        layer.editable = True
        layer.metadata["basis_name"] = str(name)

    if hasattr(viewer, "show"):
        viewer.show(block=True)

    specs: list[BasisVectorSpec] = []
    for name in basis_names:
        layer = viewer.layers[f"basis_{name}"]
        data = np.asarray(layer.data, dtype=float)
        if data.shape != (2, 2):
            raise ValueError(f"Basis vector {name!r} requires exactly two picked points; got shape {data.shape}.")
        xy_1 = (float(data[0, 1] + origin_x), float(data[0, 0] + origin_y))
        xy_2 = (float(data[1, 1] + origin_x), float(data[1, 0] + origin_y))
        if snap_to_nearest_points:
            xy_1, _atom_1, _point_1 = _snap_xy(points, xy_1)
            xy_2, _atom_2, _point_2 = _snap_xy(points, xy_2)
        specs.append(
            BasisVectorSpec(
                name=str(name),
                from_xy_px=(xy_1, xy_2),
                snap_to_nearest_points=False,
                use_length_as_period=True,
            )
        )
    return specs
