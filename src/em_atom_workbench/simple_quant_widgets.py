from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .simple_quant import DirectionSpec


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
    columns = ["atom_id", "class_id", "class_name", "x_px", "y_px"]
    result = pd.DataFrame(index=points.index)
    for column in columns:
        result[column] = points[column] if column in points.columns else pd.NA
    return result


def _snap_xy(points: pd.DataFrame, xy: tuple[float, float]) -> tuple[tuple[float, float], int]:
    tree = cKDTree(points[["x_px", "y_px"]].to_numpy(dtype=float))
    _, index = tree.query(np.asarray(xy, dtype=float), k=1)
    row = points.iloc[int(index)]
    return (float(row["x_px"]), float(row["y_px"])), int(row["atom_id"])


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
        face_color="#00a5cf",
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
            xy_1, _atom_1 = _snap_xy(quant_points, xy_1)
            xy_2, _atom_2 = _snap_xy(quant_points, xy_2)
        specs.append(
            DirectionSpec(
                name=str(name),
                from_xy_px=(xy_1, xy_2),
                snap_to_nearest_atoms=bool(snap_to_nearest_atoms),
            )
        )
    return specs
