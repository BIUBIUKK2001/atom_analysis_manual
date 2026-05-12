from __future__ import annotations

import math

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .session import AnalysisSession, AnnotationConfig


def _patch_fft_features(image: np.ndarray, x_local: float, y_local: float, radius: int) -> tuple[float, float]:
    x0 = max(int(round(x_local)) - radius, 0)
    x1 = min(int(round(x_local)) + radius + 1, image.shape[1])
    y0 = max(int(round(y_local)) - radius, 0)
    y1 = min(int(round(y_local)) + radius + 1, image.shape[0])
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return np.nan, np.nan
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(patch - np.mean(patch))))
    yy, xx = np.indices(fft_mag.shape)
    center = np.array([(fft_mag.shape[0] - 1) / 2.0, (fft_mag.shape[1] - 1) / 2.0])
    rr = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    rr[int(center[0]), int(center[1])] = 0.0
    peak_idx = np.unravel_index(np.argmax(fft_mag * (rr > 1.0)), fft_mag.shape)
    dy = float(peak_idx[0] - center[0])
    dx = float(peak_idx[1] - center[1])
    dominant_period = float(np.inf if dx == 0 and dy == 0 else patch.shape[0] / (np.hypot(dx, dy) + 1e-8))
    dominant_orientation = float(math.degrees(math.atan2(dy, dx)) % 180.0)
    return dominant_period, dominant_orientation


def suggest_annotations(session: AnalysisSession, config: AnnotationConfig) -> pd.DataFrame:
    if session.local_metrics.empty:
        raise ValueError("Local metrics are required before annotation suggestions.")

    features = session.local_metrics.copy()
    points = session.get_atom_table(preferred="curated")

    if config.include_fft:
        image = session.get_processed_image()
        origin_x, origin_y = session.get_processed_origin()
        fft_periods = []
        fft_orientations = []
        joined = points.merge(features[["atom_id"]], on="atom_id", how="right")
        for _, row in joined.iterrows():
            period, orientation = _patch_fft_features(
                image=image,
                x_local=float(row["x_px"] - origin_x),
                y_local=float(row["y_px"] - origin_y),
                radius=config.patch_radius,
            )
            fft_periods.append(period)
            fft_orientations.append(orientation)
        features["fft_periodicity_px"] = fft_periods
        features["fft_orientation_deg"] = fft_orientations

    use_columns = [column for column in config.feature_columns if column in features.columns]
    if config.include_fft:
        use_columns += [column for column in ("fft_periodicity_px", "fft_orientation_deg") if column in features.columns]

    feature_matrix = features[use_columns].replace([np.inf, -np.inf], np.nan).dropna()
    if feature_matrix.empty:
        raise ValueError("No valid feature rows are available for annotation suggestions.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix.to_numpy(dtype=float))
    model = KMeans(n_clusters=config.n_clusters, random_state=config.random_state, n_init="auto")
    labels = model.fit_predict(scaled)

    result = features[["atom_id"]].copy()
    result["suggested_cluster"] = -1
    result.loc[feature_matrix.index, "suggested_cluster"] = labels
    for column in use_columns:
        result[column] = features[column]
    return result


def _assign_polygon_labels(points: pd.DataFrame, polygons: list[list[list[float]]], labels: list[str]) -> pd.Series:
    label_series = pd.Series(index=points.index, dtype=object)
    xy = points[["x_px", "y_px"]].to_numpy(dtype=float)
    for polygon, label in zip(polygons, labels, strict=True):
        path = MplPath(np.asarray(polygon, dtype=float))
        label_series.loc[path.contains_points(xy)] = label
    return label_series


def save_annotations(
    session: AnalysisSession,
    polygons: list[list[list[float]]],
    labels: list[str],
    annotation_type: str = "domain",
) -> AnalysisSession:
    if len(polygons) != len(labels):
        raise ValueError("polygons and labels must have the same length.")

    points = session.get_atom_table(preferred="curated").copy()
    annotation_records = []
    for polygon, label in zip(polygons, labels, strict=True):
        annotation_records.append({"type": annotation_type, "label": label, "polygon": polygon})

    session.annotations = {"records": annotation_records, "annotation_type": annotation_type}

    if not points.empty:
        assigned = _assign_polygon_labels(points, polygons, labels)
        points["annotation_label"] = assigned
        if not session.curated_points.empty:
            session.curated_points = points
        elif not session.refined_points.empty:
            session.refined_points = points
        if not session.local_metrics.empty:
            session.local_metrics = session.local_metrics.drop(columns=["annotation_label"], errors="ignore").merge(
                points[["atom_id", "annotation_label"]],
                on="atom_id",
                how="left",
            )

    if annotation_records:
        session.set_stage("annotated")
    session.record_step(
        "save_annotations",
        parameters={"annotation_type": annotation_type},
        notes={"annotation_count": len(annotation_records)},
    )
    return session
