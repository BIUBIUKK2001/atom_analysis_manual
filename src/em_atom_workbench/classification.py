from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .session import AnalysisSession
from .utils import border_values, extract_patch


CLASS_COLUMNS = [
    "class_id",
    "class_name",
    "class_color",
    "class_confidence",
    "class_source",
    "class_reviewed",
]

DEFAULT_CLASS_COLORS = (
    "#00a5cf",
    "#f18f01",
    "#7a5195",
    "#2ca25f",
    "#d95f02",
    "#4c78a8",
    "#e45756",
    "#72b7b2",
)


@dataclass
class AtomColumnClassificationConfig:
    feature_channels: tuple[str, ...] | None = None
    feature_patch_radii: int | dict[str, int] = 6
    features_enabled: tuple[str, ...] = (
        "center_intensity",
        "prominence",
        "local_snr",
        "integrated_intensity",
        "mean",
        "std",
        "quantiles",
    )
    background_mode: str = "border_median"
    feature_channel_weights: dict[str, float] = field(default_factory=dict)
    quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    radial_profile_bins: int = 3
    source_table: str = "refined"
    n_classes: int = 2
    auto_suggest_n_classes: bool = False
    n_classes_range: tuple[int, int] = (2, 5)
    cluster_method: str = "gaussian_mixture"
    feature_scaling: str = "robust"
    outlier_mode: str = "confidence"
    min_class_size: int = 1
    random_state: int = 0
    max_iter: int = 300
    n_init: int = 10
    confidence_threshold: float = 0.0
    dbscan_eps: float = 0.8
    dbscan_min_samples: int = 5
    class_name_prefix: str = "class"
    unclassified_name: str = "unclassified"
    class_name_map: dict[int | str, str] = field(default_factory=dict)
    class_color_map: dict[int | str, str] = field(default_factory=dict)


def _source_points(session: AnalysisSession, source_table: str) -> tuple[str, pd.DataFrame]:
    source = str(source_table).lower()
    if source == "curated":
        table = session.curated_points
    elif source == "candidate":
        table = session.candidate_points
    elif source == "refined":
        table = session.refined_points if not session.refined_points.empty else session.candidate_points
    else:
        raise ValueError("source_table must be 'candidate', 'refined', or 'curated'.")
    if table.empty:
        raise ValueError("No atom-column points are available for classification.")
    table = table.copy().reset_index(drop=True)
    if "atom_id" not in table.columns:
        table.insert(0, "atom_id", np.arange(len(table), dtype=int))
    if "candidate_id" not in table.columns:
        table.insert(1, "candidate_id", table["atom_id"].to_numpy(dtype=int))
    return source, table


def _feature_channels(session: AnalysisSession, config: AtomColumnClassificationConfig) -> list[str]:
    channels = list(config.feature_channels) if config.feature_channels is not None else session.list_channels()
    missing = [channel for channel in channels if channel not in session.list_channels()]
    if missing:
        raise KeyError(f"Unknown feature channels: {missing}")
    if not channels:
        raise ValueError("feature_channels must contain at least one channel.")
    return [str(channel) for channel in channels]


def _patch_radius(config: AtomColumnClassificationConfig, channel_name: str) -> int:
    radii = config.feature_patch_radii
    if isinstance(radii, dict):
        return max(int(radii.get(channel_name, radii.get("default", 6))), 1)
    return max(int(radii), 1)


def _oriented_image(session: AnalysisSession, channel_name: str) -> tuple[np.ndarray, tuple[int, int]]:
    image = np.asarray(session.get_processed_image(channel_name), dtype=float)
    mode = session.get_channel_state(channel_name).contrast_mode
    if mode == "dark_dip":
        image = -image
    return image, session.get_processed_origin(channel_name)


def _center_value(image: np.ndarray, xy_global: tuple[float, float], origin_xy: tuple[int, int]) -> float:
    x_local = int(round(float(xy_global[0]) - origin_xy[0]))
    y_local = int(round(float(xy_global[1]) - origin_xy[1]))
    if x_local < 0 or y_local < 0 or y_local >= image.shape[0] or x_local >= image.shape[1]:
        return np.nan
    return float(image[y_local, x_local])


def _background_value(patch: np.ndarray, mode: str) -> tuple[float, float]:
    if patch.size == 0:
        return np.nan, np.nan
    mode_key = str(mode).lower()
    border = border_values(patch)
    if mode_key in {"border_median", "edge_median", "ring_median"}:
        values = border if border.size else patch.ravel()
        background = float(np.median(values))
        noise = float(np.std(values) + 1e-8)
        return background, noise
    if mode_key == "local_quantile":
        values = patch.ravel()
        background = float(np.quantile(values, 0.20))
        noise = float(np.std(values) + 1e-8)
        return background, noise
    raise ValueError("background_mode must be 'border_median', 'ring_median', or 'local_quantile'.")


def _radial_profile_features(patch: np.ndarray, bins: int) -> dict[str, float]:
    if patch.size == 0:
        return {f"radial_profile_{idx}": np.nan for idx in range(int(bins))}
    yy, xx = np.indices(patch.shape, dtype=float)
    cx = 0.5 * (patch.shape[1] - 1)
    cy = 0.5 * (patch.shape[0] - 1)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    edges = np.linspace(0.0, float(np.max(rr)) + 1e-8, int(bins) + 1)
    features: dict[str, float] = {}
    for idx in range(int(bins)):
        mask = (rr >= edges[idx]) & (rr <= edges[idx + 1])
        features[f"radial_profile_{idx}"] = float(np.mean(patch[mask])) if np.any(mask) else np.nan
    return features


def _channel_features(
    patch: np.ndarray,
    center: float,
    background: float,
    noise: float,
    config: AtomColumnClassificationConfig,
) -> dict[str, float]:
    enabled = set(config.features_enabled)
    result: dict[str, float] = {}
    prominence = float(center - background) if np.isfinite(center) and np.isfinite(background) else np.nan
    if "center_intensity" in enabled:
        result["center_intensity"] = center
    if "prominence" in enabled:
        result["prominence"] = prominence
    if "local_snr" in enabled:
        result["local_snr"] = prominence / noise if np.isfinite(prominence) and noise > 0 else np.nan
    if "integrated_intensity" in enabled:
        result["integrated_intensity"] = float(np.sum(patch - background)) if patch.size else np.nan
    if "mean" in enabled:
        result["mean"] = float(np.mean(patch)) if patch.size else np.nan
    if "std" in enabled:
        result["std"] = float(np.std(patch)) if patch.size else np.nan
    if "quantiles" in enabled:
        values = patch.ravel() if patch.size else np.array([], dtype=float)
        for quantile in config.quantiles:
            key = f"q{int(round(float(quantile) * 100)):02d}"
            result[key] = float(np.quantile(values, quantile)) if values.size else np.nan
    if "radial_profile" in enabled:
        result.update(_radial_profile_features(patch, config.radial_profile_bins))
    return result


def extract_atom_column_features(
    session: AnalysisSession,
    config: AtomColumnClassificationConfig,
) -> pd.DataFrame:
    _, points = _source_points(session, config.source_table)
    channels = _feature_channels(session, config)
    records: list[dict[str, Any]] = []
    for _, row in points.iterrows():
        xy_global = (float(row["x_px"]), float(row["y_px"]))
        record: dict[str, Any] = {
            "atom_id": int(row["atom_id"]),
            "candidate_id": int(row["candidate_id"]),
            "x_px": xy_global[0],
            "y_px": xy_global[1],
        }
        if "fit_amplitude" in config.features_enabled and "amplitude" in row.index:
            record["fit_amplitude"] = float(row["amplitude"])
        if "fit_sigma" in config.features_enabled and {"sigma_x", "sigma_y"}.issubset(row.index):
            record["fit_sigma"] = float(0.5 * (float(row["sigma_x"]) + float(row["sigma_y"])))
        for channel_name in channels:
            image, origin_xy = _oriented_image(session, channel_name)
            patch, _ = extract_patch(
                image=image,
                center_xy_global=xy_global,
                half_window=_patch_radius(config, channel_name),
                origin_xy=origin_xy,
            )
            center = _center_value(image, xy_global, origin_xy)
            background, noise = _background_value(patch, config.background_mode)
            for key, value in _channel_features(patch, center, background, noise, config).items():
                record[f"{channel_name}__{key}"] = value
        records.append(record)
    features = pd.DataFrame(records)
    session.classification_features = features
    session.record_step(
        "extract_atom_column_features",
        parameters=config,
        notes={"point_count": len(features), "feature_channels": channels},
    )
    return features


def _numeric_feature_columns(features: pd.DataFrame) -> list[str]:
    excluded = {"atom_id", "candidate_id", "x_px", "y_px"}
    columns: list[str] = []
    for column in features.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(features[column]):
            columns.append(str(column))
    if not columns:
        raise ValueError("No numeric classification features were extracted.")
    return columns


def _impute_feature_matrix(features: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    matrix = features[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).copy()
    for col_idx in range(matrix.shape[1]):
        values = matrix[:, col_idx]
        finite = np.isfinite(values)
        fill_value = float(np.median(values[finite])) if finite.any() else 0.0
        values[~finite] = fill_value
        matrix[:, col_idx] = values
    return matrix


def _scale_feature_matrix(matrix: np.ndarray, scaling: str) -> np.ndarray:
    scaling_key = str(scaling).lower()
    if scaling_key == "none":
        return matrix.astype(float, copy=True)
    if scaling_key == "robust":
        return RobustScaler().fit_transform(matrix)
    if scaling_key == "standard":
        return StandardScaler().fit_transform(matrix)
    if scaling_key == "minmax":
        return MinMaxScaler().fit_transform(matrix)
    raise ValueError("feature_scaling must be 'robust', 'standard', 'minmax', or 'none'.")


def _apply_channel_weights(
    matrix: np.ndarray,
    feature_columns: list[str],
    weights: dict[str, float],
) -> np.ndarray:
    weighted = matrix.astype(float, copy=True)
    for idx, column in enumerate(feature_columns):
        channel_name = column.split("__", 1)[0] if "__" in column else ""
        weighted[:, idx] *= float(weights.get(channel_name, 1.0))
    return weighted


def _relabel_by_first_feature(labels: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    positive_labels = sorted(label for label in set(labels) if label >= 0)
    ordering = sorted(positive_labels, key=lambda label: float(np.mean(matrix[labels == label, 0])))
    mapping = {old: new for new, old in enumerate(ordering)}
    return np.asarray([mapping.get(int(label), -1) for label in labels], dtype=int)


def _centroid_confidence(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    confidences = np.zeros(len(labels), dtype=float)
    positive_labels = sorted(label for label in set(labels) if label >= 0)
    if not positive_labels:
        return confidences
    centroids = np.vstack([np.mean(matrix[labels == label], axis=0) for label in positive_labels])
    distances = np.linalg.norm(matrix[:, None, :] - centroids[None, :, :], axis=2)
    nearest = np.min(distances, axis=1)
    if len(positive_labels) == 1:
        finite = np.isfinite(nearest)
        scale = float(np.percentile(nearest[finite], 90)) if finite.any() else 1.0
        return np.clip(1.0 - nearest / (scale + 1e-8), 0.0, 1.0)
    sorted_distances = np.sort(distances, axis=1)
    margin = 1.0 - sorted_distances[:, 0] / (sorted_distances[:, 1] + 1e-8)
    confidences[:] = np.clip(margin, 0.0, 1.0)
    return confidences


def _cluster_fixed_n(
    matrix: np.ndarray,
    config: AtomColumnClassificationConfig,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    method = str(config.cluster_method).lower()
    if n_classes < 1:
        raise ValueError("n_classes must be at least 1.")
    if n_classes > len(matrix):
        raise ValueError("n_classes cannot exceed the number of atom columns.")
    if method == "gaussian_mixture":
        model = GaussianMixture(
            n_components=n_classes,
            random_state=config.random_state,
            max_iter=config.max_iter,
            n_init=config.n_init,
        )
        labels = model.fit_predict(matrix)
        probabilities = model.predict_proba(matrix)
        return labels.astype(int), np.max(probabilities, axis=1)
    if method == "kmeans":
        model = KMeans(
            n_clusters=n_classes,
            random_state=config.random_state,
            max_iter=config.max_iter,
            n_init=config.n_init,
        )
        labels = model.fit_predict(matrix)
        return labels.astype(int), _centroid_confidence(matrix, labels)
    if method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_classes)
        labels = model.fit_predict(matrix)
        return labels.astype(int), _centroid_confidence(matrix, labels)
    raise ValueError("cluster_method must be 'gaussian_mixture', 'kmeans', 'agglomerative', or 'dbscan'.")


def _cluster_dbscan(
    matrix: np.ndarray,
    config: AtomColumnClassificationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    model = DBSCAN(eps=float(config.dbscan_eps), min_samples=int(config.dbscan_min_samples))
    labels = model.fit_predict(matrix).astype(int)
    confidences = np.where(labels >= 0, 1.0, 0.0).astype(float)
    return labels, confidences


def _suggest_n_classes(matrix: np.ndarray, config: AtomColumnClassificationConfig) -> int:
    start, stop = config.n_classes_range
    best_n = int(config.n_classes)
    best_score = -np.inf
    for n_classes in range(max(2, int(start)), min(int(stop), len(matrix) - 1) + 1):
        try:
            labels, _ = _cluster_fixed_n(matrix, config, n_classes)
            if len(set(labels)) < 2:
                continue
            score = float(silhouette_score(matrix, labels))
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_n = n_classes
    return best_n


def _apply_outlier_policy(
    labels: np.ndarray,
    confidences: np.ndarray,
    config: AtomColumnClassificationConfig,
) -> np.ndarray:
    mode = str(config.outlier_mode).lower()
    result = labels.astype(int, copy=True)
    if mode in {"confidence", "confidence_or_small_class"} and config.confidence_threshold > 0:
        result[confidences < float(config.confidence_threshold)] = -1
    if mode in {"small_class", "confidence_or_small_class"} and config.min_class_size > 1:
        counts = pd.Series(result[result >= 0]).value_counts()
        small_labels = set(int(label) for label, count in counts.items() if int(count) < int(config.min_class_size))
        result[[label in small_labels for label in result]] = -1
    if mode == "none":
        return labels.astype(int, copy=True)
    if mode not in {"confidence", "small_class", "confidence_or_small_class", "none"}:
        raise ValueError("outlier_mode must be 'none', 'confidence', 'small_class', or 'confidence_or_small_class'.")
    return result


def _class_name(class_id: int, config: AtomColumnClassificationConfig) -> str:
    if class_id < 0:
        return config.unclassified_name
    for key in (class_id, str(class_id), f"{config.class_name_prefix}_{class_id}"):
        if key in config.class_name_map:
            return str(config.class_name_map[key])
    return f"{config.class_name_prefix}_{class_id}"


def _class_color(class_id: int, config: AtomColumnClassificationConfig) -> str:
    if class_id < 0:
        return "#737373"
    for key in (class_id, str(class_id), f"{config.class_name_prefix}_{class_id}"):
        if key in config.class_color_map:
            return str(config.class_color_map[key])
    return DEFAULT_CLASS_COLORS[class_id % len(DEFAULT_CLASS_COLORS)]


def _classification_records(
    features: pd.DataFrame,
    labels: np.ndarray,
    confidences: np.ndarray,
    config: AtomColumnClassificationConfig,
) -> pd.DataFrame:
    assigned = features[["atom_id", "candidate_id", "x_px", "y_px"]].copy()
    assigned["class_id"] = labels.astype(int)
    assigned["class_name"] = [_class_name(int(label), config) for label in labels]
    assigned["class_color"] = [_class_color(int(label), config) for label in labels]
    assigned["class_confidence"] = confidences.astype(float)
    assigned["class_source"] = str(config.cluster_method)
    assigned["class_reviewed"] = False
    return assigned


def _merge_class_columns(table: pd.DataFrame, assigned: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    result = table.copy()
    key = "atom_id" if "atom_id" in result.columns and "atom_id" in assigned.columns else "candidate_id"
    if key not in result.columns:
        return result
    class_data = assigned[[key] + CLASS_COLUMNS].drop_duplicates(subset=[key])
    result = result.drop(columns=[column for column in CLASS_COLUMNS if column in result.columns])
    return result.merge(class_data, on=key, how="left")


def _write_classification_to_session(session: AnalysisSession, assigned: pd.DataFrame) -> None:
    if not session.candidate_points.empty:
        session.candidate_points = _merge_class_columns(session.candidate_points, assigned)
    if not session.refined_points.empty:
        session.refined_points = _merge_class_columns(session.refined_points, assigned)
    if not session.curated_points.empty:
        session.curated_points = _merge_class_columns(session.curated_points, assigned)


def _classification_summary(
    assigned: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
    config: AtomColumnClassificationConfig,
    chosen_n_classes: int,
) -> dict[str, Any]:
    class_counts = (
        assigned.groupby(["class_id", "class_name"], dropna=False)
        .size()
        .reset_index(name="count")
        .to_dict(orient="records")
    )
    joined = assigned[["atom_id", "class_id", "class_name", "class_confidence"]].merge(
        features[["atom_id"] + feature_columns],
        on="atom_id",
        how="left",
    )
    means = (
        joined.groupby(["class_id", "class_name"], dropna=False)[feature_columns + ["class_confidence"]]
        .mean(numeric_only=True)
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "chosen_n_classes": int(chosen_n_classes),
        "cluster_method": config.cluster_method,
        "feature_scaling": config.feature_scaling,
        "feature_columns": feature_columns,
        "class_counts": class_counts,
        "class_feature_means": means,
        "class_name_map": dict(config.class_name_map),
        "class_color_map": dict(config.class_color_map),
    }


def classify_atom_columns(
    session: AnalysisSession,
    config: AtomColumnClassificationConfig,
) -> AnalysisSession:
    features = extract_atom_column_features(session, config)
    feature_columns = _numeric_feature_columns(features)
    matrix = _impute_feature_matrix(features, feature_columns)
    matrix = _scale_feature_matrix(matrix, config.feature_scaling)
    matrix = _apply_channel_weights(matrix, feature_columns, config.feature_channel_weights)

    if str(config.cluster_method).lower() == "dbscan":
        labels, confidences = _cluster_dbscan(matrix, config)
        chosen_n_classes = len(set(labels) - {-1})
    else:
        chosen_n_classes = (
            _suggest_n_classes(matrix, config)
            if config.auto_suggest_n_classes
            else int(config.n_classes)
        )
        labels, confidences = _cluster_fixed_n(matrix, config, chosen_n_classes)
    labels = _relabel_by_first_feature(labels, matrix)
    labels = _apply_outlier_policy(labels, confidences, config)
    labels = _relabel_by_first_feature(labels, matrix)

    assigned = _classification_records(features, labels, confidences, config)
    _write_classification_to_session(session, assigned)
    session.classification_features = features
    session.classification_summary = _classification_summary(
        assigned,
        features,
        feature_columns,
        config,
        chosen_n_classes,
    )
    session.clear_downstream_results("classified")
    session.set_stage("classified")
    session.record_step(
        "classify_atom_columns",
        parameters=config,
        notes={
            "point_count": len(assigned),
            "chosen_n_classes": int(chosen_n_classes),
            "class_count": int(len(set(labels) - {-1})),
        },
    )
    return session


def apply_class_name_map(
    session: AnalysisSession,
    class_name_map: dict[int | str, str] | None = None,
    class_color_map: dict[int | str, str] | None = None,
    *,
    mark_reviewed: bool = True,
) -> AnalysisSession:
    class_name_map = dict(class_name_map or {})
    class_color_map = dict(class_color_map or {})

    def update_table(table: pd.DataFrame) -> pd.DataFrame:
        if table.empty or "class_id" not in table.columns:
            return table
        updated = table.copy()
        for idx, row in updated.iterrows():
            class_id = int(row["class_id"]) if pd.notna(row["class_id"]) else -1
            if class_id < 0:
                continue
            for key in (class_id, str(class_id), f"class_{class_id}", row.get("class_name")):
                if key in class_name_map:
                    updated.at[idx, "class_name"] = str(class_name_map[key])
                    if mark_reviewed:
                        updated.at[idx, "class_reviewed"] = True
                if key in class_color_map:
                    updated.at[idx, "class_color"] = str(class_color_map[key])
                    if mark_reviewed:
                        updated.at[idx, "class_reviewed"] = True
        return updated

    session.candidate_points = update_table(session.candidate_points)
    session.refined_points = update_table(session.refined_points)
    session.curated_points = update_table(session.curated_points)
    summary = dict(getattr(session, "classification_summary", {}) or {})
    summary["class_name_map"] = {**dict(summary.get("class_name_map", {})), **class_name_map}
    summary["class_color_map"] = {**dict(summary.get("class_color_map", {})), **class_color_map}
    session.classification_summary = summary
    session.record_step(
        "apply_class_name_map",
        parameters={
            "class_name_map": class_name_map,
            "class_color_map": class_color_map,
            "mark_reviewed": bool(mark_reviewed),
        },
    )
    return session


def classification_summary_table(session: AnalysisSession) -> pd.DataFrame:
    table = session.get_atom_table(preferred="curated")
    if table.empty or "class_id" not in table.columns:
        return pd.DataFrame(columns=["class_id", "class_name", "count", "mean_confidence"])
    grouped = (
        table.groupby(["class_id", "class_name"], dropna=False)
        .agg(count=("class_id", "size"), mean_confidence=("class_confidence", "mean"))
        .reset_index()
    )
    return grouped.sort_values(["class_id", "class_name"], na_position="last").reset_index(drop=True)


def _set_points_layer_add_defaults(layer: Any, *, point_size: float, color: str) -> None:
    """Keep newly added napari class-review points styled as circular points."""
    for attr, value in (
        ("current_size", float(point_size)),
        ("current_face_color", color),
        ("current_border_color", color),
        ("current_border_width", 0.0),
        ("current_symbol", "disc"),
    ):
        try:
            setattr(layer, attr, value)
        except Exception:
            continue


def launch_class_review_napari(
    session: AnalysisSession,
    *,
    image_channel: str | None = None,
    point_size: float = 5.0,
    source_table: str | None = None,
) -> Any:
    try:
        import napari
    except ImportError as exc:
        raise ImportError("napari is required for interactive class review.") from exc

    table = _source_points(session, source_table)[1] if source_table is not None else session.get_atom_table(preferred="curated")
    if table.empty or "class_id" not in table.columns:
        raise ValueError("No classified atom-column table is available for review.")
    channel_name = image_channel or session.primary_channel
    image = session.get_processed_image(channel_name)
    origin_x, origin_y = session.get_processed_origin(channel_name)
    viewer = napari.Viewer(title=f"EM Atom Workbench - Class Review - {session.name}")
    viewer.add_image(image, name=f"processed_{channel_name}", translate=(origin_y, origin_x))
    for class_id, class_points in table.groupby("class_id", dropna=False):
        label = int(class_id) if pd.notna(class_id) else -1
        name = str(class_points["class_name"].iloc[0]) if "class_name" in class_points else f"class_{label}"
        color = str(class_points["class_color"].iloc[0]) if "class_color" in class_points else _class_color(label, AtomColumnClassificationConfig())
        data = np.column_stack(
            (
                class_points["y_px"].to_numpy(dtype=float) - origin_y,
                class_points["x_px"].to_numpy(dtype=float) - origin_x,
            )
        )
        layer = viewer.add_points(
            data,
            name=f"class_{label}_{name}",
            size=point_size,
            canvas_size_limits=(4, 8),
            face_color=color,
            border_color=color,
            border_width=0.0,
            border_width_is_relative=False,
            opacity=0.95,
            symbol="disc",
        )
        layer.metadata["class_id"] = label
        layer.metadata["class_name"] = name
        layer.metadata["class_color"] = color
        layer.metadata["origin"] = {"x": origin_x, "y": origin_y}
        layer.metadata["point_size"] = float(point_size)
        layer.editable = True
        _set_points_layer_add_defaults(layer, point_size=point_size, color=color)
    return viewer


def apply_class_review_from_viewer(
    session: AnalysisSession,
    viewer_handle: Any,
    *,
    source_table: str = "refined",
    match_radius_px: float = 3.0,
) -> AnalysisSession:
    source, original = _source_points(session, source_table)
    records: list[pd.Series] = []
    used: set[int] = set()
    next_atom_id = int(pd.to_numeric(original["atom_id"], errors="coerce").max()) + 1 if not original.empty else 0
    next_candidate_id = (
        int(pd.to_numeric(original["candidate_id"], errors="coerce").max()) + 1 if not original.empty else 0
    )
    original_coords = original[["x_px", "y_px"]].to_numpy(dtype=float)

    for layer in viewer_handle.layers:
        metadata = getattr(layer, "metadata", {})
        if "class_id" not in metadata:
            continue
        class_id = int(metadata.get("class_id", -1))
        class_name = str(metadata.get("class_name", _class_name(class_id, AtomColumnClassificationConfig())))
        class_color = str(metadata.get("class_color", _class_color(class_id, AtomColumnClassificationConfig())))
        origin = metadata.get("origin", {"x": 0, "y": 0})
        origin_x = int(origin.get("x", 0))
        origin_y = int(origin.get("y", 0))
        data = np.asarray(layer.data, dtype=float)
        for y_local, x_local in data:
            x_px = float(x_local + origin_x)
            y_px = float(y_local + origin_y)
            if original_coords.size:
                distances = np.linalg.norm(original_coords - np.asarray([x_px, y_px], dtype=float), axis=1)
                ordered = np.argsort(distances)
                matched_idx = None
                for idx in ordered:
                    idx = int(idx)
                    if idx not in used and distances[idx] <= float(match_radius_px):
                        matched_idx = idx
                        break
                if matched_idx is not None:
                    used.add(matched_idx)
                    row = original.iloc[matched_idx].copy()
                else:
                    row = pd.Series(dtype=object)
                    row["atom_id"] = next_atom_id
                    row["candidate_id"] = next_candidate_id
                    next_atom_id += 1
                    next_candidate_id += 1
            else:
                row = pd.Series(dtype=object)
                row["atom_id"] = next_atom_id
                row["candidate_id"] = next_candidate_id
                next_atom_id += 1
                next_candidate_id += 1
            row["x_px"] = x_px
            row["y_px"] = y_px
            row["class_id"] = class_id
            row["class_name"] = class_name
            row["class_color"] = class_color
            row["class_confidence"] = float(row.get("class_confidence", 1.0)) if pd.notna(row.get("class_confidence", 1.0)) else 1.0
            row["class_source"] = "manual_review"
            row["class_reviewed"] = True
            records.append(row)

    reviewed = pd.DataFrame(records).reset_index(drop=True)
    if source == "curated":
        session.curated_points = reviewed
    elif source == "candidate":
        session.candidate_points = reviewed
        session.refined_points = pd.DataFrame()
    else:
        session.refined_points = reviewed
    session.clear_downstream_results("classified")
    session.set_stage("classified")
    session.record_step(
        "apply_class_review_from_viewer",
        parameters={"source_table": source_table, "match_radius_px": match_radius_px},
        notes={"reviewed_count": len(reviewed)},
    )
    return session
