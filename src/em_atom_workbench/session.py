from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any
import os
import pickle
import tempfile

import numpy as np
import pandas as pd


def _default_frame() -> pd.DataFrame:
    return pd.DataFrame()


_UNSET = object()


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_serializable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


@dataclass
class PixelCalibration:
    size: float | None = None
    unit: str = "px"
    source: str = "unknown"

    @property
    def is_calibrated(self) -> bool:
        return self.size is not None and self.unit != "px"


@dataclass
class PreprocessConfig:
    contrast_mode: str = "bright_peak"
    normalization_quantiles: tuple[float, float] = (0.01, 0.99)
    invert_for_dark_dip: bool = True
    denoise_method: str = "wiener"
    denoise_sigma: float = 0.0
    median_size: int = 3
    bilateral_sigma_color: float = 0.05
    bilateral_sigma_spatial: float = 1.0
    psf_kernel: np.ndarray | None = None
    wiener_psf_size: int = 9
    wiener_psf_sigma: float = 1.0
    wiener_balance: float = 0.05
    wiener_clip: bool = False
    background_sigma: float = 0.0
    local_contrast: bool = False
    clahe_clip_limit: float = 0.01
    roi: tuple[int, int, int, int] | None = None
    edge_mask_width: int = 0


@dataclass
class DetectionConfig:
    contrast_mode: str = "bright_peak"
    method: str = "local_extrema"
    gaussian_sigma: float = 1.0
    min_distance: int = 5
    threshold_abs: float | None = None
    threshold_rel: float = 0.05
    min_prominence: float = 0.02
    min_snr: float = 1.5
    edge_margin: int = 6
    expected_spacing_px: float | None = None
    patch_radius: int = 5
    max_candidates: int | None = None
    dedupe_radius_px: float | None = None


@dataclass
class ChannelState:
    input_path: str | None = None
    dataset_index: int | None = None
    raw_image: np.ndarray | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    preprocess_result: dict[str, Any] = field(default_factory=dict)
    contrast_mode: str = "bright_peak"


@dataclass
class RefinementConfig:
    mode: str = "legacy"
    fit_half_window: int = 5
    com_half_window: int = 4
    nn_radius_fraction: float = 0.38
    min_patch_radius_px: int = 6
    max_patch_radius_px: int = 14
    initial_sigma_px: float = 1.2
    min_sigma_px: float = 0.5
    max_sigma_px: float = 4.0
    max_center_shift_px: float = 2.5
    max_nfev: int = 4000
    gaussian_retry_count: int = 4
    gaussian_retry_shrink_factor: float = 0.9
    sigma_ratio_limit: float = 4.0
    fit_edge_margin_px: float = 1.0
    gaussian_image_source: str = "raw"
    fallback_to_quadratic: bool = True
    fallback_to_com: bool = True
    quality_floor: float = 0.0
    overlap_trigger_px: float | None = None


@dataclass
class HfO2MultichannelDetectionConfig:
    candidate_mode: str = "hfo2_multichannel"
    heavy_channel: str = "haadf"
    light_channel: str = "idpc"
    confirm_channel: str | None = "abf"
    heavy_detection: DetectionConfig = field(
        default_factory=lambda: DetectionConfig(
            contrast_mode="bright_peak",
            gaussian_sigma=1.0,
            min_distance=6,
            threshold_rel=0.12,
            min_prominence=0.05,
            min_snr=1.2,
            edge_margin=4,
            patch_radius=6,
            dedupe_radius_px=3.5,
        )
    )
    light_detection: DetectionConfig = field(
        default_factory=lambda: DetectionConfig(
            contrast_mode="bright_peak",
            gaussian_sigma=0.6,
            min_distance=2,
            threshold_rel=0.0,
            min_prominence=0.025,
            min_snr=0.8,
            edge_margin=4,
            patch_radius=5,
            dedupe_radius_px=2.0,
        )
    )
    heavy_refinement: RefinementConfig = field(
        default_factory=lambda: RefinementConfig(
            mode="adaptive_atomap",
            fit_half_window=6,
            com_half_window=4,
            min_patch_radius_px=5,
            max_patch_radius_px=12,
            gaussian_image_source="processed",
            gaussian_retry_count=1,
        )
    )
    proposal_window_radius_px: int = 5
    proposal_max_offset_px: float = 2.5
    proposal_dedupe_radius_px: float = 2.0
    heavy_neighbor_count: int = 4
    midpoint_max_distance_factor: float = 1.35
    void_max_edge_factor: float = 1.6
    heavy_suppression_radius_px: float = 3.5
    heavy_suppression_sigma_px: float = 1.6
    weak_min_prominence: float = 0.01
    weak_min_snr: float = 0.3
    confirm_min_prominence: float = 0.01
    confirm_min_snr: float = 0.25
    min_light_heavy_separation_px: float = 1.25
    overlap_trigger_px: float = 4.0


@dataclass
class CurationConfig:
    duplicate_radius_px: float = 1.2
    edge_margin: int = 6
    min_quality_score: float = 0.2
    max_fit_residual: float = 0.3
    min_spacing_px: float | None = None
    auto_drop_duplicates: bool = True
    auto_drop_edge_points: bool = False
    auto_drop_poor_fits: bool = False


@dataclass
class LatticeConfig:
    k_neighbors: int = 6
    max_distance_px: float | None = None
    mutual_only: bool = True
    min_basis_angle_deg: float = 25.0
    reference_basis: tuple[tuple[float, float], tuple[float, float]] | None = None


@dataclass
class MetricsConfig:
    reference_basis: tuple[tuple[float, float], tuple[float, float]] | None = None
    reference_points: pd.DataFrame | None = None
    reference_match_column: str = "atom_id"
    orientation_wrap_deg: float = 180.0


@dataclass
class AnnotationConfig:
    feature_columns: tuple[str, ...] = (
        "mean_nn_distance_px",
        "basis_a_length_px",
        "basis_b_length_px",
        "basis_angle_deg",
        "local_orientation_deg",
    )
    n_clusters: int = 3
    include_fft: bool = False
    patch_radius: int = 12
    random_state: int = 0


@dataclass
class VectorFieldConfig:
    name: str = "vector_field"
    scale: float = 1.0
    unit: str = "px"
    color_by: str | None = None


@dataclass
class ExportConfig:
    output_dir: str | Path
    export_profile: str = "minimal"
    save_atoms_table: bool = True
    save_metrics_table: bool = True
    save_annotations: bool = True
    save_session_pickle: bool = False
    save_fig_raw: bool = False
    save_fig_atom_overlay: bool = True
    save_fig_structure_map: bool = False
    save_fig_spacing_map: bool = False
    save_fig_strain_map: bool = False
    save_fig_vector_map: bool = False
    figure_formats: tuple[str, ...] = ("png", "pdf", "tiff")
    table_formats: tuple[str, ...] = ("csv", "parquet")
    overwrite: bool = False


@dataclass
class AnalysisSession:
    name: str = "analysis_session"
    input_path: str | None = None
    dataset_index: int | None = None
    raw_image: np.ndarray | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    pixel_calibration: PixelCalibration = field(default_factory=PixelCalibration)
    contrast_mode: str = "bright_peak"
    primary_channel: str = "primary"
    channels: dict[str, ChannelState] = field(default_factory=dict)
    workflow_mode: str = "single_channel"
    workflow_settings: dict[str, Any] = field(default_factory=dict)
    current_stage: str = "loaded"
    preprocess_result: dict[str, Any] = field(default_factory=dict)
    candidate_points: pd.DataFrame = field(default_factory=_default_frame)
    refined_points: pd.DataFrame = field(default_factory=_default_frame)
    curated_points: pd.DataFrame = field(default_factory=_default_frame)
    classification_features: pd.DataFrame = field(default_factory=_default_frame)
    classification_summary: dict[str, Any] = field(default_factory=dict)
    neighbor_graph: dict[str, Any] = field(default_factory=dict)
    local_metrics: pd.DataFrame = field(default_factory=_default_frame)
    reference_lattice: dict[str, Any] = field(default_factory=dict)
    strain_table: pd.DataFrame = field(default_factory=_default_frame)
    vpcf_results: dict[str, Any] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)
    vector_fields: dict[str, Any] = field(default_factory=dict)
    provenance: list[dict[str, Any]] = field(default_factory=list)

    def record_step(self, step_name: str, parameters: Any | None = None, notes: dict[str, Any] | None = None) -> None:
        self.provenance.append(
            {
                "step": step_name,
                "parameters": _to_serializable(parameters or {}),
                "notes": _to_serializable(notes or {}),
            }
        )

    def set_stage(self, stage: str) -> None:
        self.current_stage = str(stage)

    def set_workflow(self, mode: str, settings: dict[str, Any] | None = None) -> None:
        self.workflow_mode = str(mode)
        self.workflow_settings = dict(settings or {})
        self.record_step(
            "set_workflow",
            parameters={
                "workflow_mode": self.workflow_mode,
                "workflow_settings": self.workflow_settings,
            },
        )

    def _ensure_channels(self) -> None:
        primary_channel = str(getattr(self, "primary_channel", "primary") or "primary")
        channels = dict(getattr(self, "channels", {}) or {})
        if primary_channel not in channels:
            channels[primary_channel] = ChannelState(
                input_path=self.input_path,
                dataset_index=self.dataset_index,
                raw_image=self.raw_image,
                raw_metadata=dict(self.raw_metadata),
                preprocess_result=dict(self.preprocess_result),
                contrast_mode=self.contrast_mode,
            )
        self.primary_channel = primary_channel
        self.channels = channels
        self.sync_primary_channel_alias()

    def sync_primary_channel_alias(self) -> None:
        primary_channel = str(getattr(self, "primary_channel", "primary") or "primary")
        channels = dict(getattr(self, "channels", {}) or {})
        if primary_channel not in channels:
            return
        primary = channels[primary_channel]
        self.input_path = primary.input_path
        self.dataset_index = primary.dataset_index
        self.raw_image = primary.raw_image
        self.raw_metadata = dict(primary.raw_metadata)
        self.preprocess_result = dict(primary.preprocess_result)
        self.contrast_mode = primary.contrast_mode

    def get_channel_state(self, channel_name: str | None = None) -> ChannelState:
        self._ensure_channels()
        channel_key = str(channel_name or self.primary_channel)
        if channel_key not in self.channels:
            raise KeyError(f"Unknown channel: {channel_key}")
        return self.channels[channel_key]

    def set_primary_channel(self, channel_name: str) -> None:
        self._ensure_channels()
        channel_key = str(channel_name)
        if channel_key not in self.channels:
            raise KeyError(f"Unknown channel: {channel_key}")
        self.primary_channel = channel_key
        self.sync_primary_channel_alias()

    def set_channel_state(
        self,
        channel_name: str,
        *,
        input_path: Any = _UNSET,
        dataset_index: Any = _UNSET,
        raw_image: Any = _UNSET,
        raw_metadata: Any = _UNSET,
        preprocess_result: Any = _UNSET,
        contrast_mode: Any = _UNSET,
    ) -> ChannelState:
        self._ensure_channels()
        channel_key = str(channel_name)
        state = self.channels.get(channel_key, ChannelState())
        if input_path is not _UNSET:
            state.input_path = input_path
        if dataset_index is not _UNSET:
            state.dataset_index = dataset_index
        if raw_image is not _UNSET:
            state.raw_image = raw_image
        if raw_metadata is not _UNSET:
            state.raw_metadata = dict(raw_metadata or {})
        if preprocess_result is not _UNSET:
            state.preprocess_result = dict(preprocess_result or {})
        if contrast_mode is not _UNSET:
            state.contrast_mode = str(contrast_mode)
        self.channels[channel_key] = state
        if channel_key == self.primary_channel:
            self.sync_primary_channel_alias()
        return state

    def list_channels(self) -> list[str]:
        self._ensure_channels()
        return list(self.channels.keys())

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.dataset_index = getattr(self, "dataset_index", None)
        self.primary_channel = getattr(self, "primary_channel", "primary")
        self.workflow_mode = str(getattr(self, "workflow_mode", "single_channel") or "single_channel")
        self.workflow_settings = dict(getattr(self, "workflow_settings", {}) or {})
        channels = getattr(self, "channels", {})
        if channels:
            self.channels = {
                str(name): value if isinstance(value, ChannelState) else ChannelState(**value)
                for name, value in dict(channels).items()
            }
        else:
            self.channels = {}
        self.current_stage = getattr(self, "current_stage", "loaded")
        self.preprocess_result = getattr(self, "preprocess_result", {})
        self.candidate_points = getattr(self, "candidate_points", _default_frame())
        self.refined_points = getattr(self, "refined_points", _default_frame())
        self.curated_points = getattr(self, "curated_points", _default_frame())
        self.classification_features = getattr(self, "classification_features", _default_frame())
        self.classification_summary = dict(getattr(self, "classification_summary", {}) or {})
        self.neighbor_graph = getattr(self, "neighbor_graph", {})
        self.local_metrics = getattr(self, "local_metrics", _default_frame())
        self.reference_lattice = dict(getattr(self, "reference_lattice", {}) or {})
        self.strain_table = getattr(self, "strain_table", _default_frame())
        self.vpcf_results = getattr(self, "vpcf_results", {})
        self.annotations = getattr(self, "annotations", {})
        self.vector_fields = getattr(self, "vector_fields", {})
        self.provenance = getattr(self, "provenance", [])
        self._ensure_channels()

    def get_processed_image(self, channel_name: str | None = None) -> np.ndarray:
        channel = self.get_channel_state(channel_name)
        if channel.preprocess_result.get("processed_image") is not None:
            return channel.preprocess_result["processed_image"]
        if channel.raw_image is None:
            raise ValueError("Session does not contain an image.")
        return channel.raw_image

    def get_processed_origin(self, channel_name: str | None = None) -> tuple[int, int]:
        channel = self.get_channel_state(channel_name)
        return (
            int(channel.preprocess_result.get("origin_x", 0)),
            int(channel.preprocess_result.get("origin_y", 0)),
        )

    def get_atom_table(self, preferred: str = "curated") -> pd.DataFrame:
        if preferred == "curated" and not self.curated_points.empty:
            return self.curated_points
        if not self.refined_points.empty:
            return self.refined_points
        if not self.candidate_points.empty:
            return self.candidate_points
        return pd.DataFrame()

    def ensure_atom_ids(self, table_name: str) -> pd.DataFrame:
        table = getattr(self, table_name)
        if table.empty:
            return table
        table = table.copy()
        if "atom_id" not in table.columns:
            table.insert(0, "atom_id", np.arange(len(table), dtype=int))
        else:
            table["atom_id"] = pd.Series(table["atom_id"]).fillna(method="ffill").fillna(0).astype(int)
        setattr(self, table_name, table)
        return table

    def clear_downstream_results(self, stage: str) -> None:
        if stage == "preprocess":
            self.candidate_points = _default_frame()
            self.refined_points = _default_frame()
            self.curated_points = _default_frame()
            self.classification_features = _default_frame()
            self.classification_summary = {}
            self.neighbor_graph = {}
            self.local_metrics = _default_frame()
            self.reference_lattice = {}
            self.strain_table = _default_frame()
            self.vpcf_results = {}
            self.annotations = {}
            self.vector_fields = {}
            return
        if stage == "detect":
            self.refined_points = _default_frame()
            self.curated_points = _default_frame()
            self.classification_features = _default_frame()
            self.classification_summary = {}
            self.neighbor_graph = {}
            self.local_metrics = _default_frame()
            self.reference_lattice = {}
            self.strain_table = _default_frame()
            self.vpcf_results = {}
            self.annotations = {}
            self.vector_fields = {}
            return
        if stage == "refine":
            self.curated_points = _default_frame()
            self.classification_features = _default_frame()
            self.classification_summary = {}
            self.neighbor_graph = {}
            self.local_metrics = _default_frame()
            self.reference_lattice = {}
            self.strain_table = _default_frame()
            self.vpcf_results = {}
            self.annotations = {}
            self.vector_fields = {}
            return
        if stage == "classified":
            self.curated_points = _default_frame()
            self.neighbor_graph = {}
            self.local_metrics = _default_frame()
            self.reference_lattice = {}
            self.strain_table = _default_frame()
            self.vpcf_results = {}
            self.annotations = {}
            self.vector_fields = {}
            return
        if stage == "curate":
            self.neighbor_graph = {}
            self.local_metrics = _default_frame()
            self.reference_lattice = {}
            self.strain_table = _default_frame()
            self.vpcf_results = {}
            self.annotations = {}
            self.vector_fields = {}
            return
        if stage == "vpcf":
            self.annotations = {}
            self.vector_fields = {}
            return
        raise ValueError(f"Unsupported stage: {stage}")

    def save_pickle(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                dir=output_path.parent,
                prefix=f"{output_path.stem}_",
                suffix=".tmp",
            ) as handle:
                temp_path = Path(handle.name)
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, output_path)
        except Exception:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        return output_path

    @classmethod
    def load_pickle(cls, path: str | Path) -> "AnalysisSession":
        with Path(path).open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, cls):
            raise TypeError("Loaded object is not an AnalysisSession.")
        return loaded

    def to_manifest_dict(self) -> dict[str, Any]:
        self._ensure_channels()
        reference_lattice = dict(getattr(self, "reference_lattice", {}) or {})
        reference_lattice_keys = sorted(str(key) for key in reference_lattice.keys())
        strain_table = getattr(self, "strain_table", _default_frame())
        strain_table_rows = int(len(strain_table))
        return {
            "name": self.name,
            "input_path": self.input_path,
            "dataset_index": self.dataset_index,
            "current_stage": self.current_stage,
            "pixel_calibration": _to_serializable(self.pixel_calibration),
            "contrast_mode": self.contrast_mode,
            "primary_channel": self.primary_channel,
            "workflow_mode": self.workflow_mode,
            "workflow_settings": _to_serializable(self.workflow_settings),
            "channels": {
                name: {
                    "input_path": channel.input_path,
                    "dataset_index": channel.dataset_index,
                    "contrast_mode": channel.contrast_mode,
                    "has_raw_image": channel.raw_image is not None,
                    "has_processed_image": channel.preprocess_result.get("processed_image") is not None,
                }
                for name, channel in self.channels.items()
            },
            "raw_metadata": _to_serializable(self.raw_metadata),
            "provenance": _to_serializable(self.provenance),
            "table_sizes": {
                "candidate_points": int(len(self.candidate_points)),
                "refined_points": int(len(self.refined_points)),
                "curated_points": int(len(self.curated_points)),
                "classification_features": int(len(self.classification_features)),
                "local_metrics": int(len(self.local_metrics)),
                "strain_table": strain_table_rows,
            },
            "classification_summary": _to_serializable(self.classification_summary),
            "has_strain_table": strain_table_rows > 0,
            "reference_lattice_keys": reference_lattice_keys,
            "vpcf_result_keys": list(self.vpcf_results.keys()),
            "annotation_count": int(len(self.annotations.get("records", []))),
            "vector_field_names": list(self.vector_fields.keys()),
        }
