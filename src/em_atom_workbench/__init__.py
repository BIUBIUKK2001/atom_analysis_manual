"""Notebook-first workbench for atomic-resolution microscopy image analysis."""

from .annotation import save_annotations, suggest_annotations
from .classification import (
    AtomColumnClassificationConfig,
    apply_class_name_map,
    apply_class_review_from_viewer,
    classification_summary_table,
    classify_atom_columns,
    extract_atom_column_features,
    launch_class_review_napari,
)
from .curate import (
    apply_candidate_edits_from_viewer,
    apply_curation_from_viewer,
    curate_points,
    edit_candidates_with_napari,
    edit_hfo2_heavy_candidates_with_napari,
    edit_hfo2_light_candidates_with_napari,
    launch_napari_candidate_editor,
    launch_napari_curation,
)
from .detect import (
    detect_candidates,
    detect_hfo2_heavy_candidates,
    detect_hfo2_light_candidates,
    detect_hfo2_multichannel_candidates,
    detect_multichannel_candidates,
)
from .export import export_results
from .io import load_image, load_image_bundle
from .lattice import build_neighbor_graph
from .metrics import compute_local_metrics
from .polarization import (
    attach_vector_field,
    build_vector_field,
    vector_field_from_point_matches,
)
from .plotting import (
    launch_detection_napari_viewer,
    launch_preprocess_napari_viewer,
    launch_refinement_napari_viewer,
    plot_class_feature_scatter_matrix,
    plot_class_overlay,
)
from .preprocess import preprocess_channels, preprocess_image
from .refine import refine_points, refine_points_by_class
from .reference import (
    ReferenceLattice,
    ReferenceLatticeConfig,
    ReferenceLatticeSuggestion,
    ReferenceLatticeSuggestionConfig,
    build_reference_lattice,
    build_reference_lattice_from_suggestion,
    suggest_reference_lattices,
)
from .session import (
    AnalysisSession,
    AnnotationConfig,
    CurationConfig,
    DetectionConfig,
    ExportConfig,
    HfO2MultichannelDetectionConfig,
    LatticeConfig,
    MetricsConfig,
    PixelCalibration,
    PreprocessConfig,
    RefinementConfig,
    VectorFieldConfig,
)
from .strain import LocalAffineStrainConfig, compute_local_affine_strain
from .vpcf import (
    BatchVPCFResult,
    LocalVPCFResult,
    VPCFConfig,
    compute_batch_vpcf,
    compute_local_vpcf,
    compute_session_vpcf,
    plot_vpcf,
    points_for_vpcf,
)

__all__ = [
    "AnalysisSession",
    "AnnotationConfig",
    "AtomColumnClassificationConfig",
    "BatchVPCFResult",
    "CurationConfig",
    "DetectionConfig",
    "ExportConfig",
    "HfO2MultichannelDetectionConfig",
    "LatticeConfig",
    "LocalVPCFResult",
    "LocalAffineStrainConfig",
    "MetricsConfig",
    "PixelCalibration",
    "PreprocessConfig",
    "ReferenceLattice",
    "ReferenceLatticeConfig",
    "ReferenceLatticeSuggestion",
    "ReferenceLatticeSuggestionConfig",
    "RefinementConfig",
    "VPCFConfig",
    "VectorFieldConfig",
    "apply_class_name_map",
    "apply_class_review_from_viewer",
    "apply_candidate_edits_from_viewer",
    "apply_curation_from_viewer",
    "attach_vector_field",
    "build_neighbor_graph",
    "build_reference_lattice",
    "build_reference_lattice_from_suggestion",
    "build_vector_field",
    "classification_summary_table",
    "compute_batch_vpcf",
    "compute_local_metrics",
    "compute_local_affine_strain",
    "compute_local_vpcf",
    "compute_session_vpcf",
    "classify_atom_columns",
    "curate_points",
    "detect_candidates",
    "detect_hfo2_heavy_candidates",
    "detect_hfo2_light_candidates",
    "detect_hfo2_multichannel_candidates",
    "detect_multichannel_candidates",
    "edit_hfo2_heavy_candidates_with_napari",
    "edit_hfo2_light_candidates_with_napari",
    "edit_candidates_with_napari",
    "export_results",
    "extract_atom_column_features",
    "launch_class_review_napari",
    "launch_napari_candidate_editor",
    "launch_detection_napari_viewer",
    "launch_preprocess_napari_viewer",
    "launch_refinement_napari_viewer",
    "launch_napari_curation",
    "load_image",
    "load_image_bundle",
    "plot_vpcf",
    "plot_class_feature_scatter_matrix",
    "plot_class_overlay",
    "points_for_vpcf",
    "preprocess_channels",
    "preprocess_image",
    "refine_points",
    "refine_points_by_class",
    "save_annotations",
    "suggest_annotations",
    "suggest_reference_lattices",
    "vector_field_from_point_matches",
]

from .utils import load_or_connect_session, save_active_session, save_checkpoint

__all__.extend(
    [
        "load_or_connect_session",
        "save_active_session",
        "save_checkpoint",
    ]
)
