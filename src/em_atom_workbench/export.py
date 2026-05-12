from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from .plotting import (
    plot_atom_overlay,
    plot_class_overlay,
    plot_domain_annotation,
    plot_metric_map,
    plot_raw_image,
    plot_vector_field,
    save_figure_multi_format,
)
from .session import AnalysisSession, ExportConfig
from .utils import ensure_analysis_output_dir, serializable, write_json


def _apply_export_profile(config: ExportConfig) -> ExportConfig:
    profile = config.export_profile.lower()
    if profile == "minimal":
        return config
    if profile == "publication":
        return replace(
            config,
            save_session_pickle=False,
            save_fig_raw=True,
            save_fig_atom_overlay=True,
            save_fig_structure_map=True,
            save_fig_spacing_map=True,
            save_fig_strain_map=True,
        )
    if profile == "full_session":
        return replace(
            config,
            save_session_pickle=True,
            save_fig_raw=True,
            save_fig_atom_overlay=True,
            save_fig_structure_map=True,
            save_fig_spacing_map=True,
            save_fig_strain_map=True,
            save_fig_vector_map=True,
        )
    raise ValueError(f"Unsupported export_profile: {config.export_profile}")


def _write_table(table: pd.DataFrame, base_path: Path, formats: tuple[str, ...], overwrite: bool) -> list[Path]:
    saved: list[Path] = []
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        target = base_path.with_suffix(f".{fmt}")
        if target.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {target}")
        if fmt == "csv":
            table.to_csv(target, index=False)
        elif fmt == "parquet":
            table.to_parquet(target, index=False)
        else:
            raise ValueError(f"Unsupported table format: {fmt}")
        saved.append(target)
    return saved


def export_results(session: AnalysisSession, config: ExportConfig) -> Path:
    config = _apply_export_profile(config)
    output_root = ensure_analysis_output_dir(config.output_dir, session.name, create=True)
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    annotations_dir = output_root / "annotations"
    session_dir = output_root / "session"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "session": session.to_manifest_dict(),
        "export_profile": config.export_profile,
        "exported_files": [],
    }

    if config.save_atoms_table:
        atoms = session.get_atom_table(preferred="curated")
        if not atoms.empty:
            saved = _write_table(atoms, tables_dir / "atoms", config.table_formats, config.overwrite)
            manifest["exported_files"].extend([str(path) for path in saved])

    classification_features = getattr(session, "classification_features", pd.DataFrame())
    if isinstance(classification_features, pd.DataFrame) and not classification_features.empty:
        saved = _write_table(classification_features, tables_dir / "classification_features", config.table_formats, config.overwrite)
        manifest["exported_files"].extend([str(path) for path in saved])

    if config.save_metrics_table and not session.local_metrics.empty:
        saved = _write_table(session.local_metrics, tables_dir / "metrics", config.table_formats, config.overwrite)
        manifest["exported_files"].extend([str(path) for path in saved])

    strain_table = getattr(session, "strain_table", None)
    if isinstance(strain_table, pd.DataFrame) and not strain_table.empty:
        saved = _write_table(strain_table, tables_dir / "strain_table", config.table_formats, config.overwrite)
        manifest["exported_files"].extend([str(path) for path in saved])

    if config.save_annotations and session.annotations:
        annotation_path = annotations_dir / "annotations.json"
        if annotation_path.exists() and not config.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {annotation_path}")
        write_json(annotation_path, session.annotations)
        manifest["exported_files"].append(str(annotation_path))

    image = session.raw_image if session.raw_image is not None else session.get_processed_image()
    origin = (0, 0)
    if config.save_fig_raw and image is not None:
        fig, _ = plot_raw_image(image, title="Raw image")
        saved = save_figure_multi_format(fig, figures_dir / "raw_image", config.figure_formats)
        manifest["exported_files"].extend([str(path) for path in saved])

    atom_points = session.get_atom_table(preferred="curated")
    if config.save_fig_atom_overlay and image is not None and not atom_points.empty:
        if "class_id" in atom_points.columns:
            fig, _ = plot_class_overlay(image, atom_points, title="Atom-column class overlay", origin_xy=origin)
        else:
            fig, _ = plot_atom_overlay(image, atom_points, title="Atom overlay", origin_xy=origin)
        saved = save_figure_multi_format(fig, figures_dir / "atom_overlay", config.figure_formats)
        manifest["exported_files"].extend([str(path) for path in saved])

    if config.save_fig_structure_map and image is not None and not atom_points.empty and "annotation_label" in atom_points.columns:
        fig, _ = plot_domain_annotation(image, atom_points, title="Structure or domain annotation", origin_xy=origin)
        saved = save_figure_multi_format(fig, figures_dir / "structure_domain_map", config.figure_formats)
        manifest["exported_files"].extend([str(path) for path in saved])

    if config.save_fig_spacing_map and image is not None and not session.local_metrics.empty:
        joined = atom_points.merge(session.local_metrics[["atom_id", "mean_nn_distance_px"]], on="atom_id", how="inner")
        fig, _ = plot_metric_map(image, joined, joined["mean_nn_distance_px"], "Nearest-neighbor spacing", "Distance (px)", origin_xy=origin)
        saved = save_figure_multi_format(fig, figures_dir / "spacing_map", config.figure_formats)
        manifest["exported_files"].extend([str(path) for path in saved])

    if config.save_fig_strain_map and image is not None and not session.local_metrics.empty:
        joined = atom_points.merge(session.local_metrics[["atom_id", "strain_exx"]], on="atom_id", how="inner")
        fig, _ = plot_metric_map(image, joined, joined["strain_exx"], "Local strain-like map", "strain_exx", origin_xy=origin, cmap="coolwarm")
        saved = save_figure_multi_format(fig, figures_dir / "strain_map", config.figure_formats)
        manifest["exported_files"].extend([str(path) for path in saved])

    if config.save_fig_vector_map and image is not None and session.vector_fields:
        first_name = next(iter(session.vector_fields))
        fig, _ = plot_vector_field(image, session.vector_fields[first_name], title=f"Vector field - {first_name}", origin_xy=origin)
        saved = save_figure_multi_format(fig, figures_dir / "vector_map", config.figure_formats)
        manifest["exported_files"].extend([str(path) for path in saved])

    if config.save_session_pickle:
        pickle_path = session_dir / f"{session.name}.pkl"
        if pickle_path.exists() and not config.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {pickle_path}")
        session.save_pickle(pickle_path)
        manifest["exported_files"].append(str(pickle_path))

    manifest_path = write_json(output_root / "manifest.json", serializable(manifest))
    return manifest_path
