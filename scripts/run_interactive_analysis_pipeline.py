"""Run the atom-analysis workflow without driving the notebooks by hand.

This script keeps the same core logic as notebooks 01 and 04, but exposes it as
one blocking command. If candidate or class review is enabled, napari opens and
the script continues after the viewer window is closed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from em_atom_workbench.classification import AtomColumnClassificationConfig
from em_atom_workbench.intensity import compute_disk_intensity_table, summarize_disk_intensity
from em_atom_workbench.intensity_plotting import (
    plot_disk_aperture_preview,
    plot_disk_intensity_histogram,
    plot_disk_intensity_map,
)
from em_atom_workbench.notebook_workflows import (
    export_disk_intensity_analysis,
    export_final_atom_table_excel,
    initialize_disk_intensity_analysis,
    initialize_generic_classification_session,
    review_generic_candidates,
    run_atom_column_classification,
    run_generic_candidate_detection,
    run_generic_curation,
    run_generic_refinement,
    show_atom_column_class_review,
)
from em_atom_workbench.session import CurationConfig, DetectionConfig, PixelCalibration, RefinementConfig
from em_atom_workbench.workspace import (
    collect_project_manifest,
    export_stage_figure,
    get_stage_subdir,
    initialize_analysis_workspace,
    save_stage_session,
    write_json,
)


def _parse_key_value(items: list[str] | None, *, value_type: type = str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got {item!r}.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError(f"Missing key in {item!r}.")
        parsed[key] = None if value == "" else value_type(value)
    return parsed


def _parse_channels(items: list[str] | None) -> dict[str, str | None]:
    if not items:
        return {}
    return _parse_key_value(items)


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None or str(value).strip() == "":
        return None
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _print_result(label: str, result: Any) -> None:
    print(f"\n[{label}]")
    for message in getattr(result, "messages", []) or []:
        print(f"- {message}")
    for idx, table in enumerate(getattr(result, "tables", []) or [], start=1):
        if isinstance(table, pd.DataFrame):
            print(f"- table_{idx}: {len(table)} rows x {len(table.columns)} columns")


def _save_result_payload(result: Any, workspace: Any, stage_name: str, prefix: str, *, final: bool) -> dict[str, Any]:
    saved: dict[str, Any] = {"tables": {}, "figures": {}}
    table_dir = get_stage_subdir(workspace, stage_name, "tables")
    for idx, table in enumerate(getattr(result, "tables", []) or [], start=1):
        if not isinstance(table, pd.DataFrame):
            continue
        path = table_dir / f"{prefix}_table_{idx:02d}.csv"
        table.to_csv(path, index=False)
        saved["tables"][path.stem] = str(path)
    for idx, fig in enumerate(getattr(result, "figures", []) or [], start=1):
        if fig is None:
            continue
        paths = export_stage_figure(
            workspace,
            stage_name,
            f"{prefix}_figure_{idx:02d}",
            fig,
            final=final,
            formats=("png",),
            dpi=200,
        )
        saved["figures"][f"{prefix}_figure_{idx:02d}"] = [str(path) for path in paths]
        plt.close(fig)
    return saved


def _write_runner_manifest(workspace: Any, payload: dict[str, Any]) -> Path:
    manifest = {
        "workflow": "interactive_chat_pipeline",
        "runner": "scripts/run_interactive_analysis_pipeline.py",
        "workspace": str(workspace.root),
        **payload,
    }
    path = workspace.manifests_dir / "interactive_pipeline_manifest.json"
    write_json(path, _json_ready(manifest))
    collect_project_manifest(workspace)
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Notebook01-style atom-column analysis and optional Notebook04 disk-intensity "
            "mapping from the command line. Napari review windows block until closed."
        )
    )
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--dataset-id", default="dataset_001")
    parser.add_argument("--analysis-id", default="run_chat_001")
    parser.add_argument(
        "--channel",
        action="append",
        metavar="NAME=PATH",
        help="Input image channel. Repeat for multichannel input. If omitted, a synthetic demo is used.",
    )
    parser.add_argument("--primary-channel", default="channel_0")
    parser.add_argument(
        "--contrast-mode",
        action="append",
        metavar="NAME=bright_peak|dark_dip",
        help="Contrast mode per channel. Default is bright_peak.",
    )
    parser.add_argument(
        "--dataset-index",
        action="append",
        metavar="NAME=INDEX",
        help="Dataset index per channel for multi-dataset files.",
    )
    parser.add_argument("--pixel-size", type=float, default=None)
    parser.add_argument("--pixel-unit", default="px")
    parser.add_argument("--calibration-source", default="manual")

    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--min-distance", type=int, default=5)
    parser.add_argument("--threshold-rel", type=float, default=0.05)
    parser.add_argument("--min-prominence", type=float, default=0.02)
    parser.add_argument("--min-snr", type=float, default=1.2)
    parser.add_argument("--edge-margin", type=int, default=4)
    parser.add_argument("--patch-radius", type=int, default=6)
    parser.add_argument("--dedupe-radius-px", type=float, default=3.0)
    parser.add_argument("--merge-dedupe-radius-px", type=float, default=4.0)
    parser.add_argument("--max-candidates", type=int, default=None)

    parser.add_argument("--skip-candidate-review", action="store_true")
    parser.add_argument("--skip-class-review", action="store_true")
    parser.add_argument("--review-image-key", default="processed", choices=("raw", "processed"))
    parser.add_argument("--review-point-size", type=float, default=6.0)

    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--class-source-table", default="candidate", choices=("candidate", "refined", "curated"))
    parser.add_argument("--classification-random-state", type=int, default=0)

    parser.add_argument("--fit-half-window", type=int, default=5)
    parser.add_argument("--com-half-window", type=int, default=4)
    parser.add_argument("--initial-sigma-px", type=float, default=1.2)
    parser.add_argument("--max-center-shift-px", type=float, default=2.5)
    parser.add_argument("--nn-context-mode", default="all")

    parser.add_argument("--duplicate-radius-px", type=float, default=1.2)
    parser.add_argument("--min-quality-score", type=float, default=0.2)
    parser.add_argument("--max-fit-residual", type=float, default=0.3)
    parser.add_argument("--auto-drop-edge-points", action="store_true")
    parser.add_argument("--auto-drop-poor-fits", action="store_true")

    parser.add_argument("--run-04", action="store_true")
    parser.add_argument(
        "--intensity-coordinate-source",
        default="refined",
        choices=("candidate", "refined", "curated"),
    )
    parser.add_argument("--intensity-class-ids", default=None, help="Comma-separated class ids for Notebook04.")
    parser.add_argument("--disk-radius-px", type=float, default=2.0)
    parser.add_argument("--hist-bins", type=int, default=30)
    parser.add_argument("--map-point-size", type=float, default=32.0)
    parser.add_argument("--save-preview-figures", action="store_true")
    parser.add_argument("--no-final-figures", action="store_true")
    return parser


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    channels = _parse_channels(args.channel)
    contrast_modes = _parse_key_value(args.contrast_mode) if args.contrast_mode else {
        name: "bright_peak" for name in (channels or {"channel_0": None})
    }
    dataset_indices = _parse_key_value(args.dataset_index, value_type=int)

    calibration = None
    if args.pixel_size is not None:
        calibration = PixelCalibration(
            size=float(args.pixel_size),
            unit=str(args.pixel_unit),
            source=str(args.calibration_source),
        )

    workspace = initialize_analysis_workspace(
        output_root=Path(args.output_root),
        dataset_id=args.dataset_id,
        analysis_id=args.analysis_id,
    )
    result_root = workspace.root
    print(f"workspace: {workspace.root}")

    init_result = initialize_generic_classification_session(
        result_root=result_root,
        channels=channels,
        primary_channel=args.primary_channel,
        channel_contrast_modes=contrast_modes,
        channel_dataset_indices=dataset_indices,
        manual_calibration=calibration,
    )
    session = init_result.session
    save_stage_session(session, workspace, "01_loaded", update_active=True)
    _print_result("initialize", init_result)

    detection_config = DetectionConfig(
        contrast_mode=str(contrast_modes.get(session.primary_channel, "bright_peak")),
        gaussian_sigma=args.gaussian_sigma,
        min_distance=args.min_distance,
        threshold_rel=args.threshold_rel,
        min_prominence=args.min_prominence,
        min_snr=args.min_snr,
        edge_margin=args.edge_margin,
        patch_radius=args.patch_radius,
        max_candidates=args.max_candidates,
        dedupe_radius_px=args.dedupe_radius_px,
    )
    detection_configs = {
        channel_name: DetectionConfig(
            contrast_mode=str(contrast_modes.get(channel_name, detection_config.contrast_mode)),
            gaussian_sigma=detection_config.gaussian_sigma,
            min_distance=detection_config.min_distance,
            threshold_rel=detection_config.threshold_rel,
            min_prominence=detection_config.min_prominence,
            min_snr=detection_config.min_snr,
            edge_margin=detection_config.edge_margin,
            patch_radius=detection_config.patch_radius,
            max_candidates=detection_config.max_candidates,
            dedupe_radius_px=detection_config.dedupe_radius_px,
        )
        for channel_name in session.list_channels()
    }
    detect_result = run_generic_candidate_detection(
        session,
        result_root=result_root,
        detection_configs_by_channel=detection_configs,
        merge_dedupe_radius_px=args.merge_dedupe_radius_px,
        open_viewer=False,
    )
    session = detect_result.session
    _save_result_payload(detect_result, workspace, "01_findatom", "01_detection", final=False)
    _print_result("candidate detection", detect_result)

    if args.skip_candidate_review:
        print("\n[candidate review]\n- skipped; auto-accepting detected candidates")
        session.set_stage("candidate_reviewed")
    else:
        print("\n[candidate review]\n- napari will open; edit candidate points, then close the window to continue.")
        candidate_result = review_generic_candidates(
            session,
            result_root=result_root,
            open_viewer=True,
            image_channel=session.primary_channel,
            image_key=args.review_image_key,
            point_size=args.review_point_size,
        )
        session = candidate_result.session
        _save_result_payload(candidate_result, workspace, "01_findatom", "01_candidate_review", final=False)
        _print_result("candidate review", candidate_result)
    save_stage_session(session, workspace, "01_candidate_reviewed", update_active=True)

    classification_config = AtomColumnClassificationConfig(
        feature_channels=None,
        feature_patch_radii=args.patch_radius,
        source_table=args.class_source_table,
        n_classes=args.n_classes,
        random_state=args.classification_random_state,
    )
    classification_result = run_atom_column_classification(
        session,
        result_root=result_root,
        classification_config=classification_config,
    )
    session = classification_result.session
    _save_result_payload(classification_result, workspace, "01_findatom", "01_classification", final=False)
    _print_result("classification", classification_result)

    if args.skip_class_review:
        print("\n[class review]\n- skipped; auto-accepting automatic class labels")
    else:
        print("\n[class review]\n- napari will open; edit class layers/labels, then close the window to continue.")
        class_review_result = show_atom_column_class_review(
            session,
            result_root=result_root,
            open_viewer=True,
            image_channel=session.primary_channel,
            point_size=args.review_point_size,
            source_table=args.class_source_table,
        )
        session = class_review_result.session
        _print_result("class review", class_review_result)
    save_stage_session(session, workspace, "01_class_reviewed", update_active=True)

    refinement_config = RefinementConfig(
        fit_half_window=args.fit_half_window,
        com_half_window=args.com_half_window,
        initial_sigma_px=args.initial_sigma_px,
        max_center_shift_px=args.max_center_shift_px,
    )
    refinement_result = run_generic_refinement(
        session,
        result_root=result_root,
        refinement_config=refinement_config,
        source_table="candidate",
        nn_context_mode=args.nn_context_mode,
    )
    session = refinement_result.session
    _save_result_payload(refinement_result, workspace, "01_findatom", "01_refinement", final=False)
    save_stage_session(session, workspace, "01_refined", update_active=True)
    _print_result("refinement", refinement_result)

    curation_config = CurationConfig(
        duplicate_radius_px=args.duplicate_radius_px,
        edge_margin=args.edge_margin,
        min_quality_score=args.min_quality_score,
        max_fit_residual=args.max_fit_residual,
        auto_drop_duplicates=True,
        auto_drop_edge_points=args.auto_drop_edge_points,
        auto_drop_poor_fits=args.auto_drop_poor_fits,
    )
    curation_result = run_generic_curation(
        session,
        result_root=result_root,
        curation_config=curation_config,
    )
    session = curation_result.session
    _save_result_payload(curation_result, workspace, "01_findatom", "01_curation", final=True)
    save_stage_session(session, workspace, "01_final_curated", update_active=True)
    _print_result("curation", curation_result)

    export_result = export_final_atom_table_excel(
        session,
        workspace=workspace,
        session_source="01_final_curated",
        filename="final_atom_columns.xlsx",
        kept_only_sheet=True,
        save_csv=True,
    )
    _print_result("final atom table export", export_result)

    intensity_manifest: dict[str, Any] | None = None
    if args.run_04:
        class_ids = _parse_int_list(args.intensity_class_ids)
        print("\n[04 disk intensity]\n- computing fixed-radius disk intensity table and figures")
        context = initialize_disk_intensity_analysis(
            workspace=workspace,
            session_source="01_final_curated",
            use_active_session=False,
            result_root=result_root,
            coordinate_source=args.intensity_coordinate_source,
            use_keep_only=True,
            class_id_filter=class_ids,
            image_channel=None,
            image_key="raw",
        )
        intensity_table = compute_disk_intensity_table(
            context["points"],
            context["image"],
            disk_radius_px=args.disk_radius_px,
            channel_name=context["image_channel"],
            image_key=context["image_key"],
            coordinate_source=args.intensity_coordinate_source,
        )
        summary_table = summarize_disk_intensity(intensity_table)
        preview_fig, _ = plot_disk_aperture_preview(
            context["image"],
            context["points"],
            disk_radius_px=args.disk_radius_px,
            title=f"Disk aperture preview ({args.intensity_coordinate_source} coordinates)",
        )
        map_result = plot_disk_intensity_map(
            context["image"],
            intensity_table,
            metric="disk_intensity_sum",
            point_size=args.map_point_size,
            title=f"Disk-integrated intensity map ({args.intensity_coordinate_source} coordinates)",
        )
        map_fig = map_result[0]
        hist_fig, _ = plot_disk_intensity_histogram(
            intensity_table,
            metric="disk_intensity_sum",
            bins=args.hist_bins,
            group_by_class=True,
            title=f"Disk-integrated intensity histogram ({args.intensity_coordinate_source} coordinates)",
        )
        config = {
            "session_source": "01_final_curated",
            "session_path": None,
            "use_active_session": False,
            "coordinate_source": args.intensity_coordinate_source,
            "source_table": context["source_table"],
            "use_keep_only": True,
            "image_channel": context["image_channel"],
            "image_key": context["image_key"],
            "target_class_ids": class_ids,
            "target_class_names": None,
            "disk_radius_px": args.disk_radius_px,
            "map_metric": "disk_intensity_sum",
            "histogram_metric": "disk_intensity_sum",
            "hist_bins": args.hist_bins,
            "save_preview_figures": bool(args.save_preview_figures),
            "save_final_figures": not bool(args.no_final_figures),
        }
        intensity_manifest = export_disk_intensity_analysis(
            workspace=workspace,
            result_root=result_root,
            session=context["session"],
            intensity_table=intensity_table,
            summary_table=summary_table,
            config=config,
            preview_figures={"04A_disk_aperture_preview": preview_fig},
            final_figures={
                "04A_disk_intensity_map": map_fig,
                "04B_disk_intensity_histogram": hist_fig,
            },
            save_preview_figures=args.save_preview_figures,
            save_final_figures=not args.no_final_figures,
            figure_specs={
                "04A_disk_intensity_map": {
                    "save": True,
                    "formats": ("pdf", "png", "svg"),
                    "dpi": 600,
                    "title": f"Disk-integrated intensity map ({args.intensity_coordinate_source} coordinates)",
                    "show_title": True,
                },
                "04B_disk_intensity_histogram": {
                    "save": True,
                    "formats": ("pdf", "png", "svg"),
                    "dpi": 600,
                    "title": f"Disk-integrated intensity histogram ({args.intensity_coordinate_source} coordinates)",
                    "show_title": True,
                },
            },
        )
        plt.close(preview_fig)
        plt.close(map_fig)
        plt.close(hist_fig)
        print(f"- rows: {len(intensity_table)}")
        print(f"- manifest: {intensity_manifest.get('manifest')}")

    runner_manifest = _write_runner_manifest(
        workspace,
        {
            "args": vars(args),
            "final_session": str(workspace.sessions_dir / "01_final_curated.pkl"),
            "active_session": str(workspace.state_dir / "active_session.pkl"),
            "run_04": bool(args.run_04),
            "intensity_manifest": intensity_manifest,
        },
    )
    print(f"\n[done]\n- workspace: {workspace.root}\n- runner manifest: {runner_manifest}")
    return {"workspace": str(workspace.root), "runner_manifest": str(runner_manifest)}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = run_pipeline(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
