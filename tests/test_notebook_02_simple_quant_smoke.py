from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def test_build_02_simple_quant_notebook_generates_compilable_notebook() -> None:
    subprocess.run([sys.executable, "scripts/build_02_simple_quant_notebook.py"], check=True)
    notebook_path = Path("notebooks") / "02_Simple_quantitative_spacing_analysis.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    joined = "\n".join(_cell_source(cell) for cell in notebook.get("cells", []))
    source_by_id = {str(cell.get("id", "")): _cell_source(cell) for cell in notebook.get("cells", [])}

    for parameter in (
        "SOURCE_TABLE",
        "OPEN_ROI_PICKER",
        "OPEN_BASIS_VECTOR_PICKER",
        "BASIS_MODE",
        "BASIS_ROLES",
        "BASIS_VECTOR_SPECS",
        "GLOBAL_BASIS_FALLBACK",
        "FLIP_BASIS_NAMES",
        "TASK1A_CLASS_GROUP_MODE",
        "TASK1A_HIST_TITLE_TEMPLATE",
        "TASK1A_HIST_TITLE_OVERRIDES",
        "task1A_histogram_title_table",
        "TASK2_PAIR_MODE",
        "TASK2_PROJECTION_VECTOR",
        "TASK2_LINE_INDEX_MODE",
        "TASK1B_ANCHOR_SELECTION",
        "period_segment_table",
        "period_summary_table",
        "pair_line_summary_table",
        "cell_table",
    ):
        assert parameter in joined

    assert "plot_basis_check_on_image" in joined
    assert "build_period_histogram_title_table" in joined
    assert "all_analysis_points = analysis_points.copy()" in source_by_id["roi-selection"]
    assert "start_index=1" in source_by_id["roi-selection"]
    assert "class_group_mode=TASK1A_CLASS_GROUP_MODE" in source_by_id["task1a-run"]
    assert "TASK1A_HIST_TITLE_TEMPLATE" in source_by_id["task1a-figure-title-config"]
    assert '"{roi_display_label} {direction} {metric_short}"' in source_by_id["task1a-figure-title-config"]
    assert "task1A_histogram_title_table" in source_by_id["task1a-figures"]
    assert "basis_display_unit='A'" in source_by_id["task1a-figures"]
    assert "will not display px fallback" in source_by_id["roi-selection"]
    assert "line_index_mode=TASK2_LINE_INDEX_MODE" in source_by_id["task2-run"]
    assert "global_line_id" in source_by_id["task2-run"]
    assert "export_task1A_excel" in source_by_id["task1a-export"]
    assert "export_task1B_excel" in source_by_id["task1b-export"]
    assert "export_task2_excel" in source_by_id["task2-export"]
    assert "03_Cropped_group_centroid_analysis.ipynb" in source_by_id["task3-moved-md"]
    assert "TASK3_ROIS" not in joined
    assert "export_task3_excel" not in joined
    assert "export_notebook02_results" in source_by_id["final-export"]
    assert "group_weights" not in joined

    for cell in notebook.get("cells", []):
        assert cell.get("id")
        if cell.get("cell_type") == "code":
            ast.parse(_cell_source(cell), filename=f"02_Simple_quantitative_spacing_analysis.ipynb:{cell['id']}")


def test_build_03_cropped_group_centroid_notebook_generates_compilable_notebook() -> None:
    subprocess.run([sys.executable, "scripts/build_03_cropped_group_centroid_notebook.py"], check=True)
    notebook_path = Path("notebooks") / "03_Cropped_group_centroid_analysis.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    joined = "\n".join(_cell_source(cell) for cell in notebook.get("cells", []))
    source_by_id = {str(cell.get("id", "")): _cell_source(cell) for cell in notebook.get("cells", [])}

    for parameter in (
        "SOURCE_TABLE",
        "USE_KEEP_ONLY",
        "IMAGE_CHANNEL",
        "IMAGE_KEY",
        "OPEN_CROP_ROI_PICKER",
        "CROP_ROI",
        "crop_image_and_points_by_roi",
        "cropped_background_image",
        "OPEN_MEASUREMENT_ROI_PICKER",
        "pick_rois_on_image_with_napari",
        "transform_rois_xy",
        "center_groups",
        "center_pairs",
        "crop_basis_x_px",
        "add_crop_coordinate_columns_to_group_results",
        "plot_cropped_group_centers_and_displacements",
        "SHOW_SCALEBAR",
        "ARROW_COLOR",
        "ARROW_TAIL_WIDTH",
        "DISTANCE_CMAP",
        "export_cropped_group_centroid_excel",
        "export_notebook03_results",
    ):
        assert parameter in joined

    assert "pixel calibration" in source_by_id["load-session"]
    assert "crop-local/global" in source_by_id["run-analysis"] or "crop-local" in source_by_id["run-md"]
    assert "scalebar_length_nm" in source_by_id["figure"]
    assert "show_centers=False" in source_by_id["figure"]
    assert "#d62728" in source_by_id["figure"]
    assert "show_scalebar=SHOW_SCALEBAR" in source_by_id["figure"]
    assert "SHOW_SCALEBAR = False" in source_by_id["figure"]
    assert "arrow_tail_width=ARROW_TAIL_WIDTH" in source_by_id["figure"]
    assert "'magma'" in source_by_id["figure"]
    assert "03_cropped_group_centroid" in joined

    for cell in notebook.get("cells", []):
        assert cell.get("id")
        if cell.get("cell_type") == "code":
            ast.parse(_cell_source(cell), filename=f"03_Cropped_group_centroid_analysis.ipynb:{cell['id']}")
