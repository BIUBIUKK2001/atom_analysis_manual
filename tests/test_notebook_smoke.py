from __future__ import annotations

import ast
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest


def _load_notebook(name: str) -> dict:
    path = Path("notebooks") / name
    if not path.exists():
        pytest.skip(f"{name} is not present in this notebook set.")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_notebook(name: str) -> dict:
    path = Path("notebooks") / name
    if not path.exists():
        pytest.skip(f"{name} is not present in this notebook set.")
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _cell_sources(notebook: dict) -> list[str]:
    return [_cell_source(cell) for cell in notebook.get("cells", [])]


def _markdown_sources(notebook: dict) -> list[str]:
    return [_cell_source(cell) for cell in notebook.get("cells", []) if cell.get("cell_type") == "markdown"]


def _cell_ids(notebook: dict) -> list[str]:
    return [str(cell.get("id", "")) for cell in notebook.get("cells", [])]


def _source_by_id(notebook: dict) -> dict[str, str]:
    return {str(cell.get("id", "")): _cell_source(cell) for cell in notebook.get("cells", [])}


def _assert_parameter_cell_before_stage(cell_ids: list[str], parameter_id: str, stage_id: str) -> None:
    parameter_index = cell_ids.index(parameter_id)
    stage_index = cell_ids.index(stage_id)
    assert 1 <= stage_index - parameter_index <= 2


def test_findatom_notebook_uses_generic_classification_workflow_with_utf8_notes() -> None:
    notebook = _load_notebook("01_Findatom.ipynb")
    joined = "\n".join(_cell_sources(notebook))
    markdown = "\n".join(_markdown_sources(notebook))
    cell_ids = _cell_ids(notebook)
    source_by_id = _source_by_id(notebook)
    logic_joined = "\n".join(
        _cell_source(cell)
        for cell in notebook.get("cells", [])
        if cell.get("id") != "channel-parameters"
    )

    assert "initialize_generic_classification_session" in joined
    assert "run_generic_candidate_detection" in joined
    assert "review_generic_candidates" in joined
    assert "run_generic_refinement" in joined
    assert "launch_refinement_napari_viewer" in joined
    assert "run_atom_column_classification" in joined
    assert "run_generic_curation" in joined
    assert "export_final_atom_table_excel" in joined
    assert "AtomColumnClassificationConfig" in joined
    assert "FEATURES_ENABLED" in joined
    assert "FEATURE_CHANNEL_WEIGHTS" in joined
    assert "CLUSTER_METHOD" in joined
    assert "CONFIDENCE_THRESHOLD" in joined
    assert "CLASS_NAME_MAP" in joined
    assert "自动聚类得到的是图像衬度和局部形貌特征类别，不等价于元素鉴定" in markdown
    assert "参数配置" not in markdown
    assert "HfO2" not in logic_joined
    assert "heavy_atom" not in logic_joined
    assert "light_atom" not in logic_joined
    assert "haadf" not in logic_joined.lower()
    assert "idpc" not in logic_joined.lower()
    assert "abf" not in logic_joined.lower()

    assert "parameters" not in cell_ids
    assert "parameter-notes" not in cell_ids
    expected_order = [
        "imports",
        "channel-parameters",
        "load-session",
        "detect-parameters",
        "detect-stage",
        "candidate-review-parameters",
        "candidate-review-stage",
        "classify-parameters",
        "classify-stage",
        "review-parameters",
        "review-stage",
        "refine-parameters",
        "refine-stage",
        "refinement-review-stage",
        "curate-parameters",
        "curate-stage",
        "checkpoint-parameters",
        "checkpoint-stage",
        "final-excel-export-md",
        "final-excel-export-stage",
    ]
    assert [cell_ids.index(cell_id) for cell_id in expected_order] == sorted(
        cell_ids.index(cell_id) for cell_id in expected_order
    )

    for parameter_id, stage_id in (
        ("channel-parameters", "load-session"),
        ("detect-parameters", "detect-stage"),
        ("candidate-review-parameters", "candidate-review-stage"),
        ("classify-parameters", "classify-stage"),
        ("review-parameters", "review-stage"),
        ("refine-parameters", "refine-stage"),
        ("curate-parameters", "curate-stage"),
        ("checkpoint-parameters", "checkpoint-stage"),
    ):
        _assert_parameter_cell_before_stage(cell_ids, parameter_id, stage_id)

    assert "CHANNELS" in source_by_id["channel-parameters"]
    assert "DETECTION_CONFIGS_BY_CHANNEL" in source_by_id["detect-parameters"]
    assert "OPEN_CANDIDATE_REVIEW_VIEWER" in source_by_id["candidate-review-parameters"]
    assert "CANDIDATE_REVIEW_IMAGE_CHANNEL" in source_by_id["candidate-review-parameters"]
    assert "CANDIDATE_REVIEW_IMAGE_KEY" in source_by_id["candidate-review-parameters"]
    assert "CANDIDATE_REVIEW_POINT_SIZE" in source_by_id["candidate-review-parameters"]
    assert "REFINEMENT_CONFIG" in source_by_id["refine-parameters"]
    assert "CLASS_REFINEMENT_OVERRIDES" in source_by_id["refine-parameters"]
    assert "NN_CONTEXT_MODE" in source_by_id["refine-parameters"]
    for refinement_parameter in (
        "mode",
        "fit_half_window",
        "com_half_window",
        "nn_radius_fraction",
        "min_patch_radius_px",
        "max_patch_radius_px",
        "initial_sigma_px",
        "min_sigma_px",
        "max_sigma_px",
        "max_center_shift_px",
        "max_nfev",
        "gaussian_retry_count",
        "gaussian_retry_shrink_factor",
        "sigma_ratio_limit",
        "fit_edge_margin_px",
        "gaussian_image_source",
        "fallback_to_quadratic",
        "fallback_to_com",
        "quality_floor",
        "overlap_trigger_px",
    ):
        assert refinement_parameter in source_by_id["refine-parameters"]
    assert "硬性位移上限" in source_by_id["refine-parameters"]
    assert "不是硬性位移约束" not in source_by_id["refine-parameters"]
    assert "双高斯共享形状" in source_by_id["refine-parameters"]
    assert "OPEN_REFINEMENT_REVIEW_VIEWER" in source_by_id["refinement-review-stage"]
    assert "SHOW_CANDIDATE_LAYER_IN_REFINEMENT_REVIEW" in source_by_id["refinement-review-stage"]
    assert "REFINEMENT_REVIEW_POINT_SIZE" in source_by_id["refinement-review-stage"]
    assert "FEATURES_ENABLED" in source_by_id["classify-parameters"]
    assert "SOURCE_TABLE_FOR_CLASSIFICATION = 'candidate'" in source_by_id["classify-parameters"]
    assert "fit_amplitude" not in source_by_id["classify-parameters"]
    assert "fit_sigma" not in source_by_id["classify-parameters"]
    assert "CLUSTER_METHOD" in source_by_id["classify-parameters"]
    assert "CLASS_NAME_MAP" in source_by_id["classify-parameters"]
    assert "CLASS_COLOR_MAP" in source_by_id["classify-parameters"]
    assert "SAVE_CLASSIFIED_CHECKPOINT" in source_by_id["classify-parameters"]
    assert "OPEN_CLASS_REVIEW_VIEWER" in source_by_id["review-parameters"]
    assert "CLASS_REVIEW_POINT_SIZE" in source_by_id["review-parameters"]
    assert "source_table='candidate'" in source_by_id["review-stage"]
    assert "class_refinement_overrides=CLASS_REFINEMENT_OVERRIDES" in source_by_id["refine-stage"]
    assert "nn_context_mode=NN_CONTEXT_MODE" in source_by_id["refine-stage"]
    assert "CURATION_CONFIG" in source_by_id["curate-parameters"]
    assert "SAVE_FINAL_CHECKPOINT" in source_by_id["checkpoint-parameters"]
    assert "EXPORT_FINAL_EXCEL" in source_by_id["final-excel-export-stage"]
    assert "FINAL_EXCEL_FILENAME" in source_by_id["final-excel-export-stage"]
    assert "export_final_atom_table_excel" in source_by_id["final-excel-export-stage"]

    assert "OPEN_CLASS_REVIEW_VIEWER" not in source_by_id["classify-parameters"]
    assert "OPEN_DETECTION_OVERVIEW_VIEWER" not in joined
    assert "CURATION_CONFIG" not in source_by_id["review-parameters"]
    assert "SAVE_FINAL_CHECKPOINT" not in source_by_id["classify-parameters"]
    assert "最近邻距离仍使用全部 candidate" not in markdown

    for cell in notebook.get("cells", []):
        assert cell.get("id")
        if cell.get("cell_type") == "code":
            ast.parse(_cell_source(cell), filename=f"01_Findatom.ipynb:{cell['id']}")


def test_export_final_atom_table_excel_writes_xlsx(tmp_path: Path) -> None:
    from em_atom_workbench.notebook_workflows import export_final_atom_table_excel
    from em_atom_workbench.session import AnalysisSession, PixelCalibration

    session = AnalysisSession(name="excel_smoke", pixel_calibration=PixelCalibration(size=0.02, unit="nm"))
    session.curated_points = pd.DataFrame(
        {
            "atom_id": [0, 1],
            "x_px": [10.0, 20.0],
            "y_px": [5.0, 15.0],
            "class_id": [0, 1],
            "class_name": ["class_0", "class_1"],
            "fit_residual": [0.1, 0.2],
            "quality_score": [0.9, 0.8],
            "keep": [True, False],
        }
    )

    result = export_final_atom_table_excel(session, result_root=tmp_path)
    output_path = tmp_path / "01_findatom" / "tables" / "01_final_atom_columns.xlsx"

    assert output_path.exists()
    assert any("Final atom Excel exported" in message for message in result.messages)
    with zipfile.ZipFile(output_path) as archive:
        names = set(archive.namelist())
    assert "xl/workbook.xml" in names
    assert "xl/worksheets/sheet1.xml" in names


def test_notebook_00_splits_mode_and_path_configuration_with_chinese_notes() -> None:
    notebook = _load_optional_notebook("00_environment_and_data_io.ipynb")
    joined = "\n".join(_cell_sources(notebook))
    markdown = "\n".join(_markdown_sources(notebook))

    assert "WORKFLOW_MODE" in joined
    assert "DATA_PATH" in joined
    assert "IDPC_PATH" in joined
    assert "HAADF_PATH" in joined
    assert "这一格只负责选择当前工作流模式" in markdown
    assert "这一格只负责填写单通道原始数据路径" in markdown
    assert "这一格只负责填写双通道原始数据路径" in markdown
    assert "本 cell 跳过" not in joined
    assert "导入执行区" in markdown


def test_notebook_01_uses_staged_multichannel_detection_and_read_only_overview() -> None:
    notebook = _load_optional_notebook("01_preprocess_and_candidate_detection.ipynb")
    joined = "\n".join(_cell_sources(notebook))
    markdown = "\n".join(_markdown_sources(notebook))

    assert "load_or_connect_session" in joined
    assert "_clear_preprocess_results" in joined
    assert "detect_hfo2_heavy_candidates" in joined
    assert "detect_hfo2_light_candidates" in joined
    assert "edit_hfo2_heavy_candidates_with_napari" in joined
    assert "edit_hfo2_light_candidates_with_napari" in joined
    assert "launch_detection_napari_viewer" in joined
    assert "detect_hfo2_multichannel_candidates" not in joined
    assert "preprocess_image" not in joined
    assert "preprocess_channels" not in joined
    assert "summarize_wiener_filter" not in joined
    assert "当前 session 工作流模式不是" not in joined
    assert "本 cell 跳过" not in joined
    assert "这一格只负责填写单通道粗检测参数与执行开关" in markdown
    assert "这一格只负责填写双通道重柱阶段检测参数" in markdown
    assert "这一格只负责填写双通道轻柱阶段检测参数" in markdown
    assert "直接对原始图像" in markdown
    assert "不再做 Wiener 滤波" in markdown
    assert "这一区只提供只读 napari 总览" in markdown
    assert "heavy_reviewed" in joined


def test_notebook_02_mentions_heavy_reviewed_gate_and_keeps_detected_requirement() -> None:
    notebook = _load_optional_notebook("02_refine_and_curate_atom_positions.ipynb")
    joined = "\n".join(_cell_sources(notebook))
    markdown = "\n".join(_markdown_sources(notebook))

    assert "required_stage='detected'" in joined or 'required_stage="detected"' in joined
    assert "_clear_preprocess_results" in joined
    assert "heavy_reviewed" in markdown
    assert "必须回到 `01` 完成轻柱复核" in markdown
    assert "直接对原始输入图像精修" in markdown
    assert "丢弃旧的 preprocess 缓存再精修" in markdown


def test_merged_single_channel_notebook_keeps_linear_raw_direct_flow() -> None:
    notebook = _load_notebook("00_02_single_channel_end_to_end.ipynb")
    joined = "\n".join(_cell_sources(notebook))

    assert "synthetic_lattice_image" in joined
    assert "detect_candidates" in joined
    assert "refine_points" in joined
    assert "curate_points" in joined
    assert "launch_detection_napari_viewer" in joined
    assert "launch_refinement_napari_viewer" in joined
    assert "load_or_connect_session" not in joined
    assert "WORKFLOW_MODE" not in joined
    assert "hfo2_multichannel" not in joined


def test_merged_multichannel_notebook_uses_helper_driven_stages() -> None:
    notebook = _load_notebook("00_02_hfo2_multichannel_end_to_end.ipynb")
    joined = "\n".join(_cell_sources(notebook))
    helper_source = Path("src/em_atom_workbench/notebook_workflows.py").read_text(encoding="utf-8")

    assert "initialize_hfo2_multichannel_session" in joined
    assert "run_hfo2_heavy_detection" in joined
    assert "run_hfo2_light_detection" in joined
    assert "show_hfo2_detection_overview" in joined
    assert "run_hfo2_refine_curate" in joined
    assert "save_final_checkpoint_if_requested" in joined
    assert "def _workflow_channels" not in joined
    assert "load_or_connect_session" not in joined
    assert "WORKFLOW_MODE" not in joined
    assert "detect_hfo2_multichannel_candidates" not in joined

    assert "synthetic_hfo2_multichannel_bundle" in helper_source
    assert "load_image_bundle" in helper_source
    assert "detect_hfo2_heavy_candidates" in helper_source
    assert "detect_hfo2_light_candidates" in helper_source
    assert "edit_hfo2_heavy_candidates_with_napari" in helper_source
    assert "edit_hfo2_light_candidates_with_napari" in helper_source
    assert "launch_detection_napari_viewer" in helper_source
    assert "launch_refinement_napari_viewer" in helper_source
    assert "detect_hfo2_multichannel_candidates" not in helper_source


def test_merged_multichannel_notebook_keeps_code_cells_compact() -> None:
    notebook = _load_notebook("00_02_hfo2_multichannel_end_to_end.ipynb")
    code_line_counts = [
        len([line for line in _cell_source(cell).splitlines() if line.strip()])
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]

    assert max(code_line_counts) <= 35


def test_vpcf_notebook_is_recommended_03_and_loads_curated_session() -> None:
    notebook = _load_notebook("03_vpcf_local_order.ipynb")
    joined = "\n".join(_cell_sources(notebook))
    markdown = "\n".join(_markdown_sources(notebook))

    assert "required_stage='curated'" in joined or 'required_stage="curated"' in joined
    assert "VPCFConfig" in joined
    assert "points_for_vpcf" in joined
    assert "compute_session_vpcf" in joined
    assert "USE_KEEP_POINTS" in joined
    assert "CENTER_ATOM_ID" in joined
    assert "ROI_X_RANGE" in joined
    assert "global_average_H" in joined
    assert "region_average_H" in joined
    assert "save_active_session" in joined
    assert "single_channel" in joined
    assert "LEGACY" not in markdown


def test_legacy_03_notebook_is_marked_as_legacy() -> None:
    notebook = _load_notebook("03_lattice_and_local_metrics.ipynb")
    markdown = "\n".join(_markdown_sources(notebook))

    assert "LEGACY" in markdown
    assert "03_vpcf_local_order.ipynb" in markdown


def test_merged_notebook_code_cells_compile() -> None:
    for name in (
        "00_02_single_channel_end_to_end.ipynb",
        "00_02_hfo2_multichannel_end_to_end.ipynb",
        "03_vpcf_local_order.ipynb",
    ):
        notebook = _load_notebook(name)
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") == "code":
                ast.parse(_cell_source(cell), filename=f"{name}:{index}")


def test_notebook_code_cells_compile_and_have_ids() -> None:
    for name in (
        "00_environment_and_data_io.ipynb",
        "01_preprocess_and_candidate_detection.ipynb",
        "02_refine_and_curate_atom_positions.ipynb",
    ):
        notebook = _load_optional_notebook(name)
        for cell in notebook.get("cells", []):
            assert cell.get("id")
            if cell.get("cell_type") == "code":
                ast.parse(_cell_source(cell), filename=f"{name}:{cell['id']}")


def test_local_affine_strain_notebook_scaffold_is_pixel_first_and_compiles() -> None:
    name = "07_local_affine_strain_from_coordinates.ipynb"
    notebook = _load_notebook(name)
    joined = "\n".join(_cell_sources(notebook))
    markdown = "\n".join(_markdown_sources(notebook))

    required_sections = (
        "# 07 从原子坐标计算局部 affine strain",
        "适用范围和物理含义",
        "0. 用户参数",
        "1. 载入或复用 AnalysisSession",
        "2. 检查 curated/refined atom coordinate table",
        "3. 构建 reference lattice",
        "4. 计算 local affine strain",
        "5. 检查 strain_table 和 QC",
        "6. 初步可视化",
        "7. 导出结果",
        "8. 结果解释注意事项",
    )
    for section in required_sections:
        assert section in markdown

    required_parameters = (
        "SESSION_PATH",
        "RESULT_ROOT",
        "OUTPUT_DIR",
        "COORDINATE_SOURCE",
        "USE_KEEP_POINTS",
        "ATOM_ROLE",
        "REFERENCE_MODE",
        "REFERENCE_COORDINATE_UNIT",
        "REFERENCE_N_CANDIDATES",
        "SELECTED_REFERENCE_CANDIDATE_ID",
        "REFERENCE_ROI",
        "MANUAL_BASIS",
        "MANUAL_ORIGIN",
        "NEIGHBOR_SHELLS",
        "K_NEIGHBORS",
        "MIN_PAIRS",
        "PAIR_ASSIGNMENT_TOLERANCE",
        "MAX_CONDITION_NUMBER",
        "STRAIN_TYPE",
        "OUTPUT_FRAME",
        "WEIGHT_POWER",
        "PLOT_COMPONENTS",
        "QC_ONLY_FOR_PLOTS",
        "EXPORT_RESULTS",
        "OVERWRITE_EXPORT",
        "USE_SYNTHETIC_DEMO_IF_NO_SESSION",
        "SAVE_ACTIVE_SESSION",
        "SAVE_CHECKPOINT",
    )
    for parameter in required_parameters:
        assert parameter in joined

    assert "REFERENCE_MODE = 'suggested_cluster'" in joined
    assert "REFERENCE_COORDINATE_UNIT = 'px'" in joined
    assert "\nREFERENCE_CANDIDATE_ID =" not in joined
    assert joined.index("plot_reference_candidate_map(") < joined.index("SELECTED_REFERENCE_CANDIDATE_ID")
    for qa_metric in ("local_orientation_deg", "basis_a_length_px", "basis_b_length_px", "basis_angle_deg"):
        assert qa_metric in joined
    assert "MANUAL_BASIS = (" in joined
    assert "MANUAL_BASIS_NM" not in joined
    assert "ReferenceLatticeConfig" in joined
    assert "ReferenceLatticeSuggestionConfig" in joined
    assert "LocalAffineStrainConfig" in joined
    assert "ExportConfig" in joined
    assert "build_reference_lattice" in joined
    assert "build_reference_lattice_from_suggestion" in joined
    assert "compute_local_metrics" in joined
    assert "compute_local_affine_strain" in joined
    assert "export_results" in joined
    assert "suggest_reference_lattices" in joined
    assert "session.strain_table.to_csv" not in joined
    assert "qc_flag" in joined
    assert "_fallback_strain_scatter" not in joined
    assert "plot_reference_candidate_map(" in joined
    assert "plot_strain_component_map(session," in joined
    assert "plot_strain_qc_map(session)" in joined
    assert "当前使用 synthetic demo session，仅用于演示 notebook 流程，不代表真实实验数据。" in joined

    for cell in notebook.get("cells", []):
        assert cell.get("id")
        if cell.get("cell_type") == "code":
            ast.parse(_cell_source(cell), filename=f"{name}:{cell['id']}")
