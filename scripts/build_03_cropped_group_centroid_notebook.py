# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path


def _lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def md(cell_id: str, text: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": _lines(text)}


def code(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def build_notebook() -> dict:
    cells = [
        md(
            "title",
            """
# 03 Cropped group-centroid analysis

这个 notebook 承接 `01_Findatom.ipynb` 保存的 active session，专门执行裁剪区域内的 class group 几何中心和 group-pair 位移分析。

流程是：读取 01 的原子位置和分类；在原图上选择一个 crop ROI；按 crop ROI 外接矩形裁剪图像，并只保留 ROI 内原子；在裁剪图上选择 measurement ROI；计算每个 ROI 内各 class group 的无权重几何中心、间距和角度；最后输出只包含几何中心、箭头、group legend 和真实 nm scalebar 的正式图。
""",
        ),
        code(
            "imports",
            """
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd()
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / 'src' / 'em_atom_workbench').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
PROJECT_SRC = str(PROJECT_ROOT / 'src')
if PROJECT_SRC in sys.path:
    sys.path.remove(PROJECT_SRC)
sys.path.insert(0, PROJECT_SRC)
for module_name in list(sys.modules):
    if module_name == 'em_atom_workbench' or module_name.startswith('em_atom_workbench.'):
        del sys.modules[module_name]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from em_atom_workbench import (
    AnalysisROI,
    FigureStyleConfig,
    add_crop_coordinate_columns_to_group_results,
    assign_points_to_rois,
    compute_group_centroids_by_roi,
    compute_group_pair_displacements,
    crop_image_and_points_by_roi,
    full_image_roi,
    pick_rois_on_image_with_napari,
    pick_rois_with_napari,
    plot_charge_center_displacement_map,
    summarize_rois_and_points,
    transform_rois_xy,
    initialize_analysis_workspace,
)
from em_atom_workbench.notebook_workflows import (
    cropped_group_centroid_output_dirs,
    export_cropped_group_centroid_excel,
    export_notebook03_results,
    initialize_simple_quant_v2_analysis,
)

FIG_STYLE = FigureStyleConfig()
plt.rcParams['figure.dpi'] = 140
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def rois_to_table(local_rois, global_rois=None):
    global_rois = global_rois or local_rois
    rows = []
    for local_roi, global_roi in zip(local_rois, global_rois, strict=False):
        rows.append({
            'roi_id': local_roi.roi_id,
            'roi_name': local_roi.roi_name or local_roi.roi_id,
            'roi_color': local_roi.color,
            'polygon_local_xy_px': local_roi.polygon_xy_px,
            'polygon_global_xy_px': global_roi.polygon_xy_px,
            'enabled': bool(local_roi.enabled),
        })
    return pd.DataFrame(rows)
""",
        ),
        md(
            "workspace-md",
            """
## 0. Workspace parameters

这一格只定义当前 dataset/run 的统一 workspace。03 默认从 `workspace/state/sessions/01_final_curated.pkl` 读取 01 的最终 curated session，并把所有 03 输出写入 canonical `03_group_centroid/`。
""",
        ),
        code(
            "workspace-parameters",
            """
OUTPUT_ROOT = PROJECT_ROOT / 'results'
DATASET_ID = 'dataset_001'
ANALYSIS_ID = 'run_001'

workspace = initialize_analysis_workspace(
    output_root=OUTPUT_ROOT,
    dataset_id=DATASET_ID,
    analysis_id=ANALYSIS_ID,
)

# 为兼容旧 wrapper，RESULT_ROOT 仍然存在；新流程中它指向当前 workspace.root。
RESULT_ROOT = workspace.root

# SESSION_SOURCE='01_final_curated' 是推荐默认入口：
# workspace/state/sessions/01_final_curated.pkl
SESSION_SOURCE = '01_final_curated'

# SESSION_PATH=None 时读取 SESSION_SOURCE。
# 如需读取旧 results/_active_session.pkl 或手动 checkpoint，可在这里填 pickle 路径。
SESSION_PATH = None

print(f'workspace: {workspace.root}')
print(f'default session: {workspace.sessions_dir / (SESSION_SOURCE + ".pkl")}')
""",
        ),
        md(
            "load-md",
            """
## 1. Load 01 session / image / atom table

这里和 02 一样读取 01 保存的 `01_final_curated` stage session，并选择 `SOURCE_TABLE`、`USE_KEEP_ONLY`、`IMAGE_CHANNEL` 和 `IMAGE_KEY`。03 的正式图必须有真实 nm scalebar，所以没有 pixel calibration 时会直接停止。
""",
        ),
        code(
            "load-session",
            """
# SOURCE_TABLE：选择 01 产生的哪一张原子坐标表作为分析起点。
# curated：正式定量推荐；refined：检查精修点；candidate：通常仅用于诊断。
SOURCE_TABLE = 'curated'
USE_KEEP_ONLY = True
IMAGE_CHANNEL = None
IMAGE_KEY = 'raw'

context = initialize_simple_quant_v2_analysis(
    session_path=SESSION_PATH,
    workspace=workspace,
    session_source=SESSION_SOURCE,
    required_stage=None,
    result_root=RESULT_ROOT,
    source_table=SOURCE_TABLE,
    use_keep_only=USE_KEEP_ONLY,
    class_filter=None,
    class_id_filter=None,
    rois=None,
    image_channel=IMAGE_CHANNEL,
    image_key=IMAGE_KEY,
)
session = context['session']
image = context['image']
image_channel = context['image_channel']
analysis_points_global = context['analysis_points']
PIXEL_TO_NM = analysis_points_global.attrs.get('pixel_to_nm')
if PIXEL_TO_NM is None or not np.isfinite(float(PIXEL_TO_NM)) or float(PIXEL_TO_NM) <= 0:
    raise ValueError('03 requires a valid pixel calibration from 01 so the final figure can include a true nm scalebar.')

output_dirs = cropped_group_centroid_output_dirs(workspace=workspace)
preview_figures = {}
figures = {}
excel_exports = {}

display(context['summary_tables']['simple_quant_v2_summary'])
display(context['summary_tables']['analysis_points_preview'])
display(summarize_rois_and_points(analysis_points_global))
""",
        ),
        md(
            "crop-md",
            """
## 1. Crop ROI on the original image

打开 napari 后在原图上画 crop ROI。若 ROI 是四角矩形，即使它有旋转，03 也会严格按矩形形状采样裁剪，并把长边旋转到水平方向；其他 polygon ROI 会保留 polygon 内原子并按外接矩形裁剪。后续计算和展示使用裁剪后的局部坐标，并在导出表中保留 global 坐标。
""",
        ),
        code(
            "crop-selection",
            """
OPEN_CROP_ROI_PICKER = True

# 手动模式示例：
# CROP_ROI = AnalysisROI(
#     roi_id='crop_1',
#     roi_name='crop_1',
#     polygon_xy_px=((100, 100), (300, 100), (300, 260), (100, 260)),
# )
CROP_ROI = None
CROP_ROI_PREFIX = 'Crop'
CROP_MARGIN_PX = 0

if OPEN_CROP_ROI_PICKER:
    picked_crop_rois = pick_rois_with_napari(
        session,
        analysis_points_global,
        image_channel=image_channel,
        image_key=IMAGE_KEY,
        default_roi_prefix=CROP_ROI_PREFIX,
        start_index=1,
    )
    CROP_ROI = picked_crop_rois[0]
elif CROP_ROI is None:
    CROP_ROI = full_image_roi(image, roi_id='crop_1', roi_name='crop_full')

crop_result = crop_image_and_points_by_roi(
    image,
    analysis_points_global,
    CROP_ROI,
    margin_px=CROP_MARGIN_PX,
    crop_id='crop_1',
    crop_name=CROP_ROI.roi_name or CROP_ROI.roi_id,
)
cropped_image = crop_result['image']
cropped_background_image = cropped_image.copy() if hasattr(cropped_image, 'copy') else np.array(cropped_image, copy=True)
cropped_atom_table = crop_result['points']
crop_table = crop_result['crop_table']
crop_origin_xy_px = crop_result['origin_xy_px']
crop_basis_x_px = crop_result['basis_x_px']
crop_basis_y_px = crop_result['basis_y_px']

display(crop_table)
display(cropped_atom_table.head())
display(summarize_rois_and_points(cropped_atom_table))
""",
        ),
        md(
            "measurement-roi-md",
            """
## 2. Measurement ROIs on the cropped image

这里在裁剪图上选择用于几何中心计算的 ROI。坐标是 crop-local px；导出时会同时保存这些 ROI 的 local/global polygon。若没有选择 ROI，则默认使用整个裁剪图。
""",
        ),
        code(
            "measurement-roi-selection",
            """
OPEN_MEASUREMENT_ROI_PICKER = True
MEASUREMENT_ROIS = None
MEASUREMENT_ROI_PREFIX = 'Measure_ROI'

if OPEN_MEASUREMENT_ROI_PICKER:
    MEASUREMENT_ROIS = pick_rois_on_image_with_napari(
        cropped_background_image,
        cropped_atom_table,
        default_roi_prefix=MEASUREMENT_ROI_PREFIX,
        start_index=1,
        title='03 measurement ROI picker',
    )
if not MEASUREMENT_ROIS:
    MEASUREMENT_ROIS = [full_image_roi(cropped_background_image, roi_id='crop_full', roi_name='crop_full')]

# 重新分配 measurement ROI 前，移除 crop/global ROI 标记，避免 scope_id 沿用上一步。
measurement_source_points = cropped_atom_table.drop(columns=['roi_id', 'roi_name', 'roi_color', 'scope_id'], errors='ignore')
measurement_points = assign_points_to_rois(measurement_source_points, MEASUREMENT_ROIS)
measurement_points.attrs['pixel_to_nm'] = PIXEL_TO_NM
measurement_rois_global = transform_rois_xy(
    MEASUREMENT_ROIS,
    origin_xy_px=crop_origin_xy_px,
    basis_x_px=crop_basis_x_px,
    basis_y_px=crop_basis_y_px,
)
measurement_roi_table = rois_to_table(MEASUREMENT_ROIS, measurement_rois_global)

display(measurement_roi_table)
display(summarize_rois_and_points(measurement_points, rois=MEASUREMENT_ROIS))
""",
        ),
        md(
            "group-config-md",
            """
## 3. Configure class groups and center pairs

每个 group 可以包含一个或多个 `class_id`。默认几何中心是 ROI 内该 group 所有原子的无权重平均中心；如需加权，可设置每个 group 的 class 权重或权重列。`center_pairs` 的方向约定是 `group_A -> group_B`。
""",
        ),
        code(
            "group-config",
            """
center_groups = {
    # 'group_A': [0],
    # 'group_B': [1, 2],
}

center_pairs = [
    # ('group_A', 'group_B'),
]

MIN_POINTS_PER_GROUP = 1

# 可选：按 class_id 加权。未列出的 class_id 权重默认为 1.0。
# 写法 1：dict，适合 group 内 class_id 权重不连续或需要显式标注。
# 写法 2：list/tuple，顺序必须和 center_groups[group_name] 一致。
CENTER_GROUP_CLASS_WEIGHTS = {
    # 'group_A': {0: 1.0},
    # 'group_B': {1: 1.0, 2: 2.0},
    # 'group_B': [1.0, 2.0],
}

# 可选：按表格中的数值列加权，例如 'quality_score'。
# 可以写成全局字符串，也可以按 group 分别指定列名。
CENTER_GROUP_WEIGHT_COLUMNS = {
    # 'group_A': 'quality_score',
    # 'group_B': 'quality_score',
}

group_config_table = pd.DataFrame([
    {
        'group_name': key,
        'class_ids': ','.join(str(value) for value in values),
        'class_weights': CENTER_GROUP_CLASS_WEIGHTS.get(key, ''),
        'weight_column': CENTER_GROUP_WEIGHT_COLUMNS.get(key, '') if isinstance(CENTER_GROUP_WEIGHT_COLUMNS, dict) else CENTER_GROUP_WEIGHT_COLUMNS,
    }
    for key, values in center_groups.items()
])
display(group_config_table)
""",
        ),
        md(
            "run-md",
            """
## 4. Compute group centroids / displacements

这个 cell 复用 02 的几何中心逻辑，但结果表额外加入 crop-local/global px 坐标和 nm 坐标，以及 `dx_nm`、`dy_nm`、`distance_nm`。
""",
        ),
        code(
            "run-analysis",
            """
group_centroid_table_raw = compute_group_centroids_by_roi(
    measurement_points,
    center_groups=center_groups,
    center_group_class_weights=CENTER_GROUP_CLASS_WEIGHTS,
    center_group_weight_columns=CENTER_GROUP_WEIGHT_COLUMNS,
    min_points=MIN_POINTS_PER_GROUP,
)
group_displacement_table_raw = compute_group_pair_displacements(
    group_centroid_table_raw,
    center_pairs=center_pairs,
    pixel_to_nm=PIXEL_TO_NM,
)
group_centroid_table, group_displacement_table = add_crop_coordinate_columns_to_group_results(
    group_centroid_table_raw,
    group_displacement_table_raw,
    crop_origin_xy_px=crop_origin_xy_px,
    crop_basis_x_px=crop_basis_x_px,
    crop_basis_y_px=crop_basis_y_px,
    pixel_to_nm=PIXEL_TO_NM,
)

summary = pd.DataFrame([
    {'metric': 'crop_point_count', 'value': len(cropped_atom_table)},
    {'metric': 'measurement_roi_count', 'value': len(MEASUREMENT_ROIS)},
    {'metric': 'group_count', 'value': len(center_groups)},
    {'metric': 'center_pair_count', 'value': len(center_pairs)},
    {'metric': 'pixel_to_nm', 'value': PIXEL_TO_NM},
])

display(group_centroid_table)
display(group_displacement_table)
display(summary)
""",
        ),
        md(
            "figure-md",
            """
## 5. Plot cropped displacement arrows

最终图输出四张裁剪区域图：未叠加原图、纯箭头、纯位移大小色图、箭头+色图。色图按每个 measurement ROI / 原胞的 `group_A -> group_B` 位移大小着色；箭头以 A/B 中点为锚点，方向沿 `group_A -> group_B`，视觉长度可自动放大。
""",
        ),
        code(
            "figure",
            """
SHOW_SCALEBAR = False          # 03 默认不在图中显示 scalebar。
SCALEBAR_LENGTH_NM = None      # 若 SHOW_SCALEBAR=True，可设置具体 nm 长度；None 自动估计。
SCALEBAR_COLOR = 'white'
SCALEBAR_LINEWIDTH = 1.2
SCALEBAR_LOCATION = 'lower right'

BACKGROUND_CONTRAST_PERCENTILES = (1.0, 99.0)
VECTOR_VALUE_COLUMN = 'distance_A'  # None 时自动优先使用 distance_nm，其次 distance_A / distance_px。
VECTOR_CMAP = 'magma'
VECTOR_ALPHA = 0.46
SHOW_VECTOR_COLORBAR = True
SHOW_ROI_OUTLINES = True
ROI_OUTLINE_COLOR = '#f5f5f5'
ROI_OUTLINE_LINEWIDTH = 0.65
ROI_OUTLINE_ALPHA = 0.72

ARROW_COLOR = '#ffffff'
ARROW_EDGE_COLOR = '#111111'   # 不需要边线时设为 None。
ARROW_LINEWIDTH = 0.32
ARROW_MUTATION_SCALE = 1.0
ARROW_TAIL_WIDTH = 0.34
ARROW_HEAD_WIDTH = 1.7
ARROW_HEAD_LENGTH = 2.0
ARROW_ALPHA = 0.96
VECTOR_SCALE = 'auto'          # 'auto' 会放大小位移，数值则直接作为倍率。
VECTOR_AUTO_TARGET_FRACTION = 0.045
VECTOR_AUTO_MAX_SCALE = 30.0
SHOW_VECTOR_SCALE_LABEL = True

_charge_plot_common = dict(
    image=cropped_background_image,
    measurement_rois=MEASUREMENT_ROIS,
    group_displacement_table=group_displacement_table,
    pixel_to_nm=PIXEL_TO_NM,
    coordinate_space='local',
    style=FIG_STYLE,
    contrast_percentiles=BACKGROUND_CONTRAST_PERCENTILES,
    value_column=VECTOR_VALUE_COLUMN,
    magnitude_cmap=VECTOR_CMAP,
    magnitude_alpha=VECTOR_ALPHA,
    roi_outline_color=ROI_OUTLINE_COLOR,
    roi_outline_linewidth=ROI_OUTLINE_LINEWIDTH,
    roi_outline_alpha=ROI_OUTLINE_ALPHA,
    arrow_color=ARROW_COLOR,
    arrow_edge_color=ARROW_EDGE_COLOR,
    arrow_linewidth=ARROW_LINEWIDTH,
    arrow_mutation_scale=ARROW_MUTATION_SCALE,
    arrow_tail_width=ARROW_TAIL_WIDTH,
    arrow_head_width=ARROW_HEAD_WIDTH,
    arrow_head_length=ARROW_HEAD_LENGTH,
    arrow_alpha=ARROW_ALPHA,
    vector_scale=VECTOR_SCALE,
    vector_auto_target_fraction=VECTOR_AUTO_TARGET_FRACTION,
    vector_auto_max_scale=VECTOR_AUTO_MAX_SCALE,
    show_scalebar=SHOW_SCALEBAR,
    scalebar_length_nm=SCALEBAR_LENGTH_NM,
    scalebar_color=SCALEBAR_COLOR,
    scalebar_linewidth=SCALEBAR_LINEWIDTH,
    scalebar_location=SCALEBAR_LOCATION,
)

fig, ax = plot_charge_center_displacement_map(
    mode='raw',
    title='Cropped raw image',
    show_colorbar=False,
    show_roi_outlines=False,
    show_vector_scale_label=False,
    **_charge_plot_common,
)
figures['cropped_raw_image'] = fig
display(fig)
plt.close(fig)

fig, ax = plot_charge_center_displacement_map(
    mode='arrows',
    title='Charge-center displacement arrows',
    show_colorbar=False,
    show_roi_outlines=SHOW_ROI_OUTLINES,
    show_vector_scale_label=SHOW_VECTOR_SCALE_LABEL,
    **_charge_plot_common,
)
figures['charge_displacement_arrows'] = fig
display(fig)
plt.close(fig)

fig, ax = plot_charge_center_displacement_map(
    mode='magnitude',
    title='Charge-center displacement magnitude',
    show_colorbar=SHOW_VECTOR_COLORBAR,
    show_roi_outlines=SHOW_ROI_OUTLINES,
    show_vector_scale_label=False,
    **_charge_plot_common,
)
figures['charge_displacement_magnitude'] = fig
display(fig)
plt.close(fig)

fig, ax = plot_charge_center_displacement_map(
    mode='combined',
    title='Charge-center displacement arrows + magnitude',
    show_colorbar=SHOW_VECTOR_COLORBAR,
    show_roi_outlines=SHOW_ROI_OUTLINES,
    show_vector_scale_label=SHOW_VECTOR_SCALE_LABEL,
    **_charge_plot_common,
)
figures['charge_displacement_arrows_magnitude'] = fig
display(fig)
plt.close(fig)
""",
        ),
        md(
            "excel-md",
            """
## 6. Export Excel

导出裁剪 ROI、裁剪后原子表、measurement ROIs、group centroid、group displacement、group config 和 summary。
""",
        ),
        code(
            "excel-export",
            """
cropped_group_excel = export_cropped_group_centroid_excel(
    output_dirs,
    crop_table=crop_table,
    cropped_atom_table=cropped_atom_table,
    measurement_roi_table=measurement_roi_table,
    group_centroid_table=group_centroid_table,
    group_displacement_table=group_displacement_table,
    group_config=group_config_table,
    summary=summary,
)
excel_exports['cropped_group_centroid'] = cropped_group_excel
display(cropped_group_excel)
""",
        ),
        md(
            "final-export-md",
            """
## 7. Final export manifest

统一保存 CSV 表、正式 figures、config JSON、Excel 记录、session checkpoint 和 manifest。
""",
        ),
        code(
            "final-figure-export-parameters",
            """
# =========================
# Final figure export parameters
# =========================
# preview figures 只用于调参检查；默认不保存；若启用则写入 03_group_centroid/figures_preview。
SAVE_PREVIEW_FIGURES = False

# final figures 是正式导出的论文/报告候选图，保存到 03_group_centroid/figures_final。
SAVE_FINAL_FIGURES = True

FIGURE_FORMATS = FIG_STYLE.normalized_formats() if hasattr(FIG_STYLE, 'normalized_formats') else ('pdf', 'png', 'svg')
FIGURE_DPI = FIG_STYLE.fig_dpi
FIGURE_FONT_FAMILY = 'Arial'
FIGURE_FONT_SIZE = 9
FIG_SHOW_TITLE = True
FIG_SHOW_SCALEBAR = SHOW_SCALEBAR

FINAL_FIGURE_SPECS = {
    'cropped_raw_image': {
        'save': True,
        'title': 'Cropped raw image',
        'show_title': FIG_SHOW_TITLE,
        'font_family': FIGURE_FONT_FAMILY,
        'font_size': FIGURE_FONT_SIZE,
        'formats': FIGURE_FORMATS,
        'dpi': FIGURE_DPI,
    },
    'charge_displacement_arrows': {
        'save': True,
        'title': 'Charge-center displacement arrows',
        'show_title': FIG_SHOW_TITLE,
        'font_family': FIGURE_FONT_FAMILY,
        'font_size': FIGURE_FONT_SIZE,
        'formats': FIGURE_FORMATS,
        'dpi': FIGURE_DPI,
    },
    'charge_displacement_magnitude': {
        'save': True,
        'title': 'Charge-center displacement magnitude',
        'show_title': FIG_SHOW_TITLE,
        'font_family': FIGURE_FONT_FAMILY,
        'font_size': FIGURE_FONT_SIZE,
        'formats': FIGURE_FORMATS,
        'dpi': FIGURE_DPI,
    },
    'charge_displacement_arrows_magnitude': {
        'save': True,
        'title': 'Charge-center displacement arrows + magnitude',
        'show_title': FIG_SHOW_TITLE,
        'font_family': FIGURE_FONT_FAMILY,
        'font_size': FIGURE_FONT_SIZE,
        'formats': FIGURE_FORMATS,
        'dpi': FIGURE_DPI,
    },
}
""",
        ),
        code(
            "final-export",
            """
notebook03_tables = {
    'crop_roi': crop_table,
    'cropped_atoms': cropped_atom_table,
    'measurement_rois': measurement_roi_table,
    'group_centroids': group_centroid_table,
    'group_displacements': group_displacement_table,
    'group_config': group_config_table,
    'summary': summary,
}

notebook03_configs = {
    'notebook03_group_centroid_config': {
        'dataset_id': DATASET_ID,
        'analysis_id': ANALYSIS_ID,
        'session_source': SESSION_SOURCE,
        'source_table': SOURCE_TABLE,
        'use_keep_only': USE_KEEP_ONLY,
        'image_channel': image_channel,
        'image_key': IMAGE_KEY,
        'pixel_to_nm': PIXEL_TO_NM,
        'crop_margin_px': CROP_MARGIN_PX,
        'crop_basis_x_px': crop_basis_x_px,
        'crop_basis_y_px': crop_basis_y_px,
        'crop_roi': crop_table.to_dict(orient='records'),
        'measurement_rois': measurement_roi_table.to_dict(orient='records'),
        'center_groups': center_groups,
        'center_group_class_weights': CENTER_GROUP_CLASS_WEIGHTS,
        'center_group_weight_columns': CENTER_GROUP_WEIGHT_COLUMNS,
        'center_pairs': center_pairs,
        'min_points_per_group': MIN_POINTS_PER_GROUP,
        'show_scalebar': SHOW_SCALEBAR,
        'scalebar_color': SCALEBAR_COLOR,
        'scalebar_linewidth': SCALEBAR_LINEWIDTH,
        'scalebar_location': SCALEBAR_LOCATION,
        'background_contrast_percentiles': BACKGROUND_CONTRAST_PERCENTILES,
        'vector_value_column': VECTOR_VALUE_COLUMN,
        'vector_cmap': VECTOR_CMAP,
        'vector_alpha': VECTOR_ALPHA,
        'show_vector_colorbar': SHOW_VECTOR_COLORBAR,
        'show_roi_outlines': SHOW_ROI_OUTLINES,
        'roi_outline_color': ROI_OUTLINE_COLOR,
        'roi_outline_linewidth': ROI_OUTLINE_LINEWIDTH,
        'roi_outline_alpha': ROI_OUTLINE_ALPHA,
        'arrow_color': ARROW_COLOR,
        'arrow_edge_color': ARROW_EDGE_COLOR,
        'arrow_linewidth': ARROW_LINEWIDTH,
        'arrow_mutation_scale': ARROW_MUTATION_SCALE,
        'arrow_tail_width': ARROW_TAIL_WIDTH,
        'arrow_head_width': ARROW_HEAD_WIDTH,
        'arrow_head_length': ARROW_HEAD_LENGTH,
        'arrow_alpha': ARROW_ALPHA,
        'vector_scale': VECTOR_SCALE,
        'vector_auto_target_fraction': VECTOR_AUTO_TARGET_FRACTION,
        'vector_auto_max_scale': VECTOR_AUTO_MAX_SCALE,
        'show_vector_scale_label': SHOW_VECTOR_SCALE_LABEL,
    }
}

manifest = export_notebook03_results(
    session=session,
    output_dirs=output_dirs,
    tables=notebook03_tables,
    figures=figures,
    preview_figures=preview_figures,
    configs=notebook03_configs,
    excel_exports=excel_exports,
    workspace=workspace,
    session_source=SESSION_SOURCE,
    save_preview_figures=SAVE_PREVIEW_FIGURES,
    save_final_figures=SAVE_FINAL_FIGURES,
    final_figure_specs=FINAL_FIGURE_SPECS,
    figure_formats=FIGURE_FORMATS,
    figure_dpi=FIGURE_DPI,
)
display(manifest)
""",
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "notebooks" / "03_Cropped_group_centroid_analysis.ipynb"
    target.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
