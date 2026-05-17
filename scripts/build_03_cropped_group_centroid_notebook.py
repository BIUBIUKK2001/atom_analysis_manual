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
    plot_cropped_group_centers_and_displacements,
    summarize_rois_and_points,
    transform_rois_xy,
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
            "load-md",
            """
## 0. Load 01 session / image / atom table

这里和 02 一样读取 01 保存的 active session，并选择 `SOURCE_TABLE`、`USE_KEEP_ONLY`、`IMAGE_CHANNEL` 和 `IMAGE_KEY`。03 的正式图必须有真实 nm scalebar，所以没有 pixel calibration 时会直接停止。
""",
        ),
        code(
            "load-session",
            """
RESULT_ROOT = PROJECT_ROOT / 'results'
RESULT_ROOT.mkdir(exist_ok=True)

# SESSION_PATH：
# - None：读取 RESULT_ROOT / '_active_session.pkl'；
# - Path 或字符串：读取你手动指定的 session pickle。
SESSION_PATH = None

# SOURCE_TABLE：选择 01 产生的哪一张原子坐标表作为分析起点。
# curated：正式定量推荐；refined：检查精修点；candidate：通常仅用于诊断。
SOURCE_TABLE = 'curated'
USE_KEEP_ONLY = True
IMAGE_CHANNEL = None
IMAGE_KEY = 'raw'

context = initialize_simple_quant_v2_analysis(
    session_path=SESSION_PATH,
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

output_dirs = cropped_group_centroid_output_dirs(RESULT_ROOT)
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

每个 group 可以包含一个或多个 `class_id`。几何中心是 ROI 内该 group 所有原子的无权重平均中心。`center_pairs` 的方向约定是 `group_A -> group_B`。
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

group_config_table = pd.DataFrame([
    {'group_name': key, 'class_ids': ','.join(str(value) for value in values)}
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

最终图只显示裁剪后的背景图、`group_A -> group_B` 填充箭头和按距离插值得到的平滑着色层；不显示 measurement ROI、不显示几何中心点、不显示 scalebar。箭头起止位置严格等于计算得到的两个几何中心。
""",
        ),
        code(
            "figure",
            """
SHOW_SCALEBAR = False          # 03 默认不在图中显示 scalebar。
SCALEBAR_LENGTH_NM = None      # 若 SHOW_SCALEBAR=True，可设置具体 nm 长度；None 自动估计。
SCALEBAR_COLOR = 'black'
SCALEBAR_LINEWIDTH = 1.2
SCALEBAR_LOCATION = 'lower right'
# 箭头颜色可用任意 matplotlib 颜色名或 hex。
# 常用候选：'black'/'#111111'、'white'/'#ffffff'、'red'/'#d62728'、'blue'/'#1f77b4'、
#          'orange'/'#ff7f0e'、'yellow'/'#ffe600'、'purple'/'#9467bd'、'cyan'/'#17becf'。
ARROW_COLOR = '#ffe600'
ARROW_EDGE_COLOR = '#111111'   # 不需要边线时设为 None。
ARROW_LINEWIDTH = 0.25
ARROW_MUTATION_SCALE = 7.0
ARROW_TAIL_WIDTH = 0.65
ARROW_HEAD_WIDTH = 4.0
ARROW_HEAD_LENGTH = 4.8
ARROW_ALPHA = 0.95
# 距离插值色图常用候选：'magma'、'inferno'、'viridis'、'cividis'、'plasma'、'turbo'。
DISTANCE_CMAP = 'magma'
DISTANCE_ALPHA = 0.42
INTERPOLATION_SIGMA_PX = None  # None 表示根据裁剪图尺寸自动估计。
SHOW_DISTANCE_COLORBAR = True

fig, ax = plot_cropped_group_centers_and_displacements(
    cropped_background_image,
    group_centroid_table,
    group_displacement_table,
    pixel_to_nm=PIXEL_TO_NM,
    style=FIG_STYLE,
    title='Cropped group-center displacement',
    show_scalebar=SHOW_SCALEBAR,
    scalebar_length_nm=SCALEBAR_LENGTH_NM,
    scalebar_color=SCALEBAR_COLOR,
    scalebar_linewidth=SCALEBAR_LINEWIDTH,
    scalebar_location=SCALEBAR_LOCATION,
    show_centers=False,
    show_center_legend=False,
    arrow_color=ARROW_COLOR,
    arrow_edge_color=ARROW_EDGE_COLOR,
    arrow_linewidth=ARROW_LINEWIDTH,
    arrow_mutation_scale=ARROW_MUTATION_SCALE,
    arrow_tail_width=ARROW_TAIL_WIDTH,
    arrow_head_width=ARROW_HEAD_WIDTH,
    arrow_head_length=ARROW_HEAD_LENGTH,
    arrow_alpha=ARROW_ALPHA,
    distance_cmap=DISTANCE_CMAP,
    distance_alpha=DISTANCE_ALPHA,
    interpolation_sigma_px=INTERPOLATION_SIGMA_PX,
    show_distance_colorbar=SHOW_DISTANCE_COLORBAR,
)
figures['cropped_group_center_displacement'] = fig
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
    'notebook03_cropped_group_centroid_config': {
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
        'center_pairs': center_pairs,
        'min_points_per_group': MIN_POINTS_PER_GROUP,
        'show_scalebar': SHOW_SCALEBAR,
        'scalebar_color': SCALEBAR_COLOR,
        'scalebar_linewidth': SCALEBAR_LINEWIDTH,
        'scalebar_location': SCALEBAR_LOCATION,
        'arrow_color': ARROW_COLOR,
        'arrow_edge_color': ARROW_EDGE_COLOR,
        'arrow_linewidth': ARROW_LINEWIDTH,
        'arrow_mutation_scale': ARROW_MUTATION_SCALE,
        'arrow_tail_width': ARROW_TAIL_WIDTH,
        'arrow_head_width': ARROW_HEAD_WIDTH,
        'arrow_head_length': ARROW_HEAD_LENGTH,
        'arrow_alpha': ARROW_ALPHA,
        'distance_cmap': DISTANCE_CMAP,
        'distance_alpha': DISTANCE_ALPHA,
        'interpolation_sigma_px': INTERPOLATION_SIGMA_PX,
    }
}

manifest = export_notebook03_results(
    session=session,
    output_dirs=output_dirs,
    tables=notebook03_tables,
    figures=figures,
    configs=notebook03_configs,
    excel_exports=excel_exports,
    figure_formats=FIG_STYLE.normalized_formats(),
    figure_dpi=FIG_STYLE.fig_dpi,
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
