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
# 04 Disk-integrated intensity mapping

# 04 原子柱圆盘积分强度 mapping

这个 notebook 承接 `01_Findatom.ipynb` 保存的原子柱定位、分类、精修和 curation 结果，使用固定半径圆盘对每个原子柱附近像素强度求和，并输出 intensity mapping 和 histogram。

边界说明：这里不做自动 vacancy 判定，不设置 vacancy threshold，不做背景扣除、annulus background、Gaussian weighting/fitting、heatmap interpolation、intensity ratio map 或 composition inference。
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
    DiskIntensityConfig,
    compute_disk_intensity_table,
    initialize_analysis_workspace,
    load_active_workspace_session,
    load_stage_session,
    plot_disk_aperture_preview,
    plot_disk_intensity_histogram,
    plot_disk_intensity_map,
    summarize_disk_intensity,
)
from em_atom_workbench.notebook_workflows import (
    export_disk_intensity_analysis,
    initialize_disk_intensity_analysis,
)

plt.rcParams['figure.dpi'] = 140
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
""",
        ),
        md(
            "workspace-md",
            """
## 0. Workspace parameters

这一格只定义当前 dataset/run 的统一 workspace。04 默认从 `workspace/state/sessions/01_final_curated.pkl` 读取 01 的最终 curated session，并把所有 04 输出写入 canonical `04_intensity_mapping/`。
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

RESULT_ROOT = workspace.root

print(f'workspace: {workspace.root}')
print(f'04 output: {workspace.root / "04_intensity_mapping"}')
""",
        ),
        md(
            "session-input-md",
            """
## 1. Session input parameters

读取优先级固定为：`SESSION_PATH > USE_ACTIVE_SESSION > SESSION_SOURCE`。
""",
        ),
        code(
            "session-input-parameters",
            """
# SESSION_SOURCE：
# 默认读取 workspace/state/sessions/01_final_curated.pkl。
# 这是 Notebook01 完成最终 curation 后保存的推荐入口。
SESSION_SOURCE = '01_final_curated'

# SESSION_PATH：
# 如果你要读取某个手动保存的 session pickle，可以在这里填路径。
# 只要 SESSION_PATH 非 None，就会优先使用它。
SESSION_PATH = None

# USE_ACTIVE_SESSION：
# True 时读取 workspace/state/active_session.pkl。
# 这适合继续当前 workspace 中最新状态，但正式复现建议使用明确 stage session。
USE_ACTIVE_SESSION = False

print('session priority: SESSION_PATH > USE_ACTIVE_SESSION > SESSION_SOURCE')
print(f'default stage session: {workspace.sessions_dir / (SESSION_SOURCE + ".pkl")}')
print(f'active session: {workspace.state_dir / "active_session.pkl"}')
""",
        ),
        md(
            "coordinate-source-md",
            """
## 2. Coordinate source

这里选择用于 disk intensity integration 的坐标来源。默认使用 Gaussian-refined coordinates。
""",
        ),
        code(
            "coordinate-source-parameters",
            """
# COORDINATE_SOURCE：
# 选择用于 disk intensity integration 的原子坐标来源。
#
# - "candidate"：
# 使用 01 中粗检 + napari 人工复核后的 candidate_points。
# 适合检查低强度/疑似空位柱，因为 Gaussian fitting 可能在弱柱或异常柱上发生偏移。
#
# - "refined"：
# 使用 Gaussian fitting 后的 refined_points。
# 适合正常原子柱的亚像素精确强度积分，是多数正式强度 mapping 的推荐选择。
#
# - "curated"：
# 使用最终 curated_points，通常是 refined 后再经过 keep/filter/quality 筛选的结果。
# 适合只分析最终保留下来的点。
#
# 对于疑似 vacancy / low-intensity columns：
# 建议分别运行 candidate 和 refined 两套坐标结果进行对比。
# 如果低强度点在 refined 坐标下明显偏移或消失，说明 Gaussian fitting 可能受弱强度/邻近柱影响。
COORDINATE_SOURCE = 'refined'  # 'candidate', 'refined', or 'curated'

# USE_KEEP_ONLY：只在所选坐标表存在 keep 列时应用；没有 keep 列会自动跳过。
USE_KEEP_ONLY = True
""",
        ),
        md(
            "image-selection-md",
            """
## 3. Image / atom / ROI selection

04 第一版只做固定半径圆盘积分。建议低强度检查时先只分析一个明确 sublattice/class，避免混合不同 class 的强度分布。
""",
        ),
        code(
            "image-selection-parameters",
            """
IMAGE_CHANNEL = None
IMAGE_KEY = 'raw'

# TARGET_CLASS_IDS / TARGET_CLASS_NAMES：
# 用于选择要做 intensity mapping 的原子柱类别。
# 对低强度/疑似空位分析，推荐先只分析一个明确 sublattice/class，不要混合不同 class。
TARGET_CLASS_IDS = None
TARGET_CLASS_NAMES = None

# ROIS=None 使用 global ROI。
# 如果需要手动限制区域，可传入 AnalysisROI 列表；本 notebook 不打开 napari ROI picker。
ROIS = None
""",
        ),
        md(
            "initialize-md",
            """
## 4. Initialize analysis context

这一格读取 session、解析图像、准备选定坐标源的原子点表，并显示当前 session/source 信息。
""",
        ),
        code(
            "initialize-context",
            """
context = initialize_disk_intensity_analysis(
    session_path=SESSION_PATH,
    workspace=workspace,
    session_source=SESSION_SOURCE,
    use_active_session=USE_ACTIVE_SESSION,
    result_root=RESULT_ROOT,
    coordinate_source=COORDINATE_SOURCE,
    use_keep_only=USE_KEEP_ONLY,
    class_id_filter=TARGET_CLASS_IDS,
    class_filter=TARGET_CLASS_NAMES,
    rois=ROIS,
    image_channel=IMAGE_CHANNEL,
    image_key=IMAGE_KEY,
)

session = context['session']
points = context['points']
image = context['image']
output_dirs = context['output_dirs']
preview_figures = {}
final_figures = {}

display(context['summary_tables']['disk_intensity_setup_summary'])
display(points.head())
print(f'coordinate_source: {context["coordinate_source"]}')
print(f'source_table: {context["source_table"]}')
print(f'session_load_mode: {context["session_load_mode"]}')
print(f'resolved_session_path: {context["resolved_session_path"]}')
""",
        ),
        md(
            "point-preview-md",
            """
## 5. Preview selected points

这一格只显示已选点的数量、class 和 ROI 分布，用于确认 filtering 后的分析对象。
""",
        ),
        code(
            "preview-selected-points",
            """
point_summary_rows = [
    {'field': 'rows', 'value': len(points)},
    {'field': 'unique_points', 'value': points['point_id'].nunique() if 'point_id' in points else len(points)},
    {'field': 'coordinate_source', 'value': COORDINATE_SOURCE},
    {'field': 'source_table', 'value': context['source_table']},
]
display(pd.DataFrame(point_summary_rows))
if 'class_name' in points:
    display(points['class_name'].fillna('class_unknown').astype(str).value_counts().rename_axis('class_name').reset_index(name='count'))
if 'roi_id' in points:
    display(points['roi_id'].fillna('global').astype(str).value_counts().rename_axis('roi_id').reset_index(name='count'))
""",
        ),
        md(
            "disk-radius-md",
            """
## 6. Disk radius parameters

半径太小会受单像素噪声影响大；半径太大可能混入邻近原子柱。建议根据实际原子柱宽度和间距调整。
""",
        ),
        code(
            "disk-radius-parameters",
            """
DISK_RADIUS_PX = 2.0
""",
        ),
        md(
            "aperture-preview-md",
            """
## 7. Aperture preview

预览少量点周围的积分圆盘，检查 `DISK_RADIUS_PX` 是否合理。Preview figure 默认不保存。
""",
        ),
        code(
            "aperture-preview",
            """
fig, ax = plot_disk_aperture_preview(
    image,
    points,
    disk_radius_px=DISK_RADIUS_PX,
    max_points=20,
    random_seed=0,
    show_axes=False,
    title=f'Disk aperture preview ({COORDINATE_SOURCE} coordinates)',
)
preview_figures['04A_disk_aperture_preview'] = fig
display(fig)
plt.close(fig)
""",
        ),
        md(
            "run-md",
            """
## 8. Run disk intensity calculation

这里执行唯一的核心计算：固定半径圆盘内像素强度求和。不做背景扣除、不做归一化、不做 threshold。
""",
        ),
        code(
            "run-disk-intensity",
            """
disk_config = DiskIntensityConfig(
    image_channel=context['image_channel'],
    image_key=IMAGE_KEY,
    disk_radius_px=DISK_RADIUS_PX,
    coordinate_source=COORDINATE_SOURCE,
    use_keep_only=USE_KEEP_ONLY,
)

intensity_table = compute_disk_intensity_table(
    points,
    image,
    disk_radius_px=disk_config.disk_radius_px,
    channel_name=context['image_channel'],
    image_key=disk_config.image_key,
    coordinate_source=disk_config.coordinate_source,
)

summary_table = summarize_disk_intensity(
    intensity_table,
    group_by=('coordinate_source', 'class_id', 'class_name', 'channel_name'),
    metric='disk_intensity_sum',
)
""",
        ),
        md(
            "display-results-md",
            """
## 9. Display intensity table and summary
""",
        ),
        code(
            "display-results",
            """
display(intensity_table.head())
display(summary_table)
print(f'coordinate_source: {COORDINATE_SOURCE}')
print(f'disk_radius_px: {DISK_RADIUS_PX}')
""",
        ),
        md(
            "map-parameters-md",
            """
## 10. Intensity map parameters
""",
        ),
        code(
            "map-parameters",
            """
MAP_METRIC = 'disk_intensity_sum'
MAP_CMAP = 'viridis'
MAP_POINT_SIZE = 32
MAP_EDGE_COLOR_MODE = 'class'  # 'class', 'fixed', 'none'
MAP_FIXED_EDGECOLOR = 'white'
""",
        ),
        md(
            "plot-map-md",
            """
## 11. Plot intensity map
""",
        ),
        code(
            "plot-intensity-map",
            """
map_result = plot_disk_intensity_map(
    image,
    intensity_table,
    metric=MAP_METRIC,
    cmap=MAP_CMAP,
    point_size=MAP_POINT_SIZE,
    show_colorbar=True,
    show_axes=False,
    title=f'Disk-integrated intensity map ({COORDINATE_SOURCE} coordinates)',
    show_side_panel=True,
    show_roi_outlines=True,
    rois=ROIS,
    edgecolor_mode=MAP_EDGE_COLOR_MODE,
    fixed_edgecolor=MAP_FIXED_EDGECOLOR,
)
fig = map_result[0]
final_figures['04A_disk_intensity_map'] = fig
display(fig)
plt.close(fig)
""",
        ),
        md(
            "histogram-parameters-md",
            """
## 12. Histogram parameters
""",
        ),
        code(
            "histogram-parameters",
            """
HIST_METRIC = 'disk_intensity_sum'
HIST_BINS = 30
HIST_GROUP_BY_CLASS = True
HIST_GROUP_BY_ROI = False
""",
        ),
        md(
            "plot-histogram-md",
            """
## 13. Plot histogram
""",
        ),
        code(
            "plot-histogram",
            """
fig, axes = plot_disk_intensity_histogram(
    intensity_table,
    metric=HIST_METRIC,
    bins=HIST_BINS,
    group_by_class=HIST_GROUP_BY_CLASS,
    group_by_roi=HIST_GROUP_BY_ROI,
    title=f'Disk-integrated intensity histogram ({COORDINATE_SOURCE} coordinates)',
)
final_figures['04B_disk_intensity_histogram'] = fig
display(fig)
plt.close(fig)
""",
        ),
        md(
            "export-parameters-md",
            """
## 14. Final export parameters
""",
        ),
        code(
            "final-export-parameters",
            """
SAVE_PREVIEW_FIGURES = False
SAVE_FINAL_FIGURES = True

FIGURE_FORMATS = ('pdf', 'png', 'svg')
FIGURE_DPI = 600
FIGURE_FONT_FAMILY = 'Arial'
FIGURE_FONT_SIZE = 9

FIG_04A = {
    'save': True,
    'title': f'Disk-integrated intensity map ({COORDINATE_SOURCE} coordinates)',
    'show_title': True,
    'formats': FIGURE_FORMATS,
    'dpi': FIGURE_DPI,
    'font_family': FIGURE_FONT_FAMILY,
    'font_size': FIGURE_FONT_SIZE,
}

FIG_04B = {
    'save': True,
    'title': f'Disk-integrated intensity histogram ({COORDINATE_SOURCE} coordinates)',
    'show_title': True,
    'formats': FIGURE_FORMATS,
    'dpi': FIGURE_DPI,
    'font_family': FIGURE_FONT_FAMILY,
    'font_size': FIGURE_FONT_SIZE,
}

FINAL_FIGURE_SPECS = {
    '04A_disk_intensity_map': FIG_04A,
    '04B_disk_intensity_histogram': FIG_04B,
}
""",
        ),
        md(
            "export-md",
            """
## 15. Export results

导出 CSV、figure、config、manifest 和 04 stage session，并更新 workspace active session。
""",
        ),
        code(
            "export-results",
            """
disk_intensity_export_config = {
    'dataset_id': DATASET_ID,
    'analysis_id': ANALYSIS_ID,
    'session_source': SESSION_SOURCE,
    'session_path': None if SESSION_PATH is None else str(SESSION_PATH),
    'use_active_session': USE_ACTIVE_SESSION,
    'session_load_mode': context['session_load_mode'],
    'resolved_session_path': context['resolved_session_path'],
    'coordinate_source': COORDINATE_SOURCE,
    'source_table': context['source_table'],
    'use_keep_only': USE_KEEP_ONLY,
    'image_channel': context['image_channel'],
    'image_key': IMAGE_KEY,
    'target_class_ids': TARGET_CLASS_IDS,
    'target_class_names': TARGET_CLASS_NAMES,
    'disk_radius_px': DISK_RADIUS_PX,
    'map_metric': MAP_METRIC,
    'histogram_metric': HIST_METRIC,
    'histogram_bins': HIST_BINS,
    'histogram_group_by_class': HIST_GROUP_BY_CLASS,
    'histogram_group_by_roi': HIST_GROUP_BY_ROI,
    'save_preview_figures': SAVE_PREVIEW_FIGURES,
    'save_final_figures': SAVE_FINAL_FIGURES,
    'figure_formats': FIGURE_FORMATS,
    'figure_dpi': FIGURE_DPI,
}

manifest = export_disk_intensity_analysis(
    workspace=workspace,
    result_root=RESULT_ROOT,
    session=session,
    intensity_table=intensity_table,
    summary_table=summary_table,
    config=disk_intensity_export_config,
    preview_figures=preview_figures,
    final_figures=final_figures,
    save_preview_figures=SAVE_PREVIEW_FIGURES,
    save_final_figures=SAVE_FINAL_FIGURES,
    figure_specs=FINAL_FIGURE_SPECS,
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
    target = Path(__file__).resolve().parents[1] / "notebooks" / "04_Disk_integrated_intensity_mapping.ipynb"
    target.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
