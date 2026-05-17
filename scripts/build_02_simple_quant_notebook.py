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
# 02 任务式定量分析

这个 notebook 承接 `01_Findatom.ipynb` 保存的 active session，形成四个可独立运行和导出的正式任务：

- Task 1A：period 模式下 a/b 晶格周期统计；
- Task 1B：anchor 子晶格完整局域晶胞 polygon strain mapping；
- Task 2：strict mutual nearest-pair 与 pair-center line 分布统计。

裁剪图像内的 class group 几何中心分析已经移到 `03_Cropped_group_centroid_analysis.ipynb`，02 不再维护 Task 3 入口。

`CLASS_FILTER / CLASS_ID_FILTER` 只用于可选预览，不决定后续正式任务。每个任务都有自己的 ROI、class、方向、表格、图像和 Excel 导出。
""",
        ),
        code(
            "imports",
            """
from __future__ import annotations

from pathlib import Path
import sys

# 从当前 notebook 所在目录向上查找项目根目录，确保始终导入当前项目的 src 代码，
# 而不是系统里可能已经安装过的旧版本 em_atom_workbench。
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
    BasisVectorSpec,
    DirectionSpec,
    FigureStyleConfig,
    assign_lattice_indices,
    assign_pair_center_lines_by_projection,
    build_period_histogram_title_table,
    build_complete_cells,
    build_roi_basis_table,
    compute_cell_strain,
    find_strict_mutual_nearest_pairs,
    flip_basis_vectors,
    pick_basis_vectors_with_napari,
    pick_direction_vectors_with_napari,
    pick_rois_with_napari,
    plot_pair_center_line_assignment,
    plot_pair_line_distance_errorbar,
    plot_pair_overlay,
    plot_period_angle_delta_histograms,
    plot_period_length_histograms,
    plot_polygon_cell_map,
    plot_projection_spacing_histogram,
    resolve_anchor_period_references,
    resolve_basis_vector_specs,
    resolve_direction_specs,
    run_period_statistics_ab,
    select_points_by_roi_and_class,
    summarize_pair_lines_median_iqr,
)
from em_atom_workbench.notebook_workflows import (
    export_notebook02_results,
    export_task1A_excel,
    export_task1B_excel,
    export_task2_excel,
    initialize_simple_quant_v2_analysis,
)
from em_atom_workbench.simple_quant_plotting import (
    plot_basis_check_on_image,
    plot_measurement_segments_on_image,
    plot_roi_outlines_on_image,
    summarize_rois_and_points,
)

# FIG_STYLE 是 notebook02 的统一正式绘图风格配置。
# 后续 Task 1A/1B/2/3 的 histogram、polygon map、errorbar 等图都会复用它，
# 这样字体、dpi、线宽、颜色风格可以集中调整。
FIG_STYLE = FigureStyleConfig()
plt.rcParams['figure.dpi'] = 140
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
""",
        ),
        md(
            "load-md",
            """
## 0. Load session / image / atom table

这一组 cell 只负责读取 01 notebook 保存的 active session、原始图像和原子坐标表。这里的 class filter 只用于预览，不会决定 Task 1A/1B/2/3 的正式分析对象；正式任务都会在自己的配置 cell 中重新选择 ROI 和 class。
""",
        ),
        code(
            "load-session",
            """
# RESULT_ROOT：所有 notebook02 输出的根目录。
# 默认使用项目根目录下的 results，与 01 notebook 的 active session 约定保持一致。
RESULT_ROOT = PROJECT_ROOT / 'results'
RESULT_ROOT.mkdir(exist_ok=True)

# SESSION_PATH：
# - None：读取 RESULT_ROOT / '_active_session.pkl'，这是 01 notebook 默认保存的位置；
# - Path 或字符串：读取你手动指定的 session pickle。
SESSION_PATH = None

# SOURCE_TABLE：选择 01 产生的哪一张原子坐标表作为分析起点。
# curated：正式定量推荐；refined：检查精修点；candidate：通常仅用于诊断。
SOURCE_TABLE = 'curated'

# USE_KEEP_ONLY：如果坐标表中有 keep 列，是否只保留 keep == True 的原子柱。
# 正式分析建议 True，避免被人工剔除的点进入定量。
USE_KEEP_ONLY = True

# IMAGE_CHANNEL：叠图使用的图像通道。None 表示使用 session.primary_channel。
IMAGE_CHANNEL = None

# IMAGE_KEY：叠图背景。raw 使用原始全局像素坐标；processed 用于检查预处理图像。
IMAGE_KEY = 'raw'

# CLASS_FILTER / CLASS_ID_FILTER：
# 这里只作为预览用的默认过滤，不会决定正式 Task 1A/1B/2/3 的 class。
# 正式任务必须在各自 config cell 中独立选择 class。
CLASS_FILTER = None
CLASS_ID_FILTER = None

# initialize_simple_quant_v2_analysis 会读取 session、准备预览用 analysis_points，
# 并解析显示图像。这里 rois=None，所以只是全局预览，不做 ROI 展开。
context_preview = initialize_simple_quant_v2_analysis(
    session_path=SESSION_PATH,
    result_root=RESULT_ROOT,
    source_table=SOURCE_TABLE,
    use_keep_only=USE_KEEP_ONLY,
    class_filter=CLASS_FILTER,
    class_id_filter=CLASS_ID_FILTER,
    rois=None,
    image_channel=IMAGE_CHANNEL,
    image_key=IMAGE_KEY,
)
session = context_preview['session']
image = context_preview['image']
output_dirs = context_preview['output_dirs']

# 显示 session 概要和原子表头几行，用于确认 01 notebook 的结果已经正确加载。
display(context_preview['summary_tables']['simple_quant_v2_summary'])
display(context_preview['summary_tables']['analysis_points_preview'])
""",
        ),
        md(
            "roi-preview-md",
            """
## 1. Optional global ROI preview

这里选择 notebook02 的初始 ROI 集合 `ROIS`，用于 Task 1A/1B/2 的默认空间范围和 basis 选择检查。裁剪几何中心分析请在 03 notebook 中重新选择 crop ROI 和 measurement ROI。

这个 cell 还会稳定创建 `all_analysis_points = analysis_points.copy()`，避免后续 cell 因 pair center 未生成而找不到变量。
""",
        ),
        code(
            "roi-selection",
            """
# OPEN_ROI_PICKER：True 时打开 napari，让你手动画多个 polygon/rectangle ROI。
# False 时使用下面手动填写的 ROIS；如果 ROIS 也是 None，则使用全图 global ROI。
OPEN_ROI_PICKER = True

# ROIS：初始 ROI 集合。它用于 Task 1A/1B/2 的默认空间范围。
# 每个 ROI 是 AnalysisROI，polygon_xy_px 使用全局像素坐标 (x, y)。
ROIS = None

# DEFAULT_ROI_PREFIX：napari 画 ROI 后自动生成 roi_name 的前缀，只影响显示和导出。
DEFAULT_ROI_PREFIX = 'ROI'

# 根据 OPEN_ROI_PICKER 决定是否打开 napari。关闭 napari 后，ROIS 会保持用户手动填写值。
if OPEN_ROI_PICKER:
    ROIS = pick_rois_with_napari(
        session,
        context_preview['analysis_points'],
        image_channel=context_preview['image_channel'],
        image_key=IMAGE_KEY,
        default_roi_prefix=DEFAULT_ROI_PREFIX,
        start_index=1,
    )
elif ROIS is None:
    ROIS = [AnalysisROI(roi_id='global', roi_name='global', polygon_xy_px=None, color='#ff9f1c')]

# 使用 ROIS 重新准备正式 analysis_points。
# 同一个原子若落入多个 ROI，会在 analysis_points 中出现多行，用 roi_id/scope_id 区分。
context = initialize_simple_quant_v2_analysis(
    session_path=SESSION_PATH,
    result_root=RESULT_ROOT,
    source_table=SOURCE_TABLE,
    use_keep_only=USE_KEEP_ONLY,
    class_filter=None,
    class_id_filter=None,
    rois=ROIS,
    image_channel=IMAGE_CHANNEL,
    image_key=IMAGE_KEY,
)
analysis_points = context['analysis_points']
PIXEL_TO_NM = analysis_points.attrs.get('pixel_to_nm')
HAS_PIXEL_CALIBRATION = PIXEL_TO_NM is not None and np.isfinite(float(PIXEL_TO_NM))
if not HAS_PIXEL_CALIBRATION:
    print('Warning: current session has no pixel calibration. Formal Task 1A length figures will not display px fallback; set calibration in 01 notebook to enable Å plots.')

# all_analysis_points 是后续统一绘图/导出使用的点表。
# 这里先稳定定义为 atom 点本身，避免未生成 pair center 时后续 cell 报变量未定义。
all_analysis_points = analysis_points.copy()
roi_table = context['roi_table']
image = context['image']
output_dirs = context['output_dirs']

# figures 用来收集 notebook 中生成的正式图，final export cell 会统一保存。
figures = {}

# excel_exports 用来记录每个任务独立 Excel 的路径和写入状态。
excel_exports = {}

# ROI summary 用于快速检查每个 ROI 内点数、class 数量和 point_set 数量。
roi_summary_table = summarize_rois_and_points(analysis_points, rois=ROIS)
display(pd.DataFrame([roi.__dict__ for roi in ROIS]))
display(roi_summary_table)
display(analysis_points.head())

# 预览图：显示原始图像、ROI 边界和 class-colored atoms。
# 这不是正式定量结果，只用于确认 ROI 和 class 分布是否合理。
fig, image_ax, side_ax = plot_measurement_segments_on_image(
    image,
    analysis_points,
    pd.DataFrame(),
    rois=ROIS,
    show_roi_outlines=True,
    show_side_panel=True,
    show_axes=False,
    title='02A class-colored analysis points',
)
figures['02A_analysis_points_preview'] = fig
display(fig)
plt.close(fig)
""",
        ),
        md(
            "basis-md",
            """
## 2. Define or pick basis vectors for Task 1

这里定义 Task 1A 和 Task 1B 使用的 a/b 两个基矢方向。basis vector 的方向用于周期匹配和晶胞编号，vector 的长度默认作为初始 period 参考；Task 1B 后续会优先使用 Task 1A 得到的 anchor class 周期统计作为 reference。
""",
        ),
        code(
            "basis-selection",
            """
# OPEN_BASIS_VECTOR_PICKER：True 时打开 napari，为 a/b 每个基矢各选两个点。
# 两点顺序决定 vector 方向；vector 长度会作为初始 period_px。
OPEN_BASIS_VECTOR_PICKER = True

# BASIS_MODE：
# global：所有 ROI 共用同一套 a/b basis，适合单取向样品；
# per_roi：每个 ROI 单独选择 a/b basis，适合多畴或局部取向不同的样品。
BASIS_MODE = 'global'

# BASIS_ROLES：Task 1A/1B 需要的晶格方向。这里固定使用 a 和 b 两个方向。
BASIS_ROLES = ('a', 'b')

# BASIS_NAMES：global 模式下 napari layer 的名称；per_roi 模式会自动生成 roi_x_a/roi_x_b。
BASIS_NAMES = BASIS_ROLES

# SNAP_BASIS_POINTS_TO_NEAREST_POINTS：True 时将手点吸附到最近 atom 点，减少人工点击偏差。
SNAP_BASIS_POINTS_TO_NEAREST_POINTS = True

# BASIS_VECTOR_SPECS：无 GUI 或复现实验时可直接手动指定 basis。
# vector_px 是完整周期矢量，不是单位向量；basis_role 用于映射到 a/b。
BASIS_VECTOR_SPECS = [
    BasisVectorSpec(name='a', basis_role='a', vector_px=(10.0, 0.0)),
    BasisVectorSpec(name='b', basis_role='b', vector_px=(0.0, 10.0)),
]

# 如果打开 picker，会用 napari 中手动选择的结果覆盖上面的 BASIS_VECTOR_SPECS。
if OPEN_BASIS_VECTOR_PICKER:
    BASIS_VECTOR_SPECS = pick_basis_vectors_with_napari(
        session,
        analysis_points,
        basis_names=BASIS_NAMES,
        image_channel=context['image_channel'],
        image_key=IMAGE_KEY,
        snap_to_nearest_points=SNAP_BASIS_POINTS_TO_NEAREST_POINTS,
        point_size=5.0,
        rois=ROIS,
        basis_mode=BASIS_MODE,
        basis_roles=BASIS_ROLES,
    )

# FLIP_BASIS_NAMES：如果 napari 两个点顺序反了，在这里填 basis_name 即可翻转方向。
# 例如 ('a',) 或 ('roi_1_a',)。翻转方向不会改变 period 长度。
FLIP_BASIS_NAMES = ()

# GLOBAL_BASIS_FALLBACK：per_roi 模式下，如果某个 ROI 没有自己的 basis，
# True 允许回退到 global 同 role basis；多畴严格分析时建议 False。
GLOBAL_BASIS_FALLBACK = True

# resolve_basis_vector_specs 将 BasisVectorSpec 转换成表格：
# 包括 ux/uy 单位向量、period_px、angle_deg、来源点等。
basis_vector_table = resolve_basis_vector_specs(analysis_points, BASIS_VECTOR_SPECS)
if FLIP_BASIS_NAMES:
    basis_vector_table = flip_basis_vectors(basis_vector_table, FLIP_BASIS_NAMES)

# roi_basis_table 明确每个 ROI 的 a/b 方向应该使用哪个 basis_name。
roi_basis_table = build_roi_basis_table(
    ROIS,
    basis_vector_table,
    basis_roles=BASIS_ROLES,
    global_fallback=GLOBAL_BASIS_FALLBACK,
)
display(basis_vector_table)
display(roi_basis_table)

# basis check 图用于检查 a/b 方向、ROI 和 atom 分布是否匹配。
# 如果 a/b 方向选错，Task 1A 的 period 匹配和 Task 1B 的 lattice indexing 都会受影响。
fig, image_ax, legend_ax = plot_basis_check_on_image(
    image,
    analysis_points,
    basis_vector_table,
    rois=ROIS,
    show_roi_outlines=True,
    show_atoms=True,
    show_basis_labels=False,
    show_axes=False,
    title='Task 1 ROI and a/b basis check',
)
figures['02B_task1_basis_check'] = fig
display(fig)
plt.close(fig)
""",
        ),
        md(
            "task1a-config-md",
            """
## 3. Task 1A config: ROI/class selection

Task 1A 在每个 ROI 中沿 a/b 两个方向统计周期长度和方向偏差。这里要为 Task 1A 独立选择 ROI 内使用的 class，并决定多个 class 是分别统计还是合并统计。

默认 `class_group_mode='per_class'`，这是更安全的选择，因为不同 motif 混在一起可能让 period vector 连到错误的原子柱。

注意：`TASK1A_ROI_CLASS_SELECTION` 的 key 必须写 `roi_id`，不是图上显示的标题。`roi_id` 通常是小写的 `roi_1`、`roi_2`；`ROI_1`、`Aera_1` 这类名字通常是 `roi_name` 或图标题。请以上面 ROI summary / `display(pd.DataFrame([roi.__dict__ for roi in ROIS]))` 里 `roi_id` 那一列为准。
""",
        ),
        code(
            "task1a-config",
            """
# TASK1A_ROI_CLASS_SELECTION：Task 1A 专用 class 选择，不沿用全局 CLASS_FILTER。
# key 必须是实际 roi_id，大小写要一致；通常是 'roi_1'、'roi_2'，不是 'ROI_1' 或 'Aera_1'。
# 如果把 key 写成 ROI_1/Aera_1，而实际 roi_id 是 roi_1，这个 ROI 的设置不会命中；
# 若同时保留 'default': None，就会退回到“该 ROI 内所有 class”，导致图里突然出现很多 class/task。
# 请复制 Step 1 输出表中 roi_id 那一列的值来填写这里的 key。
#
# value 可以是：
# - None：该 ROI 内所有 class；
# - (0,)：只统计 class_id=0；
# - (0, 1)：统计 class_id=0 和 1，具体是否合并由 TASK1A_CLASS_GROUP_MODE 决定。
TASK1A_ROI_CLASS_SELECTION = {
    # 例：只统计 roi_1 和 roi_2 里的 class_id=2。
    # 注意这里是小写 roi_1 / roi_2，因为它们是 roi_id。
    # 'roi_1': (2,),
    # 'roi_2': (2,),

    # default 是兜底规则：没有单独写 key 的 ROI 会用它。
    # 如果希望未列出的 ROI 不小心跑全部 class，就不要保留 'default': None；
    # 如果确实想让所有未列出 ROI 使用全部 class，才打开这一行。
    'default': None,
}

# TASK1A_CLASS_GROUP_MODE：
# per_class：默认，多个 class 分别统计，避免不同 motif 混合导致 period vector 连错；
# union：显式合并多个 class 作为一个点集统计，只在你确认这些 class 应共同组成同一周期链时使用。
TASK1A_CLASS_GROUP_MODE = 'per_class'

# TASK1A_MATCH_RADIUS_FRACTION：period 匹配容差，相对于 period_px 的比例。
# 太小会漏掉畸变晶格；太大会误连到邻近链/邻近 motif。
TASK1A_MATCH_RADIUS_FRACTION = 0.30

# TASK1A_MATCH_RADIUS_PX：绝对像素容差。None 表示使用 period_px * fraction。
TASK1A_MATCH_RADIUS_PX = None

# TASK1A_ONE_TO_ONE：True 时同一个 target 只允许被一个 source 匹配，减少重复连线。
TASK1A_ONE_TO_ONE = True

# 把 Task 1A 参数整理成表格，方便 Excel 和 manifest 记录。
task1A_config_table = pd.DataFrame([
    {'parameter': 'class_group_mode', 'value': TASK1A_CLASS_GROUP_MODE},
    {'parameter': 'basis_roles', 'value': str(BASIS_ROLES)},
    {'parameter': 'match_radius_fraction', 'value': TASK1A_MATCH_RADIUS_FRACTION},
    {'parameter': 'match_radius_px', 'value': TASK1A_MATCH_RADIUS_PX},
    {'parameter': 'one_to_one', 'value': TASK1A_ONE_TO_ONE},
])
display(task1A_config_table)
""",
        ),
        md(
            "task1a-run-md",
            """
## 4. Task 1A: run a/b period statistics

这个 cell 真正运行 Task 1A。它会按 ROI、direction(a/b)、class selection 展开 period-vector 任务，并输出两张核心表：`period_segment_table` 和 `period_summary_table`。
""",
        ),
        code(
            "task1a-run",
            """
# run_period_statistics_ab 会自动按 ROI、a/b direction、class selection 展开任务。
# 输出的 period_segment_table 是逐线段表，period_summary_table 是每组统计摘要。
task1A_result = run_period_statistics_ab(
    analysis_points,
    basis_vector_table,
    roi_basis_table,
    roi_class_selection=TASK1A_ROI_CLASS_SELECTION,
    basis_roles=BASIS_ROLES,
    class_group_mode=TASK1A_CLASS_GROUP_MODE,
    match_radius_fraction=TASK1A_MATCH_RADIUS_FRACTION,
    match_radius_px=TASK1A_MATCH_RADIUS_PX,
    one_to_one=TASK1A_ONE_TO_ONE,
)
# period_segment_table：每一行是一条识别到的周期线段，含长度、角度、valid 状态等。
period_segment_table = task1A_result['period_segment_table']

# period_summary_table：每一行对应一个 ROI + direction + class_selection 的统计结果。
period_summary_table = task1A_result['period_summary_table']

# task1A_roi_class_selection：记录每个 ROI、方向和 class selection 的实际展开情况。
# 如果这里出现 class_id:0、class_id:1、class_id:2 等多行，而你本来只想分析一个 class，
# 优先检查上一格 TASK1A_ROI_CLASS_SELECTION 的 key 是否写成了实际 roi_id。
# 常见错误是把显示名 ROI_1/Aera_1 当成 key；实际 key 应该是小写 roi_1。
task1A_roi_class_selection = task1A_result['roi_class_selection']

# task1A_tasks：记录实际生成的 PeriodicVectorTask 参数，便于复现。
task1A_tasks = task1A_result['task1A_tasks']

display(period_segment_table.head())
display(period_summary_table)
""",
        ),
        md(
            "task1a-fig-md",
            """
## 5. Task 1A: histograms and period segment overlay

这里生成 Task 1A 的正式检查图和统计图：周期线段叠加图、长度直方图和角度偏差直方图。直方图会叠加 single Gaussian fit，并使用统一的 `FIG_STYLE`。
""",
        ),
        code(
            "task1a-figure-title-config",
            """
# 这一格专门控制 Step 5 的 histogram 图标题。默认标题保持简洁，例如：
# ROI_1 a Length
# ROI_1 b angle
#
# 你可以直接修改这个模板来统一改变所有 Task 1A histogram 标题。
# 可用占位符包括：
# - {roi_display_label}：显示用 ROI 标签，例如 ROI_1；
# - {roi_display_index}：ROI 数字编号，例如 1；
# - {direction}：a 或 b；
# - {metric_short}：Length 或 angle；
# - {class_selection}：class selection，适合需要把 class 写进标题时使用；
# - {metric_label}, {roi_name}, {roi_id}, {metric}, {value_column}, {figure_key}。
TASK1A_HIST_TITLE_TEMPLATE = "{roi_display_label} {direction} {metric_short}"

# 如果某一张图需要投稿级别的短标题或中文标题，可以在这里逐张覆盖。
# key 的格式是：(roi_id, direction, class_selection, metric)
# - metric='length' 表示周期长度直方图；
# - metric='angle' 表示角度偏差直方图。
# 也可以用完整 figure_key 作为 key，例如：
# "task1A_period_length_roi_1_a_class_id_0_length_A": "Custom title"
TASK1A_HIST_TITLE_OVERRIDES = {
    # ("roi_1", "a", "class_id:0", "length"): "ROI_1 a Length",
    # ("roi_1", "a", "class_id:0", "angle"): "ROI_1 a angle",
}

# 周期线段叠加图的标题也单独留成变量，方便按论文图注风格修改。
TASK1A_SEGMENT_OVERLAY_TITLE = "Task 1A a/b period segments"
""",
        ),
        code(
            "task1a-figures",
            """
# period_segment_table 使用 Task 1A 专用字段 p0/p1；
# 为了复用现有 segment overlay 绘图函数，这里临时重命名成 source/target 字段。
fig, image_ax, side_ax = plot_measurement_segments_on_image(
    image,
    analysis_points,
    period_segment_table.rename(columns={
        'p0_x': 'source_x_px',
        'p0_y': 'source_y_px',
        'p1_x': 'target_x_px',
        'p1_y': 'target_y_px',
        'length_px': 'distance_px',
        'length_A': 'distance_A',
    }),
    basis_vector_table=basis_vector_table,
    rois=ROIS,
    show_roi_outlines=True,
    show_basis_vectors=True,
    basis_display_unit='A',
    pixel_to_nm=PIXEL_TO_NM,
    color_by='task',
    linewidth=1.3,
    alpha=0.82,
    show_side_panel=True,
    show_axes=False,
    title=TASK1A_SEGMENT_OVERLAY_TITLE,
)
figures['task1A_period_segment_overlay'] = fig
display(fig)
plt.close(fig)

# 先生成一个标题预览表，确认每张 histogram 对应哪个 ROI、方向和 class。
# 如果标题不适合当前论文或汇报，可以回到上一格修改模板或覆盖字典后重新运行本格。
task1A_histogram_title_table = build_period_histogram_title_table(
    period_segment_table,
    title_template=TASK1A_HIST_TITLE_TEMPLATE,
    title_overrides=TASK1A_HIST_TITLE_OVERRIDES,
)
display(task1A_histogram_title_table)

# 长度直方图只使用 Å；如果没有像素标定，不会回退显示 px。
if not HAS_PIXEL_CALIBRATION:
    print('Warning: no pixel calibration found. Task 1A length histograms are skipped because formal figures should use Å only.')
length_hist_figures = plot_period_length_histograms(
    period_segment_table,
    title_template=TASK1A_HIST_TITLE_TEMPLATE,
    title_overrides=TASK1A_HIST_TITLE_OVERRIDES,
    style=FIG_STYLE,
)

# 角度偏差直方图使用 wrapped angle_delta_deg，避免 179°/-179° 周期跳变影响统计。
angle_hist_figures = plot_period_angle_delta_histograms(
    period_segment_table,
    title_template=TASK1A_HIST_TITLE_TEMPLATE,
    title_overrides=TASK1A_HIST_TITLE_OVERRIDES,
    style=FIG_STYLE,
)
for key, fig in {**length_hist_figures, **angle_hist_figures}.items():
    figures[key] = fig
    display(fig)
    plt.close(fig)
""",
        ),
        md(
            "task1a-export-md",
            """
## 6. Task 1A: export Excel

运行完 Task 1A 后可以立刻执行这个 cell，把周期线段、统计摘要、配置、ROI/class selection 和 basis vectors 写入独立 Excel 文件。
""",
        ),
        code(
            "task1a-export",
            """
# 每个任务都有独立 Excel，方便只运行到某个任务时立即导出。
# 如果某些表为空，导出函数仍会创建 sheet 并写入 warning。
task1A_excel = export_task1A_excel(
    output_dirs,
    period_segment_table=period_segment_table,
    period_summary_table=period_summary_table,
    task1A_config=task1A_config_table,
    roi_class_selection=task1A_roi_class_selection,
    basis_vectors=basis_vector_table,
)
excel_exports['task1A'] = task1A_excel
display(task1A_excel)
""",
        ),
        md(
            "task1b-config-md",
            """
## 7. Task 1B config: anchor class, reference, origin

Task 1B 只使用 anchor 子晶格构建完整局域晶胞 polygon。这里指定每个 ROI 的 anchor class、reference 来源和 lattice origin。

默认 reference 规则很严格：必须使用同一 ROI、同一 anchor class 在 Task 1A 中得到的 a/b period summary；如果缺失，notebook 会提示你重新运行 Task 1A、手动指定 reference，或基于 anchor 点重新估计，绝不会静默借用其他 class 的周期。
""",
        ),
        code(
            "task1b-config",
            """
# TASK1B_ANCHOR_SELECTION：每个 ROI 用哪个 class 作为 anchor 子晶格。
# 只有 anchor class 会参与晶胞角点构建；非 anchor 原子保留在 atom table 中但不参与 cell 构建。
TASK1B_ANCHOR_SELECTION = {
    # Default expects one anchor class per ROI. Update with your real anchor class ids.
    # 'roi_1': 0,
}

# TASK1B_MANUAL_REFERENCE_PX：当 Task 1A 没有该 anchor class 的 a/b period summary 时，
# 可以手动提供 a_ref_px/b_ref_px。这里必须按 ROI 显式填写，避免误用其他 class 的 reference。
TASK1B_MANUAL_REFERENCE_PX = {
    # Optional fallback when Task 1A lacks anchor-class period summary.
    # 'roi_1': {'a_ref_px': 10.0, 'b_ref_px': 10.0},
}

# TASK1B_ORIGIN_POINT_ID：可选手动 lattice origin。
# 如果不填，函数会自动选择投影最小的 anchor 作为 origin；自动 origin 后请检查 residual 图/表。
TASK1B_ORIGIN_POINT_ID = {
    # Optional: 'roi_1': 'atom:123'
}

# TASK1B_MAX_RESIDUAL_FRACTION：anchor 点到理想格点的 residual / min(a_ref,b_ref) 上限。
# 超过该阈值的 anchor 会标记 low_confidence，不参与 valid cell 构建。
TASK1B_MAX_RESIDUAL_FRACTION = 0.35

# resolve_anchor_period_references 会严格查找“同 ROI + 同 anchor class”的 Task 1A reference。
task1B_reference_table = resolve_anchor_period_references(period_summary_table, TASK1B_ANCHOR_SELECTION)
if not task1B_reference_table.empty and (~task1B_reference_table['valid']).any():
    print('Task 1B reference warning: missing anchor-class Task 1A summary. Re-run Task 1A for the anchor class, set TASK1B_MANUAL_REFERENCE_PX, or estimate reference from anchor points.')
display(task1B_reference_table)
""",
        ),
        md(
            "task1b-run-md",
            """
## 8. Task 1B: lattice indexing, complete cells, strain

这个 cell 执行 anchor 点二维晶格编号、完整四角 cell 构建和局域几何/strain 计算。缺角、跨出 ROI、duplicate anchor 或 residual 过大的情况会保留在 QC 表中，不会被补点或外推。
""",
        ),
        code(
            "task1b-run",
            """
# anchor_lattice_tables：保存每个 ROI 的 anchor 点 lattice_i/lattice_j 编号结果。
anchor_lattice_tables = []

# cell_tables：保留变量名用于 notebook 读者理解；最终完整 cell 表由 raw_cell_table/cell_table 生成。
cell_tables = []

# task1B_anchor_selection_rows：记录每个 ROI 的 anchor class 和 anchor 点数量。
task1B_anchor_selection_rows = []

for roi_id, anchor_class in TASK1B_ANCHOR_SELECTION.items():
    # 1) 只选出当前 ROI 内的 anchor class 点。
    anchor_points = select_points_by_roi_and_class(
        analysis_points,
        roi_ids=(roi_id,),
        class_ids=(int(anchor_class),),
        point_set='atoms',
    )
    # 2) 取当前 ROI 对应的 a/b basis。per_roi 模式下会使用该 ROI 自己的 basis；
    #    global 模式下则使用全局 a/b basis。
    basis_rows = roi_basis_table.loc[(roi_basis_table['roi_id'].astype(str) == str(roi_id)) & (roi_basis_table['found'].astype(bool))]
    basis_a = basis_vector_table.loc[basis_vector_table['basis_name'].astype(str) == str(basis_rows.loc[basis_rows['basis_role'].astype(str) == 'a'].iloc[0]['basis_name'])].iloc[0]
    basis_b = basis_vector_table.loc[basis_vector_table['basis_name'].astype(str) == str(basis_rows.loc[basis_rows['basis_role'].astype(str) == 'b'].iloc[0]['basis_name'])].iloc[0]

    # 3) reference 优先来自 Task 1A 中同 ROI、同 anchor class 的 a/b median period。
    #    如果缺失，只允许使用用户显式填写的 TASK1B_MANUAL_REFERENCE_PX。
    ref_row = task1B_reference_table.loc[task1B_reference_table['roi_id'].astype(str) == str(roi_id)]
    if not ref_row.empty and bool(ref_row.iloc[0]['valid']):
        a_ref_px = float(ref_row.iloc[0]['a_ref_px'])
        b_ref_px = float(ref_row.iloc[0]['b_ref_px'])
    elif roi_id in TASK1B_MANUAL_REFERENCE_PX:
        a_ref_px = float(TASK1B_MANUAL_REFERENCE_PX[roi_id]['a_ref_px'])
        b_ref_px = float(TASK1B_MANUAL_REFERENCE_PX[roi_id]['b_ref_px'])
    else:
        print(f'Skip Task 1B ROI {roi_id}: missing anchor-class reference. Do not silently use another class reference.')
        continue

    # 4) 将 anchor 点投影到 a_ref/b_ref 形成的二维参考晶格中，得到 lattice_i/lattice_j。
    #    duplicate anchor 和 high residual anchor 会被标记为 invalid/low_confidence。
    anchor_lattice = assign_lattice_indices(
        anchor_points,
        a_ref_px=a_ref_px,
        b_ref_px=b_ref_px,
        unit_a=(float(basis_a['ux']), float(basis_a['uy'])),
        unit_b=(float(basis_b['ux']), float(basis_b['uy'])),
        origin_point_id=TASK1B_ORIGIN_POINT_ID.get(roi_id),
        max_residual_fraction=TASK1B_MAX_RESIDUAL_FRACTION,
    )
    anchor_lattice_tables.append(anchor_lattice)
    task1B_anchor_selection_rows.append({'roi_id': roi_id, 'anchor_class_id': anchor_class, 'n_anchor_points': len(anchor_points)})

# 合并所有 ROI 的 anchor lattice 编号结果。
anchor_lattice_table = pd.concat(anchor_lattice_tables, ignore_index=True) if anchor_lattice_tables else pd.DataFrame()

# build_complete_cells 只保留四角齐全的候选 cell；缺角或跨出 ROI 的 cell 会标记 invalid。
raw_cell_table = build_complete_cells(anchor_lattice_table, rois=ROIS)

# compute_cell_strain 计算 a_local/b_local/theta/area 和 eps_a/eps_b/eps_area。
# 面积使用二维叉乘，不用 a*b 近似。
cell_table, strain_reference_table = compute_cell_strain(raw_cell_table)
valid_cell_table = cell_table.loc[cell_table.get('valid', False).astype(bool)].copy() if not cell_table.empty else pd.DataFrame()
invalid_cell_table = cell_table.loc[~cell_table.get('valid', False).astype(bool)].copy() if not cell_table.empty else pd.DataFrame()
task1B_anchor_selection_table = pd.DataFrame(task1B_anchor_selection_rows)

# QC summary 用于快速判断 anchor 编号和 cell 构建是否稳定。
task1B_qc_summary = pd.DataFrame([
    {'metric': 'anchor_rows', 'value': len(anchor_lattice_table)},
    {'metric': 'valid_anchor_rows', 'value': int(anchor_lattice_table.get('valid_anchor', pd.Series(dtype=bool)).sum()) if not anchor_lattice_table.empty else 0},
    {'metric': 'cell_rows', 'value': len(cell_table)},
    {'metric': 'valid_cells', 'value': len(valid_cell_table)},
    {'metric': 'invalid_cells', 'value': len(invalid_cell_table)},
])

# 如果大量 cell invalid，通常意味着 ROI 跨多畴、basis/reference 不适合，或 anchor class 选择不对。
if not task1B_qc_summary.empty and len(valid_cell_table) < max(1, 0.5 * len(cell_table)):
    print('Task 1B QC warning: many cells are invalid/low confidence. For multi-domain or strong orientation changes, split into smaller single-domain ROIs.')
display(anchor_lattice_table.head())
display(task1B_qc_summary)
display(cell_table.head())
""",
        ),
        md(
            "task1b-fig-md",
            """
## 9. Task 1B: polygon strain maps

这里绘制 polygon cell map。每个 valid cell 作为一个四边形叠加在原始图像上，颜色表示 `strain_a`、`strain_b`、`area_strain` 或 `area_local`。invalid cell 不绘制，边缘缺失区域保持透明。

当前实现中，Task 1A 的 a/b period reference 用于 anchor lattice indexing，也就是帮助找到完整晶胞角点。下面这些 strain map 的 reference 则由 `compute_cell_strain(raw_cell_table)` 按每个 ROI 独立计算：如果没有额外传入 reference table，就使用该 ROI 内所有 valid cell 的 median 作为内部 reference。因此 ROI_1 的 `strain_a/strain_b/area_strain` 是相对 ROI_1 自己 valid cells 的 median，ROI_2 同理相对 ROI_2 自己的 median。`area_local` 是局域晶胞绝对面积，不是相对面积；相对面积应变对应 `area_strain`。
""",
        ),
        code(
            "task1b-figures",
            """
# 默认输出四张核心 polygon map：
# strain_a/strain_b/area_strain 分别来自 cell_table 的 eps_a/eps_b/eps_area 列。
# 它们的计算公式是：
# - strain_a = (a_local - a_ref) / a_ref
# - strain_b = (b_local - b_ref) / b_ref
# - area_strain = (area_local - area_ref) / area_ref
#
# 在当前默认流程里，a_ref、b_ref、area_ref 是每个 ROI 内 valid cell 的 median，
# 所以两个 ROI 的 strain 默认是各自 ROI 内部归一化后的相对变化，不是相对另一个 ROI。
# area_local 图显示的是绝对局域晶胞面积；它本身不相对任何 reference。
TASK1B_POLYGON_MAP_LABELS = {
    'eps_a': 'strain_a',
    'eps_b': 'strain_b',
    'eps_area': 'area_strain',
    'area_local': 'area_local',
}
for value_column in ('eps_a', 'eps_b', 'eps_area', 'area_local'):
    display_label = TASK1B_POLYGON_MAP_LABELS[value_column]
    fig, ax = plot_polygon_cell_map(
        image,
        cell_table,
        value_column,
        rois=ROIS,
        style=FIG_STYLE,
        title=f'Task 1B {display_label} polygon map',
    )
    figures[f'task1B_{display_label}_polygon_map'] = fig
    display(fig)
    plt.close(fig)
""",
        ),
        md(
            "task1b-export-md",
            """
## 10. Task 1B: export Excel

将 Task 1B 的 cell table、valid/invalid cell、strain reference、anchor selection 和 QC summary 写入独立 Excel 文件。即使某些表为空，也会创建对应 sheet 并记录 warning。
""",
        ),
        code(
            "task1b-export",
            """
# 导出配置中明确记录 residual 阈值和 reference 规则，方便回溯。
task1B_config_table = pd.DataFrame([
    {'parameter': 'max_residual_fraction', 'value': TASK1B_MAX_RESIDUAL_FRACTION},
    {'parameter': 'reference_rule', 'value': 'same ROI and same anchor class Task 1A summary, or explicit manual fallback'},
])
task1B_excel = export_task1B_excel(
    output_dirs,
    cell_table=cell_table,
    strain_reference_table=strain_reference_table,
    task1B_config=task1B_config_table,
    anchor_selection=task1B_anchor_selection_table,
    qc_summary=task1B_qc_summary,
)
excel_exports['task1B'] = task1B_excel
display(task1B_excel)
""",
        ),
        md(
            "task2-config-md",
            """
## 11. Task 2 config: pair, class, projection axis

Task 2 在独立选择的 ROI/class 中寻找 strict mutual nearest-neighbor pair，并用 pair center 沿 projection axis 的一维坐标进行 line/row 分组。

注意：`projection_axis` 是用来区分 line/row 的分组方向，通常是 line/row 的法向方向，不是 line 本身的延伸方向。
""",
        ),
        code(
            "task2-config",
            """
# TASK2_ROI_CLASS_SELECTION：Task 2 专用 ROI/class 选择。
# {'default': None} 表示每个 ROI 默认使用全部 class；也可以写 {'roi_1': (0,)}。
TASK2_ROI_CLASS_SELECTION = {'default': None}

# TASK2_PAIR_MODE：
# within_class：默认，在同一个 class 内找 mutual nearest pair；
# between_classes：在 source class 和 target class 之间找 mutual nearest pair。
TASK2_PAIR_MODE = 'within_class'

# between_classes 模式需要显式设置 source/target class。
# within_class 模式下这两个参数通常保持 None，由 TASK2_ROI_CLASS_SELECTION 控制。
TASK2_SOURCE_CLASS_IDS = None
TASK2_TARGET_CLASS_IDS = None

# TASK2_MAX_PAIR_DISTANCE_PX / TASK2_MAX_PAIR_DISTANCE_A：
# pair 距离阈值。超过阈值的 pair 会保留在 pair_table 中，但 valid=False、invalid_reason='too_far'，
# 不参与 line summary。
TASK2_MAX_PAIR_DISTANCE_PX = None
TASK2_MAX_PAIR_DISTANCE_A = None

# projection_axis 用于给 pair center 分 line/row。
# 如果希望不同 ROI 的 line 按整张图同一方向统一编号，请保持 TASK2_LINE_INDEX_MODE='global'，
# 并把 TASK2_PROJECTION_VECTOR 设置成这个统一编号方向。
# 通常：如果你要区分一排一排的 line，projection vector 应接近 line/row 的法向；
# 如果你明确要“沿某个基矢方向排序编号”，则把 projection vector 设置成对应基矢方向。
# TASK2_OPEN_PROJECTION_PICKER：True 时打开 napari 重新选择 projection axis。
TASK2_OPEN_PROJECTION_PICKER = False

# TASK2_PROJECTION_VECTOR：手动 projection axis，二维像素向量。
# 它只用于 pair center 的一维投影分组，不等同于 Task 1 的 basis。
TASK2_PROJECTION_VECTOR = (1.0, 0.0)

# TASK2_LINE_INDEX_MODE：
# global：所有 ROI 的 pair center 放在同一个 projection 坐标系里统一分 line，
#         输出 global_line_id，适合比较不同 ROI 在整张图同一方向上的 line trend。
# per_roi：每个 ROI 内部单独从 1 开始编号，保留旧行为。
TASK2_LINE_INDEX_MODE = 'global'

# TASK2_LINE_TOLERANCE_PX：projection coordinate 上的分组容差。
# None 时会根据 pair center projection spacing 自动给出建议并使用 fallback。
TASK2_LINE_TOLERANCE_PX = None

# TASK2_MIN_PAIRS_PER_LINE：每条 line 至少保留多少个 pair；低于该值的 line 标记 line_too_short。
TASK2_MIN_PAIRS_PER_LINE = 2

if TASK2_OPEN_PROJECTION_PICKER:
    # napari 中画两个点定义 projection axis；snap_to_nearest_atoms=False 保留任意方向。
    TASK2_PROJECTION_SPECS = pick_direction_vectors_with_napari(
        session,
        analysis_points,
        direction_names=('task2_projection_axis',),
        image_channel=context['image_channel'],
        image_key=IMAGE_KEY,
        snap_to_nearest_atoms=False,
    )
    task2_projection_table = resolve_direction_specs(analysis_points, TASK2_PROJECTION_SPECS)
    # resolve_direction_specs 返回单位向量 ux/uy，这里作为 projection vector 使用。
    TASK2_PROJECTION_VECTOR = (
        float(task2_projection_table.iloc[0]['ux']),
        float(task2_projection_table.iloc[0]['uy']),
    )
else:
    # 手动 vector 也整理为 projection axis table，便于 Excel 和 manifest 记录。
    task2_projection_table = pd.DataFrame([{'direction_name': 'task2_projection_axis', 'ux': TASK2_PROJECTION_VECTOR[0], 'uy': TASK2_PROJECTION_VECTOR[1]}])

# task2_config_table 记录所有关键参数，导出到 Excel。
task2_config_table = pd.DataFrame([
    {'parameter': 'pair_mode', 'value': TASK2_PAIR_MODE},
    {'parameter': 'max_pair_distance_px', 'value': TASK2_MAX_PAIR_DISTANCE_PX},
    {'parameter': 'max_pair_distance_A', 'value': TASK2_MAX_PAIR_DISTANCE_A},
    {'parameter': 'line_tolerance_px', 'value': TASK2_LINE_TOLERANCE_PX},
    {'parameter': 'min_pairs_per_line', 'value': TASK2_MIN_PAIRS_PER_LINE},
    {'parameter': 'line_index_mode', 'value': TASK2_LINE_INDEX_MODE},
])
display(task2_config_table)
display(task2_projection_table)
""",
        ),
        md(
            "task2-run-md",
            """
## 12. Task 2: strict mutual nearest pairs and line grouping

这个 cell 先计算 pair table，再把 valid pair center 按 projection coordinate 聚成 line，最后输出每条 line 的 distance median 和 IQR。
""",
        ),
        code(
            "task2-run",
            """
# 1) strict mutual nearest-neighbor pair：
# within_class 下会对 unordered pair 去重，确保 (i,j) 和 (j,i) 只保留一次。
pair_table = find_strict_mutual_nearest_pairs(
    analysis_points,
    roi_class_selection=TASK2_ROI_CLASS_SELECTION,
    pair_mode=TASK2_PAIR_MODE,
    source_class_ids=TASK2_SOURCE_CLASS_IDS,
    target_class_ids=TASK2_TARGET_CLASS_IDS,
    max_pair_distance_px=TASK2_MAX_PAIR_DISTANCE_PX,
    max_pair_distance_A=TASK2_MAX_PAIR_DISTANCE_A,
)

# 2) 将 valid pair center 沿 projection axis 投影为一维坐标，再按 line_tolerance 分组。
# TASK2_LINE_INDEX_MODE='global' 时会额外生成 global_line_id：
# - line_id：每个 ROI 内部的局部 line 编号；
# - global_line_id：整张图统一 projection 坐标系下的 line 编号，用于跨 ROI 对齐作图。
pair_line_table, line_grouping_summary = assign_pair_center_lines_by_projection(
    pair_table,
    projection_vector=TASK2_PROJECTION_VECTOR,
    line_tolerance_px=TASK2_LINE_TOLERANCE_PX,
    min_pairs_per_line=TASK2_MIN_PAIRS_PER_LINE,
    line_index_mode=TASK2_LINE_INDEX_MODE,
    pixel_to_nm=PIXEL_TO_NM,
)

# 3) 每条 line 统计 pair distance 的 median 和 Q1-Q3 IQR。
pair_line_summary_table = summarize_pair_lines_median_iqr(pair_line_table)

# Task 2 不把 pair center 合并到 all_analysis_points 作为后续任务输入；
# 这里保持 all_analysis_points 稳定为 atom 点表。
pair_center_points = pd.DataFrame()
all_analysis_points = analysis_points.copy()

display(pair_table.head())
display(line_grouping_summary)
display(pair_line_summary_table)
""",
        ),
        md(
            "task2-fig-md",
            """
## 13. Task 2: pair/line figures

这里绘制 Task 2 的正式图：pair overlay、pair center line assignment、projection spacing QC histogram，以及 line_id vs pair distance median + IQR errorbar。
""",
        ),
        code(
            "task2-figures",
            """
# pair overlay：原图上画 pair segment 和 pair center，用于检查配对是否合理。
fig, ax = plot_pair_overlay(image, pair_table, rois=ROIS, style=FIG_STYLE, title='Task 2 strict mutual nearest pairs')
figures['task2_pair_overlay'] = fig
display(fig)
plt.close(fig)

# line assignment：不同 line 的 pair center 用不同颜色显示，用于检查 line 分组。
fig, ax = plot_pair_center_line_assignment(image, pair_line_table, rois=ROIS, style=FIG_STYLE)
figures['task2_pair_center_line_assignment'] = fig
display(fig)
plt.close(fig)

# projection spacing histogram：显示相邻 pair center projection gap，用于判断 line_tolerance 是否合适。
# 有像素标定时横坐标使用 Å；没有标定时会 warning 并回退到 px。
fig, ax = plot_projection_spacing_histogram(pair_line_table, style=FIG_STYLE)
figures['task2_projection_spacing_histogram'] = fig
display(fig)
plt.close(fig)

# line median + IQR errorbar：Task 2 的主统计图。
# 如果 pair_line_summary_table 中有 global_line_id，横坐标会自动使用 Global line index，
# 这样不同 ROI 的同一全图 line 编号可以在同一个 x 位置比较。
fig, ax = plot_pair_line_distance_errorbar(pair_line_summary_table, pair_line_table, style=FIG_STYLE)
figures['task2_pair_line_distance_iqr'] = fig
display(fig)
plt.close(fig)
""",
        ),
        md(
            "task2-export-md",
            """
## 14. Task 2: export Excel

将 pair table、valid/invalid pair、line summary、projection axis 和 line grouping summary 写入独立 Excel 文件。
""",
        ),
        code(
            "task2-export",
            """
# Task 2 独立 Excel 导出：包含 valid/invalid pair、line summary 和 projection axis 配置。
task2_excel = export_task2_excel(
    output_dirs,
    pair_table=pair_line_table,
    pair_line_summary_table=pair_line_summary_table,
    task2_config=task2_config_table,
    projection_axis_table=task2_projection_table,
    line_grouping_summary=line_grouping_summary,
)
excel_exports['task2'] = task2_excel
display(task2_excel)
""",
        ),
        md(
            "task3-moved-md",
            """
## 15. Cropped group-centroid analysis moved to notebook 03

原 02 步骤 15-18 的 class group 几何中心分析已经迁移到 `03_Cropped_group_centroid_analysis.ipynb`。新的 03 workflow 会先在原图上选择 crop ROI、裁剪图像并保留 ROI 内原子，再在裁剪图中选择 measurement ROI，最后输出只含几何中心、箭头、group legend 和真实 nm scalebar 的图。
""",
        ),
        md(
            "final-export-md",
            """
## 16. Final export: tables / figures / configs / manifest

这个最终导出 cell 会统一保存所有 task 的 CSV 表、正式 figure、配置 JSON、Excel 导出记录和 session checkpoint，并写入 notebook02 manifest。
""",
        ),
        code(
            "final-export",
            """
# notebook02_tables：最终统一导出的 CSV 表。
# 每个任务的核心表都放在这里，final export 会写入 tables 目录并登记到 manifest。
notebook02_tables = {
    'analysis_points': analysis_points,
    'roi_table': roi_table,
    'roi_summary_table': roi_summary_table,
    'basis_vector_table': basis_vector_table,
    'roi_basis_table': roi_basis_table,
    'period_segment_table': period_segment_table,
    'period_summary_table': period_summary_table,
    'cell_table': cell_table,
    'strain_reference_table': strain_reference_table,
    'pair_table': pair_line_table,
    'pair_line_summary_table': pair_line_summary_table,
}

# notebook02_configs：最终统一导出的配置 JSON。
# 这里只记录关键运行参数；每个任务的完整参数还会写入各自 Excel。
notebook02_configs = {
    'notebook02_config': {
        'source_table': SOURCE_TABLE,
        'use_keep_only': USE_KEEP_ONLY,
        'image_channel': context['image_channel'],
        'image_key': context['image_key'],
        'basis_mode': BASIS_MODE,
        'basis_roles': BASIS_ROLES,
        'task1A_class_group_mode': TASK1A_CLASS_GROUP_MODE,
        'task2_pair_mode': TASK2_PAIR_MODE,
        'cropped_group_centroid_note': 'moved to notebook03',
    }
}

# export_notebook02_results 会统一保存：
# - CSV tables；
# - figures 中收集的正式图；
# - configs JSON；
# - excel_exports 中记录的 Task 1A / Task 1B / Task 2 Excel；
# - session checkpoint；
# - manifest.json。
manifest = export_notebook02_results(
    session=session,
    output_dirs=output_dirs,
    tables=notebook02_tables,
    figures=figures,
    configs=notebook02_configs,
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
    target = Path(__file__).resolve().parents[1] / "notebooks" / "02_Simple_quantitative_spacing_analysis.ipynb"
    target.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
