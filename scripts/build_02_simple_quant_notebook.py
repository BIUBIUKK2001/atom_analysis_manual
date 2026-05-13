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
# 02 多 ROI 简单间距与周期向量定量分析
# ROI-resolved simple spacing and periodic-vector analysis

这个 notebook 承接 `01_Findatom.ipynb` 保存的 refined / curated 原子柱坐标。它只做泛体系的坐标几何定量：多 ROI、多 class、用户指定 basis vector、周期向量匹配、pair center 点集、line guide 和统一的 `measurement_segments` 输出表。

边界：这里不做 local affine strain，不构建 reference lattice strain tensor，不做 vPCF，不做 phase/domain 自动识别，不做 polarization / vacancy 等材料专用解释，不做连续插值 heatmap，也不做自动晶格识别。所有方向和周期都由用户显式给出。
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
import pandas as pd
from IPython.display import display

from em_atom_workbench import (
    AnalysisROI,
    BasisVectorSpec,
    LineGuideTask,
    NearestForwardTask,
    PairSegmentTask,
    PeriodicVectorTask,
    combine_analysis_points,
    make_pair_center_points,
    pick_basis_vectors_with_napari,
    pick_rois_with_napari,
    resolve_basis_vector_specs,
)
from em_atom_workbench.notebook_workflows import (
    export_simple_quant_v2_analysis,
    initialize_simple_quant_v2_analysis,
    run_simple_quant_measurements,
)
from em_atom_workbench.simple_quant_plotting import (
    plot_line_guides_on_image,
    plot_measurement_segments_on_image,
    plot_spacing_histogram,
    plot_spacing_profile,
)

plt.rcParams['figure.dpi'] = 140
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
""",
        ),
        md(
            "active-session-md",
            """
## 1. 读取 active session 参数

02 是 01 的 downstream notebook，不会在找不到 active session 时自动生成 synthetic demo。如果这里报错，请先运行 01 并保存 `_active_session.pkl`，或手动填写 `SESSION_PATH`。
""",
        ),
        code(
            "active-session-parameters",
            """
# RESULT_ROOT：01 notebook 保存 active session 和 02 输出结果的根目录。
# 默认是项目根目录下的 results；如果 01 使用了别的结果目录，这里要保持一致。
RESULT_ROOT = PROJECT_ROOT / 'results'
RESULT_ROOT.mkdir(exist_ok=True)

# SESSION_PATH：
# - None：读取 RESULT_ROOT / '_active_session.pkl'，这是 01 默认写入的位置。
# - pathlib.Path 或字符串：读取你指定的 checkpoint，例如 results/.../session.pkl。
# 如果 active session 不存在，本 notebook 会直接报清晰错误，不会生成演示数据。
SESSION_PATH = None

active_path = RESULT_ROOT / '_active_session.pkl' if SESSION_PATH is None else Path(SESSION_PATH)
print(f'active session target: {active_path}')
""",
        ),
        md("input-md", "## 2. 输入 session / source table / image 参数"),
        code(
            "input-parameters",
            """
# SOURCE_TABLE：选择 01 输出的哪张坐标表作为分析起点。
# - 'curated'：默认，优先使用人工复核后的最终点；最适合正式定量。
# - 'refined'：使用精修后的点但不一定经过最终 keep/class 复核；适合快速检查。
# - 'candidate'：使用候选点；通常只用于诊断，不建议正式测量。
SOURCE_TABLE = 'curated'

# USE_KEEP_ONLY：如果坐标表里有 keep 列，是否只保留 keep == True 的点。
# 默认 True，适合正式分析；如果你想检查被剔除点对结果的影响，可改 False。
USE_KEEP_ONLY = True

# IMAGE_CHANNEL：叠图使用哪个通道。
# None 表示使用 session.primary_channel；多通道数据可改成具体通道名。
IMAGE_CHANNEL = None

# IMAGE_KEY：叠图使用 raw 还是 processed 图像。
# - 'raw'：默认，坐标通常在全局 raw 像素坐标中，适合最终展示。
# - 'processed'：用于检查预处理图像上的位置，但要注意 processed origin。
IMAGE_KEY = 'raw'
""",
        ),
        md("roi-parameters-md", "## 3. ROI 选择参数"),
        code(
            "roi-parameters",
            """
# OPEN_ROI_PICKER：是否打开 napari 让用户画多个 ROI。
# 默认 True，推荐首次分析时使用。关闭 napari 后，每个 polygon/rectangle 会转成一个 AnalysisROI。
# 如果在无 GUI 环境运行，设为 False，并在 ROIS 中手动写 polygon。
OPEN_ROI_PICKER = True

# ROIS：手动 ROI 参数。None 表示如果不打开 picker，就使用 global 全图 ROI。
# polygon_xy_px 使用图像全局像素坐标 (x, y)，不要求首尾闭合；计算时会自动闭合。
# ROI 可以重叠，同一个点落入多个 ROI 时会生成多行 analysis_points，scope_id 会区分。
ROIS = None

# DEFAULT_ROI_PREFIX：napari 画 ROI 后自动命名的前缀。
# 只影响显示和导出表，不影响计算逻辑；你可以后续在 ROIS 中手动改 roi_name。
DEFAULT_ROI_PREFIX = 'ROI'
""",
        ),
        md("roi-run-md", "## 4. 运行 ROI 选择或使用参数 ROI"),
        code(
            "roi-selection",
            """
roi_context = initialize_simple_quant_v2_analysis(
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
session = roi_context['session']
image = roi_context['image']
output_dirs = roi_context['output_dirs']

if OPEN_ROI_PICKER:
    ROIS = pick_rois_with_napari(
        session,
        roi_context['analysis_points'],
        image_channel=roi_context['image_channel'],
        image_key=IMAGE_KEY,
        default_roi_prefix=DEFAULT_ROI_PREFIX,
    )
elif ROIS is None:
    ROIS = [AnalysisROI(roi_id='global', roi_name='global', polygon_xy_px=None, color='#ff9f1c')]

display(pd.DataFrame([roi.__dict__ for roi in ROIS]))
""",
        ),
        md("point-filter-md", "## 5. point set / class filter 参数"),
        code(
            "point-filter-parameters",
            """
# CLASS_FILTER：按 class_name 保留 atom 点。
# None 表示不过滤；例如 ('class_0', 'class_1') 只分析这两类。
# 如果 01 中你给 class 起了物理名字，也可以直接填那些 class_name。
CLASS_FILTER = None

# CLASS_ID_FILTER：按 class_id 保留 atom 点。
# None 表示不过滤；例如 (0, 1) 只分析 class_id 为 0 和 1 的点。
# 如果同时设置 CLASS_FILTER 和 CLASS_ID_FILTER，二者会叠加，点必须同时满足。
CLASS_ID_FILTER = None

# POINT_SET_ATOMS：普通原子柱点集名称，保持 'atoms' 即可。
# 后面生成 pair centers 时，会新增 point_set='pair_centers'，并可用同样任务逻辑分析。
POINT_SET_ATOMS = 'atoms'
""",
        ),
        md("prepare-points-md", "## 6. 准备 analysis_points 并预览 class-colored atoms"),
        code(
            "prepare-analysis-points",
            """
context = initialize_simple_quant_v2_analysis(
    session_path=SESSION_PATH,
    result_root=RESULT_ROOT,
    source_table=SOURCE_TABLE,
    use_keep_only=USE_KEEP_ONLY,
    class_filter=CLASS_FILTER,
    class_id_filter=CLASS_ID_FILTER,
    rois=ROIS,
    image_channel=IMAGE_CHANNEL,
    image_key=IMAGE_KEY,
)
session = context['session']
analysis_points = context['analysis_points']
roi_table = context['roi_table']
image = context['image']
output_dirs = context['output_dirs']
figures = {}

for table in context['summary_tables'].values():
    display(table)
display(analysis_points.head())

fig, image_ax, side_ax = plot_measurement_segments_on_image(
    image,
    analysis_points,
    pd.DataFrame(),
    show_side_panel=True,
    show_axes=False,
    title='02A class-colored analysis points',
)
figures['02A_analysis_points_preview'] = fig
display(fig)
plt.close(fig)
""",
        ),
        md("basis-parameters-md", "## 7. basis vector 选择参数"),
        code(
            "basis-parameters",
            """
# OPEN_BASIS_VECTOR_PICKER：是否打开 napari 让用户为每个 basis 放两个点。
# 默认 True。每个 basis 的两个点定义完整 vector，长度会保留下来作为周期长度。
# 如果你已经知道 vector_px，设为 False 并修改 BASIS_VECTOR_SPECS。
OPEN_BASIS_VECTOR_PICKER = True

# BASIS_NAMES：需要选择的 basis 名称。
# 常见二维晶格可用 ('a', 'b')；如果只分析一个方向，也可以只保留 ('a',)。
BASIS_NAMES = ('a', 'b')

# SNAP_BASIS_POINTS_TO_NEAREST_POINTS：napari 选点后是否吸附到最近 atom/pair center。
# True 可避免手点偏差，适合从原子柱中心定义周期；False 可保留任意手动矢量。
SNAP_BASIS_POINTS_TO_NEAREST_POINTS = True

# BASIS_VECTOR_SPECS：非交互 basis 参数。
# vector_px=(dx, dy) 会直接定义完整周期矢量，不会归一化。
# from_atom_ids / from_point_ids / from_xy_px 也可以定义矢量；period_px 默认等于矢量长度。
BASIS_VECTOR_SPECS = [
    BasisVectorSpec(name='a', vector_px=(10.0, 0.0)),
    BasisVectorSpec(name='b', vector_px=(0.0, 10.0)),
]
""",
        ),
        md("basis-run-md", "## 8. napari 选择 basis vector 或解析参数输入"),
        code(
            "basis-selection",
            """
if OPEN_BASIS_VECTOR_PICKER:
    BASIS_VECTOR_SPECS = pick_basis_vectors_with_napari(
        session,
        analysis_points,
        basis_names=BASIS_NAMES,
        image_channel=context['image_channel'],
        image_key=IMAGE_KEY,
        snap_to_nearest_points=SNAP_BASIS_POINTS_TO_NEAREST_POINTS,
        point_size=5.0,
    )

basis_vector_table = resolve_basis_vector_specs(analysis_points, BASIS_VECTOR_SPECS)
display(basis_vector_table)
""",
        ),
        md("basis-overlay-md", "## 9. basis vector overlay 检查"),
        code(
            "basis-overlay",
            """
fig, image_ax, side_ax = plot_measurement_segments_on_image(
    image,
    analysis_points,
    pd.DataFrame(),
    basis_vector_table=basis_vector_table,
    show_side_panel=True,
    show_axes=False,
    title='02B basis vector check',
)
figures['02B_basis_vector_check'] = fig
display(fig)
plt.close(fig)
""",
        ),
        md("measurement-parameters-md", "## 10. measurement task 参数"),
        code(
            "measurement-parameters",
            """
# MIN_PARALLEL_DISTANCE_PX：
# nearest-forward 测量时，target 必须在 source 的 basis 正方向上至少这么远。
# 默认 1 px 用于排除自身或几乎重合点；如果点有重复或很近的伪点，可适当调大。
MIN_PARALLEL_DISTANCE_PX = 1.0

# MAX_PARALLEL_DISTANCE_PX：
# nearest-forward 的最大正向距离。None 表示不限制。
# 规则晶格中可设为略大于 basis length，避免跨周期；畸变大时可放宽或保持 None。
MAX_PARALLEL_DISTANCE_PX = None

# PERPENDICULAR_TOLERANCE_PX：
# nearest-forward 允许 target 偏离 basis 直线的横向距离。
# 太小会漏掉畸变或弯曲行列；太大会串到邻近行列。规则数据可从 2-3 px 起试。
PERPENDICULAR_TOLERANCE_PX = 3.0

# ANGLE_TOLERANCE_DEG：
# source-target 连线相对 basis 方向的角度误差上限。
# 太小会漏掉真实畸变；太大会允许明显偏斜的错误 pair。常用 10-20 deg。
ANGLE_TOLERANCE_DEG = 15.0

# MATCH_RADIUS_FRACTION：
# periodic-vector 匹配时允许目标点偏离 expected_position = source + basis_vector 的程度。
# 实际半径 = basis_length_px * MATCH_RADIUS_FRACTION。
# 太小：畸变晶格中真实周期点可能匹配不到；太大：可能跨到相邻链或错误原子。
# 规则晶格建议 0.20-0.30；畸变较大时可试 0.35-0.45。
MATCH_RADIUS_FRACTION = 0.30

# PAIR_MAX_DISTANCE_PX：
# class-to-class pair 最近邻的最大允许距离。
# None 表示不限制；正式使用时建议设为略大于预期 pair 距离，防止跨远邻。
PAIR_MAX_DISTANCE_PX = None

available_class_ids = tuple(int(v) for v in sorted(analysis_points['class_id'].dropna().unique())) if 'class_id' in analysis_points else ()
PAIR_SOURCE_CLASS_IDS = (available_class_ids[0],) if len(available_class_ids) >= 2 else None
PAIR_TARGET_CLASS_IDS = (available_class_ids[1],) if len(available_class_ids) >= 2 else None

NEAREST_FORWARD_TASKS = [
    NearestForwardTask(
        name='a_nearest_forward_atoms',
        basis='a',
        point_set='atoms',
        min_parallel_distance_px=MIN_PARALLEL_DISTANCE_PX,
        max_parallel_distance_px=MAX_PARALLEL_DISTANCE_PX,
        perpendicular_tolerance_px=PERPENDICULAR_TOLERANCE_PX,
        angle_tolerance_deg=ANGLE_TOLERANCE_DEG,
    )
]

PERIODIC_VECTOR_TASKS = [
    PeriodicVectorTask(
        name='a_periodic_atoms',
        basis='a',
        point_set='atoms',
        match_radius_fraction=MATCH_RADIUS_FRACTION,
        one_to_one=True,
    )
]

PAIR_SEGMENT_TASKS = []
if PAIR_SOURCE_CLASS_IDS is not None and PAIR_TARGET_CLASS_IDS is not None:
    PAIR_SEGMENT_TASKS.append(
        PairSegmentTask(
            name='nearest_class_pair_atoms',
            point_set='atoms',
            source_class_ids=PAIR_SOURCE_CLASS_IDS,
            target_class_ids=PAIR_TARGET_CLASS_IDS,
            max_distance_px=PAIR_MAX_DISTANCE_PX,
            unique_pairs=True,
            create_pair_centers=True,
            pair_center_class_name='pair_center',
        )
    )
else:
    print('PairSegmentTask example skipped: need at least two class_id values. Edit PAIR_SOURCE_CLASS_IDS / PAIR_TARGET_CLASS_IDS if needed.')

MEASUREMENT_TASKS = [*NEAREST_FORWARD_TASKS, *PERIODIC_VECTOR_TASKS, *PAIR_SEGMENT_TASKS]
""",
        ),
        md("run-atom-measurements-md", "## 11. 运行 atom measurement tasks，得到 measurement_segments"),
        code(
            "run-atom-measurements",
            """
atom_result = run_simple_quant_measurements(
    analysis_points,
    basis_vector_table,
    MEASUREMENT_TASKS,
)
atom_measurement_segments = atom_result['measurement_segments']
pair_center_points = atom_result['pair_center_points']
atom_summaries = atom_result['summaries']

display(atom_summaries)
display(atom_measurement_segments.head())
if not pair_center_points.empty:
    display(pair_center_points.head())
""",
        ),
        md("pair-center-md", "## 12. 可选：由 pair segments 生成 pair_centers"),
        code(
            "pair-center-generation",
            """
# CREATE_PAIR_CENTERS：
# True 表示把 pair segment 的中点变成 point_set='pair_centers'。
# pair_centers 可以像 atom 一样参与 spacing、periodic-vector 和 line guide 分析。
# 如果 PAIR_SEGMENT_TASKS 中已经 create_pair_centers=True，这里通常不需要再手动生成。
CREATE_PAIR_CENTERS = True

if CREATE_PAIR_CENTERS and pair_center_points.empty:
    pair_segments_for_centers = atom_measurement_segments.loc[
        atom_measurement_segments['task_type'].isin(['explicit_pair', 'nearest_class_pair'])
    ].copy()
    pair_center_points = make_pair_center_points(pair_segments_for_centers, class_name='pair_center')

all_analysis_points = combine_analysis_points(analysis_points, pair_center_points)
print(f'atom rows: {len(analysis_points)}')
print(f'pair center rows: {len(pair_center_points)}')
display(pair_center_points.head() if not pair_center_points.empty else pd.DataFrame())
""",
        ),
        md("pair-center-measurements-md", "## 13. 可选：对 pair_centers 运行 measurement tasks"),
        code(
            "pair-center-measurements",
            """
# RUN_PAIR_CENTER_MEASUREMENTS：
# True 时对 pair_centers 再跑一个 periodic-vector 示例任务。
# 如果没有生成 pair_centers，本格会跳过，不会报错。
RUN_PAIR_CENTER_MEASUREMENTS = True

PAIR_CENTER_MEASUREMENT_TASKS = [
    PeriodicVectorTask(
        name='a_periodic_pair_centers',
        basis='a',
        point_set='pair_centers',
        match_radius_fraction=MATCH_RADIUS_FRACTION,
        one_to_one=True,
    )
]

if RUN_PAIR_CENTER_MEASUREMENTS and not pair_center_points.empty:
    pair_center_result = run_simple_quant_measurements(
        all_analysis_points,
        basis_vector_table,
        PAIR_CENTER_MEASUREMENT_TASKS,
    )
    pair_center_measurement_segments = pair_center_result['measurement_segments']
    pair_center_summaries = pair_center_result['summaries']
else:
    pair_center_measurement_segments = pd.DataFrame()
    pair_center_summaries = pd.DataFrame()

display(pair_center_summaries)
display(pair_center_measurement_segments.head())
""",
        ),
        md("line-parameters-md", "## 14. line guide 参数"),
        code(
            "line-guide-parameters",
            """
# LINE_GROUP_AXIS：
# 't' 表示按垂直于 basis 的投影聚成一条条沿 basis 方向的 row。
# 's' 表示按 basis 方向投影聚成一条条垂直 basis 的 column。
LINE_GROUP_AXIS = 't'

# LINE_TOLERANCE_PX：
# 一维投影分组的容差。太小会把同一行拆碎；太大会合并相邻行。
# 建议从原子定位误差的 2-3 倍开始试，例如 2-4 px。
LINE_TOLERANCE_PX = 3.0

# MIN_POINTS_PER_LINE：
# 一条 line 至少要包含多少点才保留。
# 调大可去掉短碎线；调小可检查边缘或小 ROI，但可能引入不稳定 line。
MIN_POINTS_PER_LINE = 4

# GENERATE_LINE_CONSECUTIVE_SEGMENTS：
# True 时 line guide 还会输出 line 内相邻点的 measurement_segments。
# 如果你只想画 guide，不想把 line spacing 合入 measurement_segments，可设 False。
GENERATE_LINE_CONSECUTIVE_SEGMENTS = True

# MAX_IN_LINE_GAP_PX：
# line 内连续点距离超过该值时标记 gap_exceeds_max。
# None 表示不限制；若存在缺点或畴边界，可设为略大于预期行内间距。
MAX_IN_LINE_GAP_PX = None

LINE_GUIDE_TASKS = [
    LineGuideTask(
        name='a_rows_by_t',
        basis='a',
        point_set='atoms',
        group_axis=LINE_GROUP_AXIS,
        line_tolerance_px=LINE_TOLERANCE_PX,
        min_points_per_line=MIN_POINTS_PER_LINE,
        generate_consecutive_segments=GENERATE_LINE_CONSECUTIVE_SEGMENTS,
        max_in_line_gap_px=MAX_IN_LINE_GAP_PX,
    )
]
""",
        ),
        md("line-run-md", "## 15. 生成 line_guides 和可选 line_consecutive segments"),
        code(
            "line-guide-run",
            """
line_result = run_simple_quant_measurements(
    all_analysis_points,
    basis_vector_table,
    LINE_GUIDE_TASKS,
)
line_guides = line_result['line_guides']
line_measurement_segments = line_result['measurement_segments']
line_summaries = line_result['summaries']

display(line_guides.head())
display(line_summaries)
display(line_measurement_segments.head())
""",
        ),
        md("segment-overlay-md", "## 16. measurement segments 粗彩色短线叠加图"),
        code(
            "segment-overlay-parameters",
            """
# SHOW_IMAGE_AXES：默认 False，隐藏 x/y 坐标轴，避免把图像坐标误解为 basis 坐标。
SHOW_IMAGE_AXES = False

# SHOW_SIDE_PANEL：默认 True，class legend、ROI legend、task legend 和统计摘要都放图外。
SHOW_SIDE_PANEL = True

# SEGMENT_COLOR_BY：
# - 'task'：默认，不同测量任务用不同颜色，适合检查多任务叠加。
# - 'roi'：不同 ROI 用不同颜色，适合比较空间区域。
# - 'class_pair'：按 source class -> target class 上色。
# - 'fixed'：所有线段一个固定颜色。
# - 'value'：按数值列 colormap，仅在你明确需要连续色条时使用。
SEGMENT_COLOR_BY = 'task'

# SEGMENT_LINEWIDTH：measurement segment 线宽。
# 默认 3.0，用粗短线突出实际测量的 pair；点很密时可降到 1.5-2.0。
SEGMENT_LINEWIDTH = 3.0

# SEGMENT_ALPHA：measurement segment 透明度。
# 默认 0.90；叠加太密时可降到 0.45-0.70。
SEGMENT_ALPHA = 0.90

# SHOW_VALUE_COLORBAR：只有 SEGMENT_COLOR_BY='value' 时才建议打开。
# 默认 False，避免恢复成旧版那种默认色块/色条主导的图。
SHOW_VALUE_COLORBAR = False
""",
        ),
        code(
            "segment-overlay",
            """
segment_tables = [
    atom_measurement_segments,
    pair_center_measurement_segments,
    line_measurement_segments,
]
measurement_segments = pd.concat(
    [table for table in segment_tables if table is not None and not table.empty],
    ignore_index=True,
) if any(table is not None and not table.empty for table in segment_tables) else pd.DataFrame()

fig, image_ax, side_ax = plot_measurement_segments_on_image(
    image,
    all_analysis_points,
    measurement_segments,
    basis_vector_table=basis_vector_table,
    line_guides=line_guides,
    color_by=SEGMENT_COLOR_BY,
    linewidth=SEGMENT_LINEWIDTH,
    alpha=SEGMENT_ALPHA,
    show_value_colorbar=SHOW_VALUE_COLORBAR,
    show_side_panel=SHOW_SIDE_PANEL,
    show_axes=SHOW_IMAGE_AXES,
    title='02C measurement segments',
)
figures['02C_measurement_segments'] = fig
display(fig)
plt.close(fig)
""",
        ),
        md("line-overlay-md", "## 17. line guides 直线叠加图，编号在图外"),
        code(
            "line-overlay",
            """
# LINE_LABEL_MODE：
# 'all' 显示全部 line id；'every_n' 隔几个显示；'selected' 只显示 selected_line_ids；'none' 不显示。
# label 坐标由 line guide 计算在图外边缘，绘图时 clip_on=False。
LINE_LABEL_MODE = 'all'
LINE_LABEL_EVERY_N = 2
SELECTED_LINE_IDS = None

fig, image_ax, side_ax = plot_line_guides_on_image(
    image,
    all_analysis_points,
    line_guides,
    basis_vector_table=basis_vector_table,
    label_mode=LINE_LABEL_MODE,
    label_every_n=LINE_LABEL_EVERY_N,
    selected_line_ids=SELECTED_LINE_IDS,
    label_outside=True,
    show_side_panel=SHOW_SIDE_PANEL,
    show_axes=SHOW_IMAGE_AXES,
    title='02D line guides',
)
figures['02D_line_guides'] = fig
display(fig)
plt.close(fig)
""",
        ),
        md("summary-md", "## 18. histogram / profile / ROI summary"),
        code(
            "summary-plots",
            """
VALUE_COLUMN = 'distance_pm' if 'distance_pm' in measurement_segments and measurement_segments['distance_pm'].notna().any() else 'distance_px'
summary_tables = [
    atom_summaries,
    pair_center_summaries,
    line_summaries,
]
summaries = pd.concat([table for table in summary_tables if table is not None and not table.empty], ignore_index=True) if any(table is not None and not table.empty for table in summary_tables) else pd.DataFrame()

fig, ax = plot_spacing_histogram(
    measurement_segments,
    value_column=VALUE_COLUMN,
    group_column='task_name',
    title='02E measurement histogram',
)
figures['02E_measurement_histogram'] = fig
display(fig)
plt.close(fig)

if 'parallel_distance_px' in measurement_segments:
    fig, ax = plot_spacing_profile(
        measurement_segments,
        x_column='parallel_distance_px',
        y_column=VALUE_COLUMN,
        group_column='roi_id',
        title='02F spacing profile by ROI',
    )
    figures['02F_spacing_profile_by_roi'] = fig
    display(fig)
    plt.close(fig)

display(roi_table)
display(summaries)
""",
        ),
        md("export-md", "## 19. 导出 tables / figures / configs / manifest / session checkpoint"),
        code(
            "export",
            """
config = {
    'source_table': SOURCE_TABLE,
    'use_keep_only': USE_KEEP_ONLY,
    'class_filter': CLASS_FILTER,
    'class_id_filter': CLASS_ID_FILTER,
    'image_channel': context['image_channel'],
    'image_key': context['image_key'],
    'rois': ROIS,
    'basis_vector_specs': BASIS_VECTOR_SPECS,
    'measurement_tasks': MEASUREMENT_TASKS,
    'pair_center_measurement_tasks': PAIR_CENTER_MEASUREMENT_TASKS,
    'line_guide_tasks': LINE_GUIDE_TASKS,
    'segment_color_by': SEGMENT_COLOR_BY,
    'segment_linewidth': SEGMENT_LINEWIDTH,
    'segment_alpha': SEGMENT_ALPHA,
    'show_side_panel': SHOW_SIDE_PANEL,
}

manifest = export_simple_quant_v2_analysis(
    session=session,
    output_dirs=output_dirs,
    analysis_points=all_analysis_points,
    roi_table=roi_table,
    basis_vector_table=basis_vector_table,
    measurement_segments=measurement_segments,
    pair_center_points=pair_center_points,
    line_guides=line_guides,
    summaries=summaries,
    figures=figures,
    config=config,
)
display(manifest)
""",
        ),
        md(
            "optional-hzo-template",
            """
## 20. Optional HZO-like 参数模板（不自动运行）

下面只是“如何把用户已命名类别填进任务”的模板。请先确认 01 中的 `class_id` / `class_name`，再复制到参数格中修改。本 notebook 的核心函数、字段名、默认逻辑都不假设 HZO 或任何材料体系。

```python
# 示例：把 class_id 替换成你的实际类别后再复制使用。
# NEAREST_FORWARD_TASKS = [
#     NearestForwardTask(name='a_forward_selected', basis='a', source_class_ids=(0,), target_class_ids=(0,))
# ]
# PAIR_SEGMENT_TASKS = [
#     PairSegmentTask(name='selected_pair_centers', source_class_ids=(0,), target_class_ids=(1,), max_distance_px=8.0, create_pair_centers=True)
# ]
# PERIODIC_VECTOR_TASKS = [
#     PeriodicVectorTask(name='a_periodic_selected', basis='a', class_ids=(0,), match_radius_fraction=0.30)
# ]
```
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
