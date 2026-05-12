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
# 02 Simple quantitative spacing analysis

这个 notebook 承接 `01_Findatom.ipynb` 保存的 refined / curated 原子柱坐标，做简单、可控、泛体系的坐标几何定量分析：指定方向最近邻间距、指定原子对距离、按 s/t 投影分组的行列内连续间距与 line width，以及色块式 colormap overlay。

边界：本 notebook 不构建 reference lattice，不计算 local affine strain，不做 vPCF，不做 phase/domain 自动识别，也不做材料专用物理解释。
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
from IPython.display import display

from em_atom_workbench import (
    DirectionSpec,
    DirectionalSpacingTask,
    LineGroupingTask,
    PairDistanceTask,
    pick_direction_vectors_with_napari,
    resolve_direction_specs,
)
from em_atom_workbench.notebook_workflows import (
    export_simple_quant_analysis,
    initialize_simple_quant_analysis,
    run_directional_spacing_analysis,
    run_line_spacing_analysis,
    run_pair_distance_analysis,
)
from em_atom_workbench.simple_quant_plotting import (
    plot_atom_id_overlay,
    plot_direction_overlay,
    plot_line_assignment_overlay,
    plot_line_spacing_blocks_on_image,
    plot_line_width_summary,
    plot_pair_blocks_on_image,
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
## 1. 读取 active session

默认读取 `results/_active_session.pkl`。如果你要读取某个手动 checkpoint，把 `SESSION_PATH` 改成对应 `.pkl` 路径；否则保持 `None`。
""",
        ),
        code(
            "active-session-parameters",
            """
# RESULT_ROOT：01 notebook 保存 active session 的结果目录。
RESULT_ROOT = PROJECT_ROOT / 'results'
RESULT_ROOT.mkdir(exist_ok=True)

# SESSION_PATH：None 表示读取 RESULT_ROOT / '_active_session.pkl'。
SESSION_PATH = None

active_path = RESULT_ROOT / '_active_session.pkl'
print(f'active session: {active_path}')
""",
        ),
        md(
            "general-parameters-md",
            """
## 2. 02 总参数

这一格只控制坐标来源、keep 过滤、类别过滤、ROI 和用于叠图的图像通道。
""",
        ),
        code(
            "general-parameters",
            """
# SOURCE_TABLE：坐标来源，可选 'curated' / 'refined' / 'candidate'。
SOURCE_TABLE = 'curated'

# USE_KEEP_ONLY：如果表中有 keep 列，True 表示只分析 keep == True 的点。
USE_KEEP_ONLY = True

# IMAGE_CHANNEL：用于叠图的通道；None 表示使用 session.primary_channel。
IMAGE_CHANNEL = None

# IMAGE_KEY：用于叠图的图像，可选 'raw' 或 'processed'。
IMAGE_KEY = 'raw'

# CLASS_FILTER：按 class_name 保留点；None 表示不过滤，例如 ('class_0', 'class_1')。
CLASS_FILTER = None

# CLASS_ID_FILTER：按 class_id 保留点；None 表示不过滤，例如 (0, 1)。
CLASS_ID_FILTER = None

# ROI：像素坐标范围 (x_min_px, x_max_px, y_min_px, y_max_px)；None 表示全图。
ROI = None
""",
        ),
        md(
            "prepare-md",
            """
## 3. 准备 quant_points

这一步从 active session 读取坐标表，统一成 02 使用的 `quant_points`。如果 session 没有像素标定，nm/pm 相关列会保留为 NaN，px 结果仍可继续分析。
""",
        ),
        code(
            "prepare-quant-points",
            """
context = initialize_simple_quant_analysis(
    session_path=SESSION_PATH,
    result_root=RESULT_ROOT,
    source_table=SOURCE_TABLE,
    use_keep_only=USE_KEEP_ONLY,
    class_filter=CLASS_FILTER,
    class_id_filter=CLASS_ID_FILTER,
    roi=ROI,
    image_channel=IMAGE_CHANNEL,
    image_key=IMAGE_KEY,
)
session = context['session']
quant_points = context['quant_points']
image = context['image']
output_dirs = context['output_dirs']
figures = {}
tables = {'quant_points': quant_points}

for table in context['summary_tables'].values():
    display(table)
display(quant_points.head())
""",
        ),
        md(
            "atom-preview-md",
            """
## 4. atom_id / class preview

用 atom_id 叠图检查坐标和类别是否来自你想分析的点集。点很多时会自动隐藏文字标签。
""",
        ),
        code(
            "atom-preview",
            """
fig, ax = plot_atom_id_overlay(
    image,
    quant_points,
    label_column='atom_id',
    max_labels=300,
    class_color=True,
    title='02A atom_id preview',
)
figures['02A_atom_id_overlay'] = fig
display(fig)
plt.close(fig)
""",
        ),
        md(
            "direction-parameters-md",
            """
## 5. 方向定义参数

优先用 napari 在图上为每个方向点两个点；如果不打开交互窗口，则使用 `DIRECTIONS` 中的显式参数方向。
""",
        ),
        code(
            "direction-parameters",
            """
# OPEN_DIRECTION_PICKER：True 时打开 napari，让用户为每个方向放两个点。
OPEN_DIRECTION_PICKER = True

# DIRECTION_NAMES：napari 里需要选择的方向名。
DIRECTION_NAMES = ('u', 'v')

# SNAP_DIRECTION_POINTS_TO_NEAREST_ATOMS：True 表示把 napari 选点吸附到最近 atom column。
SNAP_DIRECTION_POINTS_TO_NEAREST_ATOMS = True

# DIRECTIONS：非交互兜底方向；vector_px=(dx, dy)，坐标单位是 px。
DIRECTIONS = [
    DirectionSpec(name='u', vector_px=(1.0, 0.0)),
    DirectionSpec(name='v', vector_px=(0.0, 1.0)),
]
""",
        ),
        md(
            "resolve-directions-md",
            """
## 6. napari 交互选择方向或解析参数方向

每个方向只需要两个点。关闭 napari 后会生成 `direction_table`，后续所有 spacing / line grouping 都引用这里的方向名。
""",
        ),
        code(
            "resolve-directions",
            """
if OPEN_DIRECTION_PICKER:
    picked_directions = pick_direction_vectors_with_napari(
        session,
        quant_points,
        direction_names=DIRECTION_NAMES,
        image_channel=context['image_channel'],
        image_key=IMAGE_KEY,
        snap_to_nearest_atoms=SNAP_DIRECTION_POINTS_TO_NEAREST_ATOMS,
        point_size=5.0,
    )
else:
    picked_directions = DIRECTIONS

direction_table = resolve_direction_specs(quant_points, picked_directions)
tables['direction_table'] = direction_table
display(direction_table)
""",
        ),
        md(
            "direction-overlay-md",
            """
## 7. direction overlay 检查

检查方向箭头是否符合你的分析定义。若方向不对，回到上一格重新选点或修改 `DIRECTIONS`。
""",
        ),
        code(
            "direction-overlay",
            """
fig, ax = plot_direction_overlay(
    image,
    quant_points,
    direction_table,
    title='02B direction overlay',
)
figures['02B_direction_overlay'] = fig
display(fig)
plt.close(fig)
""",
        ),
        md(
            "directional-spacing-parameters-md",
            """
## 8. directional spacing 参数

对每个 source atom，沿指定方向找 forward 最近 target atom。`perpendicular_tolerance_px` 控制横向容许偏移，`angle_tolerance_deg` 控制方向角容许误差。
""",
        ),
        code(
            "directional-spacing-parameters",
            """
DIRECTIONAL_SPACING_TASKS = [
    DirectionalSpacingTask(
        name='u_forward_spacing_all',
        direction='u',
        source_class_ids=None,
        target_class_ids=None,
        perpendicular_tolerance_px=3.0,
        angle_tolerance_deg=15.0,
        min_parallel_distance_px=1.0,
    )
]
""",
        ),
        md(
            "run-directional-spacing-md",
            """
## 9. 运行 directional spacing
""",
        ),
        code(
            "run-directional-spacing",
            """
directional_result = run_directional_spacing_analysis(
    quant_points,
    direction_table,
    DIRECTIONAL_SPACING_TASKS,
)
directional_spacing_table = directional_result['directional_spacing_table']
directional_spacing_summary = directional_result['directional_spacing_summary']
tables['directional_spacing_table'] = directional_spacing_table
display(directional_spacing_summary)
display(directional_spacing_table.head())
""",
        ),
        md(
            "directional-spacing-blocks-md",
            """
## 10. directional spacing 色块图

默认把 source-target 连线画成矩形色块，颜色对应 spacing 值；不是插值 heatmap。
""",
        ),
        code(
            "directional-spacing-blocks",
            """
directional_value = 'distance_pm' if 'distance_pm' in directional_spacing_table and directional_spacing_table['distance_pm'].notna().any() else 'distance_px'
fig, ax = plot_pair_blocks_on_image(
    image,
    directional_spacing_table,
    value_column=directional_value,
    title='02C directional spacing blocks',
    colorbar_label=directional_value,
)
figures['02C_directional_spacing_blocks'] = fig
display(fig)
plt.close(fig)
""",
        ),
        md(
            "directional-spacing-histogram-md",
            """
## 11. directional spacing histogram
""",
        ),
        code(
            "directional-spacing-histogram",
            """
fig, ax = plot_spacing_histogram(
    directional_spacing_table,
    value_column=directional_value,
    group_column='measurement_name',
    title='02D directional spacing histogram',
)
figures['02D_directional_spacing_histogram'] = fig
display(fig)
plt.close(fig)
""",
        ),
        md(
            "pair-distance-parameters-md",
            """
## 12. pair distance 参数

可做 class-to-class 最近邻 pair，也可以用 `explicit_atom_pairs=((id1, id2), ...)` 指定原子对。
""",
        ),
        code(
            "pair-distance-parameters",
            """
PAIR_DISTANCE_TASKS = [
    PairDistanceTask(
        name='nearest_class0_class1',
        source_class_ids=(0,),
        target_class_ids=(1,),
        mode='nearest',
        max_distance_px=8.0,
        unique_pairs=True,
    )
]
""",
        ),
        md("run-pair-distance-md", "## 13. 运行 pair distance"),
        code(
            "run-pair-distance",
            """
pair_result = run_pair_distance_analysis(
    quant_points,
    direction_table,
    PAIR_DISTANCE_TASKS,
)
pair_distance_table = pair_result['pair_distance_table']
pair_distance_summary = pair_result['pair_distance_summary']
tables['pair_distance_table'] = pair_distance_table
display(pair_distance_summary)
display(pair_distance_table.head())
""",
        ),
        md("pair-distance-figures-md", "## 14. pair distance 色块图和 histogram"),
        code(
            "pair-distance-figures",
            """
pair_value = 'distance_pm' if 'distance_pm' in pair_distance_table and pair_distance_table['distance_pm'].notna().any() else 'distance_px'
fig, ax = plot_pair_blocks_on_image(
    image,
    pair_distance_table,
    value_column=pair_value,
    title='02E pair distance blocks',
    colorbar_label=pair_value,
)
figures['02E_pair_distance_blocks'] = fig
display(fig)
plt.close(fig)

fig, ax = plot_spacing_histogram(
    pair_distance_table,
    value_column=pair_value,
    group_column='pair_name',
    title='02F pair distance histogram',
)
figures['02F_pair_distance_histogram'] = fig
display(fig)
plt.close(fig)
""",
        ),
        md(
            "line-grouping-parameters-md",
            """
## 15. line grouping 参数

沿方向 u 计算投影 s 和垂直投影 t。`group_axis='t'` 常用于按 t 聚成 rows，`group_axis='s'` 常用于按 s 聚成 columns。
""",
        ),
        code(
            "line-grouping-parameters",
            """
LINE_GROUPING_TASKS = [
    LineGroupingTask(
        name='rows_group_by_t_along_u',
        direction='u',
        class_ids=None,
        group_axis='t',
        line_tolerance_px=3.0,
        min_atoms_per_line=4,
    ),
    LineGroupingTask(
        name='columns_group_by_s_along_u',
        direction='u',
        class_ids=None,
        group_axis='s',
        line_tolerance_px=3.0,
        min_atoms_per_line=4,
    ),
]
""",
        ),
        md("run-line-spacing-md", "## 16. 运行 line spacing"),
        code(
            "run-line-spacing",
            """
line_result = run_line_spacing_analysis(
    quant_points,
    direction_table,
    LINE_GROUPING_TASKS,
)
line_assignments = line_result['line_assignments']
line_spacing_table = line_result['line_spacing_table']
line_summary = line_result['line_summary']
tables['line_assignments'] = line_assignments
tables['line_spacing_table'] = line_spacing_table
tables['line_summary'] = line_summary
display(line_summary.head())
display(line_spacing_table.head())
""",
        ),
        md("line-spacing-blocks-md", "## 17. line spacing 色块图和 line assignment overlay"),
        code(
            "line-spacing-blocks",
            """
line_value = 'spacing_to_next_pm' if 'spacing_to_next_pm' in line_spacing_table and line_spacing_table['spacing_to_next_pm'].notna().any() else 'spacing_to_next_px'
fig, ax = plot_line_assignment_overlay(
    image,
    line_spacing_table,
    title='02G line assignment overlay',
)
figures['02G_line_assignment_overlay'] = fig
display(fig)
plt.close(fig)

fig, ax = plot_line_spacing_blocks_on_image(
    image,
    line_spacing_table,
    value_column=line_value,
    title='02H line spacing blocks',
    colorbar_label=line_value,
)
figures['02H_line_spacing_blocks'] = fig
display(fig)
plt.close(fig)
""",
        ),
        md("line-profile-md", "## 18. line spacing profile 和 line width summary"),
        code(
            "line-profile-width",
            """
fig, ax = plot_spacing_profile(
    line_spacing_table,
    x_column='sort_coord_px',
    y_column=line_value,
    group_column='line_id',
    title='02I line spacing profile',
)
figures['02I_line_spacing_profile'] = fig
display(fig)
plt.close(fig)

fig, ax = plot_line_width_summary(
    line_summary,
    title='02J line width summary',
)
figures['02J_line_width_summary'] = fig
display(fig)
plt.close(fig)
""",
        ),
        md(
            "export-md",
            """
## 19. 导出 tables / figures / configs / manifest / session checkpoint

导出目录固定为 `results/02_simple_quant/`。此步骤会在 session annotations 中记录 02 输出位置，并另存 `02_simple_quant_session.pkl`。
""",
        ),
        code(
            "export",
            """
config = {
    'source_table': SOURCE_TABLE,
    'use_keep_only': USE_KEEP_ONLY,
    'class_filter': CLASS_FILTER,
    'class_id_filter': CLASS_ID_FILTER,
    'roi': ROI,
    'image_channel': context['image_channel'],
    'image_key': context['image_key'],
    'directional_spacing_tasks': DIRECTIONAL_SPACING_TASKS,
    'pair_distance_tasks': PAIR_DISTANCE_TASKS,
    'line_grouping_tasks': LINE_GROUPING_TASKS,
}
manifest = export_simple_quant_analysis(
    session=session,
    output_dirs=output_dirs,
    tables=tables,
    figures=figures,
    config=config,
    direction_table=direction_table,
)
display(manifest)
""",
        ),
        md(
            "optional-template",
            """
## 20. Optional HZO-like 参数模板（不自动运行）

下面只是一个“如何按用户已命名类别填写参数”的模板，请先把类别名替换成你在 01 中人工确认的 `class_name` 或 `class_id`。本 notebook 的核心分析不假设这些名字，也不会把它们用于默认逻辑。

```python
# 示例：按你的类别命名替换 class_id / class_name 后再复制到参数格中使用。
# DirectionalSpacingTask(name='u_forward_selected', direction='u', source_class_ids=(0,), target_class_ids=(1,))
# PairDistanceTask(name='nearest_selected_pair', source_class_ids=(0,), target_class_ids=(1,), max_distance_px=8.0)
# LineGroupingTask(name='selected_rows', direction='u', class_ids=(0, 1), group_axis='t')
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
