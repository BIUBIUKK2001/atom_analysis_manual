# Atom Analysis Manual

`atom_analysis_manual` 是一个面向二维原子分辨电子显微图像的 notebook-first Python 工具包。仓库中的 Python 包名为 `em-atom-workbench`。

本项目的核心目标是服务于广泛材料体系的原子柱定位、类别复核、坐标精修和后续几何定量分析，而不是只针对某一个材料体系。HfO2 相关函数作为材料特定扩展保留在源码中，但不是项目的唯一或主要定位。

当前用户工作流已经建立了四个 notebook，并统一使用 analysis workspace 组织输入、stage session、表格、图像和 manifest：

- `notebooks/01_Findatom.ipynb`：通用原子柱定位、自动聚类、人工复核、按类别精修、最终筛选和原子表导出。
- `notebooks/02_Simple_quantitative_spacing_analysis.ipynb`：基于 01 结果的任务式定量分析，包括周期统计、晶格索引、pair 距离、line grouping 和导出。
- `notebooks/03_Cropped_group_centroid_analysis.ipynb`：裁剪 ROI 后的组质心与位移分析，用于从局部图像区域中统计类别组中心和组间位移。
- `notebooks/04_Disk_integrated_intensity_mapping.ipynb`：从 01 final curated 或 active session 读取已分类原子柱坐标，可选择 candidate/refined/curated 坐标源，以固定半径圆盘积分每个原子柱附近像素强度，输出 intensity map 和 histogram，用于筛查是否存在显著低强度原子柱。

自动聚类类别和定量输出都应被视为可检查的分析结果。类别物理含义、ROI 选择、basis vector 选择和最终解释仍需用户判断。

## 当前状态

源码层已经包含以下能力：

- 图像/session 读取、active session、checkpoint 和 Excel/manifest 导出。
- project-level analysis workspace、stage session、shared metadata 和跨 notebook 状态恢复。
- 单通道与多通道候选原子柱检测。
- 基于局部图像特征的原子柱自动聚类。
- napari 候选点复核、类别复核、ROI/basis vector 交互选择。
- 按类别的亚像素坐标精修和最终筛选。
- 简单定量分析：ROI/class 筛选、basis vector、nearest-forward segment、periodic-vector segment、pair segment、line guide、pair center、period statistics、group centroid 和 displacement。
- 局域几何、reference lattice、local affine strain、vPCF、绘图和导出工具。
- HfO2 heavy/light 多通道辅助函数。

notebook 层当前已经建立 01、02、03、04 四个主流程。后续仍可继续扩展更专门的材料体系 notebook、批处理 notebook 或论文图整理 notebook。

## Notebook 工作流

### 01 Findatom

`01_Findatom.ipynb` 是第一阶段工作流，用于从图像得到可复核的原子坐标表。

主要步骤：

1. 设置图像通道、主显示通道、dataset index 和可选像素标定。
2. 初始化 `AnalysisSession`。
3. 在一个或多个通道上检测候选原子柱。
4. 在 napari 中人工复核候选点。
5. 提取局部图像特征并自动聚类。
6. 在 napari 中复核类别。
7. 按类别进行亚像素精修。
8. 自动筛选重复点、边缘点、低质量点和异常拟合点。
9. 保存 active session、checkpoint。
10. 导出最终原子表 Excel。

### 02 Simple Quantitative Spacing Analysis

`02_Simple_quantitative_spacing_analysis.ipynb` 从 01 完成的 session、active session 或导出的原子表开始，用于任务式间距和几何统计。

当前任务：

1. 读取 session、图像和原子表。
2. 预览全局 ROI 和类别着色点。
3. 定义或交互选择 basis vectors。
4. Task 1A：按 ROI/class 统计 a/b 周期，绘制直方图和 segment overlay，并导出 Excel。
5. Task 1B：基于 anchor/reference/origin 做 lattice indexing、complete cell、local strain 和 polygon strain map，并导出 Excel。
6. Task 2：寻找 strict mutual nearest pairs，按 projection line 分组，绘制 pair/line 图，并导出 Excel。
7. 最终导出 tables、figures、configs 和 manifest。

02 中原先的 cropped group-centroid 分析已经拆分到 03 notebook。

### 03 Cropped Group-Centroid Analysis

`03_Cropped_group_centroid_analysis.ipynb` 用于在原图中先裁剪局部 ROI，再在裁剪坐标系内做组质心和位移分析。

主要步骤：

1. 读取 01 session、图像和原子表。
2. 在原图上定义 crop ROI。
3. 在裁剪图像上定义 measurement ROIs。
4. 配置 class groups 和 center pairs。
5. 计算 group centroids 与 displacements。
6. 绘制裁剪区域中的 displacement arrows。
7. 导出 Excel。
8. 导出最终 manifest。

### 04 Disk-Integrated Intensity Mapping

`04_Disk_integrated_intensity_mapping.ipynb` 用于基于 01 已经完成的原子柱定位、分类、精修和 curation 结果，对每个原子柱附近固定半径圆盘内的像素强度求和。

主要步骤：

1. 读取 01 stage session、active session 或手动指定 session pickle。
2. 选择 `candidate`、`refined` 或 `curated` 坐标源。
3. 可选按 class 和 ROI 筛选原子柱。
4. 预览固定半径积分圆盘。
5. 计算 `disk_intensity_sum` 和 `disk_intensity_mean`。
6. 绘制原图上的 intensity map。
7. 绘制 `disk_intensity_sum` histogram。
8. 导出 CSV、figure、config、manifest 和 04 stage session。

04 不做自动 vacancy 判定，只提供 disk intensity distribution。低强度群体需要结合 histogram、mapping、成像条件和 class 物理意义判断。如果关注疑似 vacancy，建议分别比较 candidate coordinates 与 refined coordinates 两种积分结果。
## 命令行交互流水线

除 notebook 之外，当前仓库新增了命令行入口：

```powershell
python scripts\run_interactive_analysis_pipeline.py --output-root results --dataset-id dataset_001 --analysis-id run_chat_001 --run-04
```

该脚本复用 notebook 01 和 04 的核心函数，适合不想逐格运行 notebook 时使用。默认行为：

- 创建或复用 analysis workspace。
- 运行 01 风格的图像导入、候选点检测、分类、精修、curation 和最终原子表导出。
- 如果未设置 `--skip-candidate-review` 或 `--skip-class-review`，会打开 napari 复核窗口；脚本会阻塞等待窗口关闭后继续。
- 设置 `--run-04` 时继续运行 disk-integrated intensity mapping，并导出 intensity 表、图和 manifest。
- 支持 `--channel NAME=PATH`、`--primary-channel`、`--pixel-size`、检测参数、分类类别数、精修参数、curation 参数和 04 的积分半径等命令行参数。

该脚本用于自动化同一套交互分析逻辑，不替代 notebook 中的逐步调参和人工检查。

## Analysis workspace and output structure

新工作流推荐所有 notebook 共享同一个 analysis workspace。用户只需要在 01、02、03、04 开头设置同一组参数：

```python
OUTPUT_ROOT = Path("D:/analysis_outputs")
DATASET_ID = "HZO_sample01_area03"
ANALYSIS_ID = "run_001"
```

对应 workspace 路径为：

```text
D:/analysis_outputs/HZO_sample01_area03/run_001/
```

目录结构固定为：

```text
<OUTPUT_ROOT>/<DATASET_ID>/<ANALYSIS_ID>/
  project_config.json
  state/
    active_session.pkl
    active_session.json
    latest_session.json
    sessions/
      01_loaded.pkl
      01_candidate_reviewed.pkl
      01_class_reviewed.pkl
      01_refined.pkl
      01_final_curated.pkl
      02_simple_quant.pkl
      03_group_centroid.pkl
      04_intensity_mapping.pkl
  shared/
    channel_summary.csv
    pixel_calibration.json
    input_metadata.json
  01_findatom/
    configs/
    tables/
    figures_preview/
    figures_final/
    checkpoints/
    manifest.json
  02_simple_quant/
    configs/
    tables/
    figures_preview/
    figures_final/
    session/
    manifest.json
  03_group_centroid/
    configs/
    tables/
    figures_preview/
    figures_final/
    session/
    manifest.json
  04_intensity_mapping/
    configs/
    tables/
    figures_preview/
    figures_final/
    session/
    manifest.json
  manifests/
    01_findatom_manifest.json
    02_simple_quant_manifest.json
    03_group_centroid_manifest.json
    04_intensity_mapping_manifest.json
    project_manifest.json
```

`state/sessions/01_final_curated.pkl` 是 02、03 和 04 的默认入口。`state/active_session.pkl` 是当前 workspace 的 latest pointer，用于恢复最近运行状态；它不再代表全项目唯一入口。重新打开 notebook 时，只要设置同样的 `OUTPUT_ROOT / DATASET_ID / ANALYSIS_ID`，02/03/04 就能继续读取 01 保存的 final curated session。

`figures_preview/` 用于可选保存调参预览图，默认不保存。`figures_final/` 用于正式导出的图，标题、字体、格式、dpi 和部分 legend/颜色开关在 notebook 的 final export 参数 cell 中设置。`configs/` 保存运行参数，`tables/` 保存 CSV/Excel 表格，stage `manifest.json` 和 `manifests/` 记录导出路径、session 来源和 workspace schema。

分析多个数据时，切换 `DATASET_ID`；同一数据的不同分析版本则切换 `ANALYSIS_ID`。旧的 `results/_active_session.pkl` 兼容逻辑仍保留，可通过 `SESSION_PATH` 手动读取，但新流程推荐使用 workspace 与 `state/sessions/*`。

## 代码模块

主要源码位于 `src/em_atom_workbench/`：

- `notebook_workflows.py`：notebook 级编排函数、Excel 导出、figure/manifest 导出。
- `workspace.py`：project-level workspace、stage session、shared metadata 和 manifest 管理。
- `figure_config.py`：final figure export 参数规范化，不改变绘图核心 API。
- `simple_quant.py`：ROI、basis、segment、period、pair、line、group centroid 和 displacement 计算。
- `simple_quant_plotting.py`：simple quant 相关 overlay、histogram、basis、segment、polygon 和 displacement 绘图。
- `intensity.py`：固定半径圆盘积分强度点表准备、像素积分和 summary。
- `intensity_plotting.py`：圆盘积分半径预览、intensity map 和 histogram 绘图。
- `simple_quant_widgets.py`：napari ROI、direction、basis vector 交互选择。
- `classification.py`、`detect.py`、`refine.py`、`curate.py`：01 原子定位和分类主流程。
- `strain.py`、`reference.py`、`metrics.py`、`lattice.py`、`vpcf.py`：后续几何和结构分析能力。

## 安装

Windows/conda 推荐方式：

```powershell
.\setup_windows.ps1
conda activate em-atom-workbench
```

手动安装：

```powershell
conda env create -f environment.yml
conda activate em-atom-workbench
python -m pip install -e .
python -m ipykernel install --user --name em-atom-workbench --display-name "Python (em-atom-workbench)"
```

napari、PyQt、HyperSpy、RosettaSciIO 等属于交互或 DM 文件读取相关依赖。核心 synthetic 流程和大部分测试不需要打开 napari。

## 重新生成 Notebook

notebook 由脚本生成：

```powershell
python scripts\build_01_findatom_notebook.py
python scripts\build_02_simple_quant_notebook.py
python scripts\build_03_cropped_group_centroid_notebook.py
python scripts\build_04_disk_integrated_intensity_notebook.py
```

修改 notebook 结构或同步源码单元时，优先修改 builder script，再重新生成 notebook。

## 项目结构

```text
.
|-- README.md
|-- environment.yml
|-- pyproject.toml
|-- setup_windows.ps1
|-- docs/
|   `-- local_affine_strain_plan.md
|-- examples/
|   `-- sample_data_placeholder.md
|-- notebooks/
|   |-- 01_Findatom.ipynb
|   |-- 02_Simple_quantitative_spacing_analysis.ipynb
|   |-- 03_Cropped_group_centroid_analysis.ipynb
|   `-- 04_Disk_integrated_intensity_mapping.ipynb
|-- scripts/
|   |-- build_01_findatom_notebook.py
|   |-- build_02_simple_quant_notebook.py
|   |-- build_03_cropped_group_centroid_notebook.py
|   |-- build_04_disk_integrated_intensity_notebook.py
|   `-- run_interactive_analysis_pipeline.py
|-- src/
|   `-- em_atom_workbench/
|       |-- classification.py
|       |-- curate.py
|       |-- detect.py
|       |-- export.py
|       |-- figure_config.py
|       |-- intensity.py
|       |-- intensity_plotting.py
|       |-- io.py
|       |-- lattice.py
|       |-- metrics.py
|       |-- notebook_workflows.py
|       |-- plotting.py
|       |-- reference.py
|       |-- refine.py
|       |-- session.py
|       |-- simple_quant.py
|       |-- simple_quant_plotting.py
|       |-- simple_quant_widgets.py
|       |-- strain.py
|       |-- vpcf.py
|       |-- workspace.py
|       `-- widgets.py
`-- tests/
```

## 测试

完整测试：

```powershell
python -m pytest
```

当前 notebook 和 simple quant 相关重点测试：

```powershell
python -m pytest tests/test_workspace.py tests/test_simple_quant.py tests/test_simple_quant_plotting.py tests/test_intensity.py tests/test_notebook_04_disk_intensity_smoke.py tests/test_notebook_02_simple_quant_smoke.py tests/test_notebook02_exports.py tests/test_notebook_smoke.py
```

命令行流水线脚本可先做语法检查：

```powershell
python -m py_compile scripts/run_interactive_analysis_pipeline.py
```

测试主要使用 synthetic data，用来检查接口、表格 schema、notebook 代码单元、导出文件和 session 状态转移，不依赖私有显微数据。

## 注意事项

- 本项目面向广泛材料体系的原子柱定位和几何分析，不局限于 HfO2。
- 多通道分析默认输入图像已经空间配准且 shape 兼容。
- 自动 class id 是图像特征类别，不等同于元素标签。
- 定量结果依赖用户选择的 ROI、class group、basis vector、pair 规则和裁剪区域。
- 04 不做自动 vacancy 判定，只输出 fixed-radius disk intensity distribution；低强度群体需要结合 histogram、mapping、成像条件和 class 物理意义判断。
- 如果关注疑似 vacancy，建议比较 candidate coordinates 与 refined coordinates 两种积分结果。
- DM3/DM4 读取依赖可选 HyperSpy/RosettaSciIO，真实仪器 metadata 需要逐例检查。
- `results/` 中生成的输出默认不进入 Git，只保留 `results/.gitkeep`。

## License

`pyproject.toml` 当前声明 MIT license metadata。正式开源分发前建议补充独立的 `LICENSE` 文件。
