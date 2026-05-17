# Atom Analysis Manual

`atom_analysis_manual` 是一个面向二维原子分辨电子显微图像的 notebook-first Python 工具包。仓库中的 Python 包名为 `em-atom-workbench`。

本项目的核心目标是服务于广泛材料体系的原子柱定位、类别复核、坐标精修和后续几何定量分析，而不是只针对某一个材料体系。HfO2 相关函数作为材料特定扩展保留在源码中，但不是项目的唯一或主要定位。

当前用户工作流已经建立了三个 notebook：

- `notebooks/01_Findatom.ipynb`：通用原子柱定位、自动聚类、人工复核、按类别精修、最终筛选和原子表导出。
- `notebooks/02_Simple_quantitative_spacing_analysis.ipynb`：基于 01 结果的任务式定量分析，包括周期统计、晶格索引、pair 距离、line grouping 和导出。
- `notebooks/03_Cropped_group_centroid_analysis.ipynb`：裁剪 ROI 后的组质心与位移分析，用于从局部图像区域中统计类别组中心和组间位移。

自动聚类类别和定量输出都应被视为可检查的分析结果。类别物理含义、ROI 选择、basis vector 选择和最终解释仍需用户判断。

## 当前状态

源码层已经包含以下能力：

- 图像/session 读取、active session、checkpoint 和 Excel/manifest 导出。
- 单通道与多通道候选原子柱检测。
- 基于局部图像特征的原子柱自动聚类。
- napari 候选点复核、类别复核、ROI/basis vector 交互选择。
- 按类别的亚像素坐标精修和最终筛选。
- 简单定量分析：ROI/class 筛选、basis vector、nearest-forward segment、periodic-vector segment、pair segment、line guide、pair center、period statistics、group centroid 和 displacement。
- 局域几何、reference lattice、local affine strain、vPCF、绘图和导出工具。
- HfO2 heavy/light 多通道辅助函数。

notebook 层当前已经建立 01、02、03 三个主流程。后续仍可继续扩展更专门的材料体系 notebook、批处理 notebook 或论文图整理 notebook。

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

## 代码模块

主要源码位于 `src/em_atom_workbench/`：

- `notebook_workflows.py`：notebook 级编排函数、Excel 导出、figure/manifest 导出。
- `simple_quant.py`：ROI、basis、segment、period、pair、line、group centroid 和 displacement 计算。
- `simple_quant_plotting.py`：simple quant 相关 overlay、histogram、basis、segment、polygon 和 displacement 绘图。
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
|   `-- 03_Cropped_group_centroid_analysis.ipynb
|-- scripts/
|   |-- build_01_findatom_notebook.py
|   |-- build_02_simple_quant_notebook.py
|   `-- build_03_cropped_group_centroid_notebook.py
|-- src/
|   `-- em_atom_workbench/
|       |-- classification.py
|       |-- curate.py
|       |-- detect.py
|       |-- export.py
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
python -m pytest tests/test_simple_quant.py tests/test_simple_quant_plotting.py tests/test_notebook_02_simple_quant_smoke.py tests/test_notebook02_exports.py tests/test_notebook_smoke.py
```

测试主要使用 synthetic data，用来检查接口、表格 schema、notebook 代码单元、导出文件和 session 状态转移，不依赖私有显微数据。

## 注意事项

- 本项目面向广泛材料体系的原子柱定位和几何分析，不局限于 HfO2。
- 多通道分析默认输入图像已经空间配准且 shape 兼容。
- 自动 class id 是图像特征类别，不等同于元素标签。
- 定量结果依赖用户选择的 ROI、class group、basis vector、pair 规则和裁剪区域。
- DM3/DM4 读取依赖可选 HyperSpy/RosettaSciIO，真实仪器 metadata 需要逐例检查。
- `results/` 中生成的输出默认不进入 Git，只保留 `results/.gitkeep`。

## License

`pyproject.toml` 当前声明 MIT license metadata。正式开源分发前建议补充独立的 `LICENSE` 文件。
