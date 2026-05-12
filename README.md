# EM Atom Workbench

`em-atom-workbench` 是一个 notebook-first 的 Python 包，用于 2D 原子分辨电子显微图像分析。当前版本围绕 STEM 类强度图像构建，重点支持原子柱定位、亚像素精修、人工校正、局域结构量化、vPCF 局域有序度分析、结构/取向/畴标注，以及面向论文整理的图表和表格导出。

这个包的定位是“透明、可检查、可人工接管”的研究工作流，而不是黑箱式自动相识别器。推荐在 Jupyter notebook 中逐步运行，每一步都保存到 `AnalysisSession`，便于回看参数、检查中间结果和导出最终数据。

## 当前完成状态

目前已经完成一个可运行的 v1 工作台，主流程覆盖：

1. 读取 2D 显微图像和像素标定信息。
2. 可选地进行保守预处理，或直接在原始图像上检测原子柱候选点。
3. 支持单通道候选点检测，以及 HfO2 场景下的 `HAADF + iDPC + optional ABF` 多通道候选点检测。
4. 对候选点进行亚像素精修。
5. 通过 napari 进行人工增删改点和 QC。
6. 基于坐标构建邻居图，计算局域晶格、间距、角度、位移和 strain-like 指标。
7. 计算单通道 vPCF，用于局域有序度和邻域矢量分布分析。
8. 进行结构、取向、畴区域标注，支持人工标签和聚类建议。
9. 生成 publication-style 图像、CSV/Parquet 表格、manifest 和 session 快照。

## 已完成能力总结

- 包结构和环境
  - 使用 `src/em_atom_workbench/` 的标准包结构。
  - `pyproject.toml` 支持 editable install。
  - `environment.yml` 和 `setup_windows.ps1` 提供 Windows/conda 工作流。
  - 依赖中的 napari、HyperSpy/RosettaSciIO 采用可选或延迟导入思路，方便基础包先运行。

- Session 和流程状态
  - `AnalysisSession` 作为中心状态容器，保存原图、通道、候选点、精修点、人工筛选点、局域指标、vPCF、标注、矢量场和 provenance。
  - 支持 pickle 保存/读取、manifest 输出、active session 连接、checkpoint/snapshot 管理。
  - 已加入 workflow stage 管理，重新运行上游步骤时会清理下游缓存，减少旧结果混入。

- 图像读取和像素标定
  - 支持 TIFF、MRC、PNG/JPEG 等 `skimage.io.imread` 可读格式。
  - 支持 DM3/DM4，通过 HyperSpy/RosettaSciIO 读取。
  - 可从 HyperSpy axes、DM original metadata、MRC voxel size、TIFF resolution 中提取标定。
  - 支持 `dataset_index` 选择多 dataset DM 文件。
  - 支持手动标定覆盖。
  - 支持多通道 bundle 读取，并检查通道尺寸和共享标定。

- 预处理
  - 已实现归一化、暗峰反转、ROI crop、edge mask。
  - 支持 Wiener 去卷积/滤波诊断、median/bilateral 类去噪、背景扣除、局部对比增强。
  - 支持单通道和多通道预处理。
  - 当前 notebook 默认更偏向 raw-direct 流程，也就是检测和精修优先使用原始图像，预处理作为可选辅助。

- 候选点检测
  - 单通道检测支持 `bright_peak`、`dark_dip` 和 `mixed_contrast`。
  - 候选点表包含坐标、强度、局部背景、SNR、prominence、通道和角色等信息。
  - 支持去重、边界过滤、候选数量限制和 local-extrema 参数调节。
  - HfO2 多通道流程已完成分阶段检测：
    - 先在 HAADF 中检测 heavy columns。
    - heavy candidates 人工 review 后作为锚点。
    - 再在 iDPC 中检测 light columns。
    - 可用 ABF 作为 confirm channel。
  - 也保留了一步式 `detect_hfo2_multichannel_candidates` API。

- 亚像素精修
  - 支持 legacy refinement 和 adaptive_atomap 风格精修。
  - 使用 CoM 初始化、受约束 2D Gaussian 拟合，并提供 quadratic/CoM fallback。
  - 可根据近邻距离自适应 patch 半径。
  - 对相近 light candidates 提供 overlap shared-shape refinement 路径，保留 close-pair metadata。
  - 精修结果写入质量分数、残差、sigma、fallback 路径等诊断列。

- 人工校正和 napari 交互
  - 支持 candidate editor 和 final curation viewer。
  - 单通道可编辑候选点或最终 atom points。
  - HfO2 多通道支持 heavy/light 分阶段 napari 编辑，light 编辑时保留 heavy reference layer。
  - 编辑后可 roundtrip 回 session，并自动重建 point table schema。
  - 自动 QC flags 覆盖重复点、边缘点、低质量拟合和过近间距。

- 局域结构量化
  - `lattice.py` 构建近邻图和局域 basis。
  - `metrics.py` 计算近邻距离、basis 长度、basis 角、局域取向、位移和 strain-like 指标。
  - 这些指标是基于坐标的探索性量化，不声称替代完整晶体学模型。

- vPCF 局域有序度
  - `vpcf.py` 已实现 local vPCF、batch/global average vPCF 和 session-level vPCF。
  - 支持根据 `keep == True` 过滤人工保留点。
  - session-level 输出会根据像素标定转换到 nm。
  - 支持 ROI 子区域 vPCF 和示例中心原子的 local vPCF。
  - 当前 vPCF notebook 面向单通道 curated atom table。

- 标注、矢量场和可视化
  - 支持 manual polygon labels 保存到 atom table。
  - 支持基于局域几何特征的聚类建议，作为人工标注参考。
  - 支持通用 vector field 构建和 matched-point displacement mapping。
  - 提供 raw image、FFT、预处理诊断、atom overlay、metric map、domain map、vector map、neighbor graph 和 vPCF plot 等绘图函数。

- 导出
  - `export_results` 可导出 figures、tables、annotations、manifest 和可选 session pickle。
  - 如果 `session.strain_table` 已存在且非空，`export_results` 会同时导出 local affine strain 表格。
  - 表格默认支持 `csv` 和 `parquet`。
  - 图片默认支持 `png`、`pdf` 和 `tiff`。
  - 支持 `minimal`、`publication` 和 `full_session` export profiles。

- 测试覆盖
  - `tests/` 中已有合成数据测试，覆盖 IO/标定、预处理、检测、精修、多通道流程、raw-direct 流程、plotting、export、session workflow、active session、notebook smoke test、vPCF 和工具函数。

## 推荐使用场景

- 单张 2D HAADF-STEM、iDPC 或类似原子分辨强度图像。
- 已经配准到同一 field of view 的多通道 HfO2 数据，例如 `iDPC + HAADF`，可选 `ABF`。
- 需要人工检查、人工修点、导出坐标表和论文图的研究流程。
- 需要从 curated coordinates 出发做局域几何、vPCF、畴标注或矢量场分析的探索性工作。

## 暂不承诺的范围

- 不承诺通用自动晶相识别。
- 不承诺任意成像条件下自动识别氧柱。
- 不包含任意多通道图像配准算法，多通道输入应已对齐并共享 ROI。
- 不处理 3D volume、时间序列或大规模批处理流水线。
- vPCF 当前 notebook 面向单通道 curated atom table。
- napari 相关功能依赖本机 Qt/GPU/display stack。

## 安装

### Windows 推荐方式

```powershell
.\setup_windows.ps1
conda activate em-atom-workbench
```

脚本会创建或更新 conda 环境、以 editable mode 安装当前包，并注册 Jupyter kernel。

### 手动安装

```powershell
conda env create -f environment.yml
conda activate em-atom-workbench
python -m pip install -e .
python -m ipykernel install --user --name em-atom-workbench --display-name "Python (em-atom-workbench)"
```

如果只需要核心计算，不使用 napari 或 DM3/DM4，可先安装基础依赖。若要使用交互校正和 DM 文件读取，请安装 optional dependency 组或使用 `environment.yml`。

## Notebook 工作流

`notebooks/` 是主要用户界面，当前包含 7 个 notebook，Markdown 和流程说明面向中文用户。

- `00_02_single_channel_end_to_end.ipynb`
  - 单通道端到端流程，覆盖数据读取、候选点检测、精修、人工修点和 checkpoint。

- `00_02_hfo2_multichannel_end_to_end.ipynb`
  - HfO2 多通道端到端流程，覆盖通道配置、synthetic/real 数据入口、HAADF heavy review、iDPC light detection、final refinement/curation。

- `03_vpcf_local_order.ipynb`
  - 推荐的 03 分析 notebook，用 curated atom table 计算 local/global/ROI vPCF。

- `03_lattice_and_local_metrics.ipynb`
  - 坐标邻居图、局域晶格和 strain-like metrics 分析。当前可作为 legacy 或补充分析入口。

- `04_structure_domain_annotation.ipynb`
  - 结构、取向、畴标注，支持人工 polygon labels 和聚类建议。

- `05_publication_figures.ipynb`
  - 论文图和正式导出流程。

- `06_polarization_and_vector_mapping_optional.ipynb`
  - 可选矢量场和 polarization-style mapping 示例。

## 快速开始

### 单通道

```python
from pathlib import Path

from em_atom_workbench import (
    CurationConfig,
    DetectionConfig,
    ExportConfig,
    RefinementConfig,
    VPCFConfig,
    compute_local_metrics,
    compute_session_vpcf,
    curate_points,
    detect_candidates,
    export_results,
    load_image,
    refine_points,
)

session = load_image(
    Path("path/to/image.dm4"),
    manual_calibration={"size": 0.01, "unit": "nm"},
    contrast_mode="bright_peak",
)

session = detect_candidates(session, DetectionConfig(contrast_mode="bright_peak"))
session = refine_points(
    session,
    RefinementConfig(mode="adaptive_atomap", gaussian_image_source="raw"),
)
session = curate_points(session, CurationConfig())
session = compute_local_metrics(session)
session = compute_session_vpcf(session, VPCFConfig(r_max_px=40.0))

manifest_path = export_results(
    session,
    ExportConfig(output_dir="results", export_profile="publication", overwrite=True),
)
```

### HfO2 多通道

```python
from pathlib import Path

from em_atom_workbench import (
    HfO2MultichannelDetectionConfig,
    detect_hfo2_heavy_candidates,
    detect_hfo2_light_candidates,
    edit_hfo2_heavy_candidates_with_napari,
    edit_hfo2_light_candidates_with_napari,
    load_image_bundle,
)

session = load_image_bundle(
    {
        "idpc": Path("path/to/idpc.dm4"),
        "haadf": Path("path/to/haadf.dm4"),
        "abf": Path("path/to/abf.dm4"),
    },
    primary_channel="idpc",
    manual_calibration={"size": 0.01, "unit": "nm"},
    contrast_modes={
        "idpc": "bright_peak",
        "haadf": "bright_peak",
        "abf": "dark_dip",
    },
)

config = HfO2MultichannelDetectionConfig(
    heavy_channel="haadf",
    light_channel="idpc",
    confirm_channel="abf",
)

session = detect_hfo2_heavy_candidates(session, config)
session = edit_hfo2_heavy_candidates_with_napari(
    session,
    heavy_channel=config.heavy_channel,
)
session = detect_hfo2_light_candidates(session, config)
session = edit_hfo2_light_candidates_with_napari(
    session,
    heavy_channel=config.heavy_channel,
    light_channel=config.light_channel,
)
```

## 项目结构

```text
.
|-- environment.yml
|-- pyproject.toml
|-- setup_windows.ps1
|-- README.md
|-- IMPLEMENTATION_PLAN.md
|-- FINAL_AUDIT.md
|-- examples/
|-- notebooks/
|-- results/
|-- scripts/
|-- tests/
`-- src/
    `-- em_atom_workbench/
        |-- __init__.py
        |-- annotation.py
        |-- curate.py
        |-- detect.py
        |-- export.py
        |-- io.py
        |-- lattice.py
        |-- metrics.py
        |-- notebook_workflows.py
        |-- plotting.py
        |-- polarization.py
        |-- preprocess.py
        |-- refine.py
        |-- session.py
        |-- styles.py
        |-- utils.py
        |-- vpcf.py
        `-- widgets.py
```

## 测试

```powershell
python -m pytest
```

当前测试集重点使用 synthetic data，目的是验证算法接口、数据表 schema、session 状态转移、导出文件和 notebook 代码单元的基本一致性。

## 关键限制

- DM3/DM4 读取依赖 HyperSpy 和 RosettaSciIO，且不同仪器导出的 metadata 结构可能不同。
- 多通道 workflow 假设输入图像已经空间配准。
- 坐标质量会直接影响局域晶格、strain-like metrics 和 vPCF。
- 自动聚类标注只是建议层，最终结构标签应由用户检查。
- vector-field 模块是通用位移映射框架，不等同于完整自动 polarization 物理解释。

## 后续可扩展方向

- 更丰富的 HfO2 或其他材料体系多通道模板。
- 更强的批处理 helpers。
- 更完善的 notebook widget 和 ROI 绘制体验。
- 更明确的材料体系 reference-lattice 模型。
- 更多导出 schema 和跨软件互操作格式。
