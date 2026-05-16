# Atom Analysis Manual

`atom_analysis_manual` is a notebook-first Python toolkit for atom-column localization and quantitative analysis in 2D electron microscopy images. The Python package in this repository is named `em-atom-workbench`.

The package is designed for broad materials systems rather than one material family. Its main goal is a transparent workflow for atom-column finding, class-aware review, position refinement, and follow-up geometric quantification from curated coordinates. HfO2-specific helpers exist as one extension path, but they are not the central scope of the package.

The current user-facing workflow is built around two notebooks:

- `notebooks/01_Findatom.ipynb`: atom-column detection, classification, manual review, class-aware refinement, curation, checkpointing, and final atom-table export.
- `notebooks/02_Simple_quantitative_spacing_analysis.ipynb`: task-based quantitative spacing analysis from the completed 01 session or exported atom table.

Automatic clustering labels and quantitative task outputs are intended as inspectable analysis aids. Physical interpretation remains user-reviewed.

## Current Scope

The source package already contains modules for image/session IO, atom detection, feature-based classification, napari review, refinement, local metrics, reference-lattice handling, local affine strain, vPCF, plotting, export, and session/checkpoint management.

The notebook layer currently has two established workflows:

- `01_Findatom.ipynb`: generic atom-column localization and class-aware refinement.
- `02_Simple_quantitative_spacing_analysis.ipynb`: ROI/class/basis/task-based quantitative analysis.

The repository also contains:

- Notebook builder scripts in `scripts/`.
- Tests under `tests/` covering the notebooks, simple quantification, plotting, export helpers, local affine strain, and core workflow behavior.
- A design note for local affine strain under `docs/local_affine_strain_plan.md`.
- A placeholder example note under `examples/sample_data_placeholder.md`.

Large raw data, generated figures, temporary files, caches, and analysis outputs are intentionally ignored by Git. Keep raw microscopy data and generated `results/` outputs local unless you explicitly decide to publish them.

## Notebook Workflows

### 01 Findatom

`notebooks/01_Findatom.ipynb` guides the first-stage workflow:

1. Configure image channels, primary display channel, dataset index, and optional manual pixel calibration.
2. Initialize an `AnalysisSession`.
3. Detect candidate atom columns on one or more channels.
4. Review candidate points in napari before classification.
5. Extract local image features and cluster atom columns into user-reviewable classes.
6. Optionally review and edit class assignments in napari.
7. Refine atom-column positions by class.
8. Apply final automatic curation checks and keep/drop flags.
9. Save active-session files and checkpoints.
10. Export the final atom table to Excel.

### 02 Simple Quantitative Spacing Analysis

`notebooks/02_Simple_quantitative_spacing_analysis.ipynb` starts from the completed 01 session, an active session file, or an exported atom table. It provides task-oriented quantification:

1. Load session/image/atom table and prepare analysis points.
2. Preview global ROI and class-colored atoms.
3. Define or pick basis vectors.
4. Task 1A: ROI/class-filtered a/b period statistics, histograms, period segment overlays, and Excel export.
5. Task 1B: anchor/reference/origin-based lattice indexing, complete-cell tables, local strain, polygon maps, and Excel export.
6. Task 2: strict mutual nearest pair finding, projection-line grouping, pair/line figures, and Excel export.
7. Task 3: ROI/class group centroids, displacement vectors, group-center figures, and Excel export.
8. Final export of tables, figures, configs, and manifest.

The 02 workflow uses unified `measurement_segments` tables where possible, along with supporting tables such as `analysis_points`, `basis_vector_table`, `pair_center_points`, and `line_guides`.

## Package Features

The `em_atom_workbench` package currently includes:

- Image/session IO: `load_image`, `load_image_bundle`, `AnalysisSession`, active-session helpers, and checkpoints.
- Candidate detection: single-channel and multichannel local-extrema detection with deduplication and quality metrics.
- Atom-column classification: patch feature extraction, clustering, class naming/coloring, summary tables, and napari class review.
- Manual review and curation: candidate editing, final curation, QC flags, and keep/drop columns.
- Refinement: class-aware subpixel refinement with adaptive patch sizing and fallback paths.
- Simple quantification: ROI/class selection, basis vectors, nearest-forward segments, periodic-vector segments, pair segments, line guides, pair centers, period statistics, group centroids, and displacement summaries.
- Local geometry: neighbor graph construction and local metric computation.
- Reference lattice and strain: reference-lattice suggestion/building and local affine strain computation.
- Plotting/export: overlays, class plots, basis checks, segment maps, histograms, polygon maps, Excel exports, figures, and manifests.
- Material-specific extension helpers: HfO2 staged heavy/light multichannel detection helpers for aligned HAADF/iDPC/optional ABF data.
- vPCF: local, batch, and session-level vector pair correlation analysis.

## Installation

On Windows with conda, use the setup script:

```powershell
.\setup_windows.ps1
conda activate em-atom-workbench
```

Manual installation:

```powershell
conda env create -f environment.yml
conda activate em-atom-workbench
python -m pip install -e .
python -m ipykernel install --user --name em-atom-workbench --display-name "Python (em-atom-workbench)"
```

Optional interactive and DM-file features rely on packages such as napari, PyQt, HyperSpy, and RosettaSciIO. Core synthetic workflows and tests can run without opening napari.

## Regenerating Notebooks

The notebooks are generated from builder scripts:

```powershell
python scripts\build_01_findatom_notebook.py
python scripts\build_02_simple_quant_notebook.py
```

Use these scripts when notebook structure or source-controlled cells need to be rebuilt.

## Project Structure

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
|   `-- 02_Simple_quantitative_spacing_analysis.ipynb
|-- scripts/
|   |-- build_01_findatom_notebook.py
|   `-- build_02_simple_quant_notebook.py
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

## Testing

Run the full test suite from the repository root:

```powershell
python -m pytest
```

For the current notebook and simple-quantification workflow, the focused test set is:

```powershell
python -m pytest tests/test_simple_quant.py tests/test_simple_quant_plotting.py tests/test_notebook_02_simple_quant_smoke.py tests/test_notebook02_exports.py tests/test_strain_api.py tests/test_notebook_smoke.py
```

The tests mostly use synthetic data so that algorithm interfaces, table schemas, session-stage transitions, notebook code cells, and exports can be checked without private microscopy datasets.

## Notes And Limitations

- The package aims at broad atom-column localization and analysis workflows, not only HfO2.
- The established notebook workflows are currently 01 Findatom and 02 Simple Quantitative Spacing Analysis.
- Multichannel analysis assumes channels are already spatially aligned and have compatible shapes.
- DM3/DM4 reading depends on optional HyperSpy/RosettaSciIO support and may need metadata checks for instrument-specific files.
- Automatic class IDs are image-feature classes, not physical element labels by themselves.
- Quantitative tasks depend on curated atom-coordinate quality, chosen ROIs, chosen classes, and user-defined or picked basis vectors.
- HfO2-specific heavy/light helpers are available in the package as material-specific extensions, but there is not yet a separate HfO2 notebook workflow in this repository.
- Generated outputs under `results/` are ignored except for `results/.gitkeep`.

## License

`pyproject.toml` currently declares MIT license metadata. Add a standalone `LICENSE` file before relying on this repository as a formally distributed open-source package.
