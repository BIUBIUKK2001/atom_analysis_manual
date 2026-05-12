# Atom Analysis Manual

`atom_analysis_manual` is the working repository for a notebook-first Python toolkit for atom-column localization and analysis in 2D electron microscopy images. The Python package inside the repository is named `em-atom-workbench`.

The package is designed for broad materials systems rather than a single material family. Its first priority is robust, inspectable atom-column finding, classification, manual review, and position refinement across different atomic-resolution image datasets. HfO2-specific helpers exist in the source code as one extension path, but they are not the central scope of the package.

The current user-facing notebook workflow is centered on `notebooks/01_Findatom.ipynb`: load one or more aligned image channels, detect atom-column candidates, manually review candidates in napari, cluster columns by local image features, review class labels, refine positions by class, curate final points, and save checkpoints for later analysis.

This repository is intended for transparent, inspectable research workflows. It is not a fully automatic phase identifier, and automatic clustering labels should be treated as image-feature groups until they are checked by the user.

## Current Scope

The current source code already contains modules for detection, classification, manual review, refinement, local metrics, reference-lattice handling, local affine strain, vPCF, plotting, export, and session/checkpoint management.

The notebook layer is still being built. At the moment, only the first workflow notebook has been established:

- One main notebook: `notebooks/01_Findatom.ipynb`.

The repository also contains:

- A reusable Python package under `src/em_atom_workbench/`.
- A notebook builder script for the first workflow: `scripts/build_01_findatom_notebook.py`.
- Tests under `tests/` covering IO, detection, classification, refinement, curation, local metrics, vPCF, local affine strain, plotting, export, and session behavior.
- A short design note for local affine strain under `docs/local_affine_strain_plan.md`.
- A placeholder example note under `examples/sample_data_placeholder.md`.

Large raw data, generated figures, temporary files, caches, and analysis outputs are intentionally ignored by Git. Keep raw microscopy data and generated `results/` outputs local unless you explicitly decide to publish them.

## Current Notebook Workflow

`notebooks/01_Findatom.ipynb` currently guides the analysis through these stages:

1. Configure input channels, primary display channel, dataset index, and optional manual pixel calibration.
2. Initialize an `AnalysisSession`.
3. Detect candidate atom columns on one or more image channels.
4. Review candidate points in napari before classification.
5. Extract local image features and cluster atom columns into user-reviewable classes.
6. Optionally review and edit class assignments in napari.
7. Refine atom-column positions by class with configurable Gaussian/center-of-mass settings.
8. Apply final automatic curation checks such as duplicate, edge, quality, fit residual, and spacing flags.
9. Save active-session files and final checkpoints.

The notebook can run with real image paths or with the built-in synthetic demo data path used by the workflow helpers. Downstream notebooks for the already implemented analysis modules still need to be designed and assembled.

## Package Features

The `em_atom_workbench` package currently includes:

- Image/session IO: `load_image`, `load_image_bundle`, `AnalysisSession`, active-session and checkpoint helpers.
- Candidate detection: single-channel and multichannel local-extrema detection with deduplication and quality metrics.
- Atom-column classification: local patch feature extraction, clustering, class naming/coloring, summary tables, and napari review.
- Manual review and curation: candidate editing, final point curation, QC flags, and keep/drop columns.
- Refinement: class-aware subpixel refinement with adaptive patch sizing and fallback paths.
- Material-specific extension helpers: HfO2 staged heavy/light multichannel detection helpers for aligned HAADF/iDPC/optional ABF data.
- Local geometry: neighbor graph construction and local metric computation.
- Reference lattice and strain: reference-lattice suggestion/building and local affine strain computation.
- vPCF: local, batch, and session-level vector pair correlation analysis.
- Plotting/export: overlays, class plots, vPCF plots, formal result export, and manifest generation.

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

Optional interactive and DM-file features rely on packages such as napari, PyQt, HyperSpy, and RosettaSciIO. The core tests and many synthetic workflows can run without opening napari.

## Quick API Example

```python
from pathlib import Path

from em_atom_workbench.notebook_workflows import (
    display_notebook_result,
    initialize_generic_classification_session,
    run_generic_candidate_detection,
    run_atom_column_classification,
)
from em_atom_workbench import (
    AtomColumnClassificationConfig,
    DetectionConfig,
    PixelCalibration,
)

result = initialize_generic_classification_session(
    result_root="results",
    channels={"haadf": Path("path/to/haadf.tif")},
    primary_channel="haadf",
    channel_contrast_modes={"haadf": "bright_peak"},
    manual_calibration=PixelCalibration(size=0.01, unit="nm", source="manual"),
)
session = result.session
display_notebook_result(result)

result = run_generic_candidate_detection(
    session,
    result_root="results",
    detection_configs_by_channel={
        "haadf": DetectionConfig(
            contrast_mode="bright_peak",
            gaussian_sigma=1.0,
            min_distance=5,
            threshold_rel=0.05,
        )
    },
)
session = result.session
display_notebook_result(result)

result = run_atom_column_classification(
    session,
    result_root="results",
    classification_config=AtomColumnClassificationConfig(
        source_table="candidate",
        n_classes=3,
    ),
)
session = result.session
display_notebook_result(result)
```

This API example only shows the early programmatic path. For normal use, prefer `notebooks/01_Findatom.ipynb` because it exposes the current workflow parameters and includes the required manual review stages.

## Regenerating The Current Notebook

`notebooks/01_Findatom.ipynb` is generated by:

```powershell
python scripts\build_01_findatom_notebook.py
```

Use this script when `01_Findatom.ipynb` needs to be rebuilt from source-controlled cells.

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
|   `-- 01_Findatom.ipynb
|-- scripts/
|   `-- build_01_findatom_notebook.py
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
|       |-- strain.py
|       |-- vpcf.py
|       `-- widgets.py
`-- tests/
```

## Testing

Run the test suite from the repository root:

```powershell
python -m pytest
```

The tests mostly use synthetic data so that algorithm interfaces, table schemas, session-stage transitions, and exports can be checked without private microscopy datasets.

## Notes And Limitations

- The package aims at broad atom-column localization and analysis workflows, not only HfO2.
- The implemented source modules are ahead of the notebook layer; only `01_Findatom.ipynb` is currently established as a notebook workflow.
- The current notebook focuses on generic atom-column localization, classification, manual review, and refinement.
- HfO2-specific heavy/light helpers are available in the package as material-specific extensions, but the current repository does not include a separate HfO2 notebook.
- Multichannel analysis assumes channels are already spatially aligned and have compatible shapes.
- DM3/DM4 reading depends on optional HyperSpy/RosettaSciIO support and may need metadata checks for instrument-specific files.
- Automatic class IDs are not physical element labels by themselves.
- Coordinate quality directly affects local metrics, reference-lattice fitting, local affine strain, and vPCF results.
- Generated outputs under `results/` are ignored except for `results/.gitkeep`.

## License

`pyproject.toml` currently declares MIT license metadata. Add a standalone `LICENSE` file before relying on this repository as a formally distributed open-source package.
