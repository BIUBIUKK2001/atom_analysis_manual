# EM Atom Workbench v1 Implementation Plan

## Project Structure

- Use a `src` layout with the package under `src/em_atom_workbench/`.
- Keep notebooks as the primary user interface in `notebooks/`.
- Keep lightweight synthetic-data tests under `tests/`.
- Reserve `examples/` for sample-data notes and `results/` for session snapshots and exports.

## Core Modules

- `session.py`: data models, serialization, provenance, stage bookkeeping
- `io.py`: file loading, metadata handling, calibration extraction, manual override support
- `preprocess.py`: normalization, inversion, denoising, background subtraction, ROI logic
- `detect.py`: candidate detection with explicit contrast-mode handling and diagnostics
- `refine.py`: CoM plus Gaussian-based subpixel refinement with fallbacks
- `curate.py`: QC flags and napari-based manual edit integration
- `lattice.py`: neighbor graph and local basis estimation
- `metrics.py`: geometry, spacing, angle, displacement, and strain-like calculations
- `annotation.py`: manual ROI labels and semiautomatic descriptor-based suggestions
- `polarization.py`: general vector-field framework
- `plotting.py`, `styles.py`, `export.py`: publication-ready visualization and export
- `widgets.py`, `utils.py`: notebook helpers and shared utilities

## Notebook Workflow

- `00`: environment notes, data IO, metadata review, calibration, session initialization
- `01`: preprocessing and candidate detection
- `02`: refinement, QC review, napari correction, final atom tables
- `03`: neighbor graph, lattice vectors, spacing, displacement, strain-like metrics
- `04`: manual and semiautomatic annotation of structure, orientation, and domains
- `05`: publication-quality figures and export workflow
- `06`: optional vector-field and polarization-style mapping demo

## Environment Setup

- Provide `environment.yml` with a Windows-friendly conda-forge stack built around Python 3.11.
- Provide `setup_windows.ps1` to create or update the environment, install the package in editable mode, and register a Jupyter kernel.
- Keep optional runtime dependencies imported lazily where possible so import errors are informative rather than fatal.

## Algorithm Flow

1. Load image and metadata into `AnalysisSession`.
2. Apply conservative preprocessing while preserving atomic contrast.
3. Detect candidates using configurable local-extrema logic, with support for `bright_peak`, `dark_dip`, and `mixed_contrast`.
4. Refine points using local CoM and constrained 2D Gaussian fitting with fallback logic.
5. Apply QC flags and allow manual point editing in napari.
6. Build local neighbor graph and estimate local basis vectors.
7. Compute coordinate-based local metrics.
8. Add manual or semiautomatic annotations.
9. Generate publication-ready figures and structured exports.

## Export Logic

- Keep intermediate results in memory inside the session object.
- Save session snapshots only when requested from notebooks or helper calls.
- Support dedicated export profiles through `ExportConfig`.
- Export figures, tables, annotations, a manifest, and optionally a session pickle into a per-analysis output folder.

## Testing Strategy

- Use synthetic peak images to test detection and refinement.
- Use synthetic lattices to test neighbor-graph and metrics logic.
- Use temporary directories to test export tables, figures, manifests, and overwrite behavior.
- Finish with an import audit, notebook existence check, environment consistency check, and `pytest` discoverability check.

