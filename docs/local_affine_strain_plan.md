# Local Affine Strain Implementation Note

This note audits the current coordinate/session model and notebook workflow before adding local affine strain. It does not implement the strain algorithm.

## 1. Coordinate/session audit

### Coordinate storage in `AnalysisSession`

`AnalysisSession` is the central mutable workflow object in `src/em_atom_workbench/session.py`.

- Raw detected coordinates are stored in `session.candidate_points`.
- Subpixel-refined coordinates are stored in `session.refined_points`.
- QC-filtered or manually curated coordinates are stored in `session.curated_points`.
- `session.get_atom_table(preferred="curated")` returns `curated_points` when non-empty, otherwise `refined_points`, otherwise `candidate_points`.
- Neighbor/basis state is stored in `session.neighbor_graph`.
- Existing scalar local geometry metrics are stored in `session.local_metrics`.

### Coordinate columns and units

The atom tables are pixel-first.

- `x_px`, `y_px`: global image coordinates in pixels. These are the canonical coordinate columns used by detection, refinement, curation, lattice building, metrics, plotting, export, and vPCF point selection.
- `x_local_px`, `y_local_px`: local coordinates relative to a processed-image origin/crop, mainly for detection tables and napari roundtrips.
- `x_input_px`, `y_input_px`: pre-refinement input coordinates, in pixels.
- `x_phys`, `y_phys`, `unit`: attached during refinement by `attach_physical_coordinates()`. These are in `session.pixel_calibration.unit`, not guaranteed to be nm.
- Existing local lattice/metric outputs are in px for distances and basis lengths, deg for angles/orientation, and dimensionless for strain-like components.
- vPCF session outputs separately convert calibrated distances/axes to nm, but that convention is currently specific to `vpcf.py`, not the atom coordinate tables or `local_metrics`.

### Role/channel/keep/QC columns

Role and channel metadata already exist in candidate/refined workflows.

- Candidate tables include `column_role`, `seed_channel`, `confirm_channel`, `parent_heavy_id`, `support_score`, and `confirm_score`.
- Refinement preserves passthrough metadata for `column_role`, `seed_channel`, `confirm_channel`, `parent_heavy_id`, and `contrast_mode_used`.
- Curation adds QC-style columns: `flag_duplicate`, `flag_edge`, `flag_low_quality`, `flag_poor_fit`, `flag_spacing_violation`, and `keep`.
- There is no single `qc` column. Existing QC is represented by the flag columns, `quality_score`, `fit_residual`, and `keep`.

### Existing local metrics and basis data

`src/em_atom_workbench/lattice.py` builds `session.neighbor_graph` with:

- `edges`: `source_atom_id`, `target_atom_id`, `distance_px`, `dx_px`, `dy_px`.
- `basis_table`: `atom_id`, `basis_a_x`, `basis_a_y`, `basis_b_x`, `basis_b_y`, `basis_a_length_px`, `basis_b_length_px`, `basis_angle_deg`, `local_orientation_deg`.
- `directed_neighbors`: row-index based neighbor lists.
- `config`: the `LatticeConfig`.

`src/em_atom_workbench/metrics.py` writes `session.local_metrics` with:

- nearest-neighbor distance statistics.
- bond-angle statistics.
- basis lengths, basis angle, and local orientation.
- optional reference displacement columns.
- existing strain-like columns: `strain_exx`, `strain_eyy`, `strain_exy`.

Important gap: `session.local_metrics` does not currently copy local basis vector components (`basis_a_x`, `basis_a_y`, `basis_b_x`, `basis_b_y`). Those are only in `session.neighbor_graph["basis_table"]`.

### Least invasive storage plan

For the first local affine strain implementation, keep storage aligned with the existing session-first pattern:

- Add a small `reference_lattice` field to `AnalysisSession`, defaulting to `{}`. Store reference basis vectors, coordinate unit, source mode, selected atom/ROI metadata, and config/provenance there.
- Add a `strain_table` field to `AnalysisSession`, defaulting to an empty `DataFrame`. Store per-atom affine strain outputs there instead of overloading `local_metrics`.
- Backfill both fields in `AnalysisSession.__setstate__()` for old pickles.
- Clear `strain_table` and `reference_lattice` when upstream coordinate stages are invalidated, especially `preprocess`, `detect`, `refine`, and `curate`.
- Include `strain_table` row count and `reference_lattice` summary keys in `to_manifest_dict()`.

This is slightly more explicit than storing everything inside `neighbor_graph` or `local_metrics`, but it avoids ambiguous table semantics and keeps export/notebook integration clean.

## 2. Notebook workflow audit

### Existing notebooks

- `00_02_single_channel_end_to_end.ipynb`: merged single-channel import, detection, optional candidate review, refinement, curation, active session update, and optional checkpoint.
- `00_02_hfo2_multichannel_end_to_end.ipynb`: merged HfO2 multichannel workflow using helper functions for session initialization, HAADF heavy-column detection/review, iDPC light-column detection/review, overview, refinement/curation, and optional checkpoint.
- `03_vpcf_local_order.ipynb`: current recommended single-channel `03`, computes local/global/ROI vPCF from curated points and writes `session.vpcf_results`.
- `03_lattice_and_local_metrics.ipynb`: legacy neighbor graph / local metrics / strain-like workflow. It is explicitly marked `LEGACY`.
- `04_structure_domain_annotation.ipynb`: loads a metrics-stage-or-later session, generates cluster suggestions or applies manual polygon labels, and plots a domain annotation map.
- `05_publication_figures.ipynb`: loads a metrics-stage-or-later session, generates static figures, and optionally runs formal export.
- `06_polarization_and_vector_mapping_optional.ipynb`: optional vector-field demo based on curated points.

### Naming convention

Notebook filenames use a numeric workflow prefix plus a snake_case role:

- `00_02_..._end_to_end.ipynb` for merged early-stage workflows.
- `03_...`, `04_...`, `05_...`, `06_...` for downstream analysis stages.
- `_optional` marks optional workflows.
- Legacy notebooks remain in place but are clearly marked in Markdown rather than removed.

### Chinese Markdown and section style

Current notebooks use Chinese-facing Markdown:

- Top-level heading: `# NN 中文标题`.
- Numbered sections: `## 1. ...`, `## 2. ...`.
- Optional sections: `## 可选：...`.
- Opening paragraph starts with `本 notebook...`.
- Common explanatory labels include `使用约定：`, `这一步...`, and `这一格...`.
- Code terms and file names are wrapped in backticks.
- Bullets are short, practical workflow notes rather than long theory blocks.

### User-editable parameters

User-facing parameters are exposed in top-level code cells near the start of each notebook.

- Constants use uppercase names, for example `DATA_PATH`, `SESSION_PATH`, `USE_KEEP_POINTS`, `R_MAX_PX`, `OPEN_*_VIEWER`, `SAVE_FINAL_CHECKPOINT`, and `CHECKPOINT_FILENAME`.
- Structured algorithm parameters use config dataclasses, for example `DetectionConfig`, `RefinementConfig`, `CurationConfig`, `VPCFConfig`, `LatticeConfig`, `MetricsConfig`, and `ExportConfig`.
- Manual inputs use plain editable variables such as `POLYGONS`, `LABELS`, and path variables.

### Session loading and reuse

- The merged `00_02` notebooks create a fresh `AnalysisSession` and save it to the active session files.
- Downstream notebooks load reusable state with `load_or_connect_session(RESULT_ROOT, required_stage=..., session_path=SESSION_PATH)`.
- Active session files are `results/_active_session.pkl` and `results/_active_session.json`.
- Explicit checkpoints are saved under `results/<session_name>/checkpoints/<filename>`.
- `save_session_snapshot()` exists in utilities, but the inspected notebooks primarily use active sessions and optional checkpoints.

### Checkpoints, snapshots, and export

- Main stages call `save_active_session()` after successful computation.
- Optional final checkpoint cells call `save_checkpoint()` when enabled.
- `05_publication_figures.ipynb` uses `ExportConfig` and `export_results()` behind a `RUN_EXPORT` switch.
- `export_results()` currently writes atom and metrics tables, annotations, selected figures, a manifest, and optionally the session pickle.

### Plotting calls

Notebooks call plotting functions directly or through `notebook_workflows.py`.

- Image/point overlays: `plot_raw_image()`, `plot_atom_overlay()`.
- Local metrics: `plot_neighbor_graph()`, `plot_metric_map()`, `plot_histogram_or_distribution()`.
- Annotation/vector/vPCF: `plot_domain_annotation()`, `plot_vector_field()`, `plot_vpcf()`.
- Multichannel helper stages return figures through `NotebookResult`, then display them with `display_notebook_result()`.

### Notebook smoke tests

Notebook smoke tests exist in `tests/test_notebook_smoke.py`.

- They load notebook JSON and inspect code/Markdown strings.
- They compile selected code cells with `ast.parse`.
- They verify the merged single-channel and HfO2 notebooks use the intended workflow calls.
- They verify `03_vpcf_local_order.ipynb` is the recommended `03`.
- They verify `03_lattice_and_local_metrics.ipynb` is marked `LEGACY`.
- Optional old split notebooks are skipped when absent.

To extend them, add the new local affine strain notebook to the compile smoke test and add assertions for its Chinese headings, parameter constants, `load_or_connect_session`, active-session update, plotting calls, and default export/checkpoint switches.

## 3. Proposed notebook integration for local affine strain

### Recommended filename

Use `04_local_affine_strain.ipynb` for the main workflow notebook.

Rationale: local affine strain should run after curated coordinates and before structure/domain annotation and publication export. The current `04_structure_domain_annotation.ipynb` already expects a metrics-stage-or-later session, so the strain notebook should become the upstream stage in a future notebook-numbering cleanup. Until that cleanup, keep existing notebooks unchanged and document that this new `04` is the new strain entry point.

### Recommended section order

1. `# 04 局域仿射应变`
2. `## 1. 连接 curated session 与参数配置`
3. `## 2. 参考晶格定义与预览`
4. `## 3. 计算局域仿射应变`
5. `## 4. 应变图像 QA`
6. `## 5. 拟合质量与分布 QA`
7. `## 6. 可选导出与最终 checkpoint`

### Recommended top-level parameter cell

Use uppercase user-editable constants plus future config dataclasses:

```python
SESSION_PATH = None

USE_KEEP_POINTS = True
ROLE_FILTER = None  # 例如: 'light_atom' / 'heavy_atom' / None
REFERENCE_MODE = 'median_local_basis'  # 'manual_basis', 'roi_median', 'atom_ids'
REFERENCE_BASIS_PX = None  # 例如: ((10.2, 0.0), (0.0, 9.8))
REFERENCE_ATOM_IDS = None
REFERENCE_ROI_X_RANGE = None
REFERENCE_ROI_Y_RANGE = None

NEIGHBOR_K = 6
MAX_NEIGHBOR_DISTANCE_PX = None
MIN_NEIGHBORS = 3
MIN_BASIS_ANGLE_DEG = 25.0

DISPLAY_COMPONENT = 'strain_exx'
SAVE_ACTIVE_SESSION = True
RUN_EXPORT = False
SAVE_FINAL_CHECKPOINT = False
CHECKPOINT_FILENAME = '04_local_affine_strain.pkl'
```

When the API exists, wrap the related values in `ReferenceLatticeConfig` and `LocalAffineStrainConfig` in a separate short code cell, matching the existing config style.

### Recommended notebook outputs

Display these outputs in cells:

- Session summary: session name, stage, workflow mode, calibration, point count, keep-filter count, role filter.
- Point preview: first rows of the filtered atom table.
- Reference lattice summary: basis vectors, lengths, angle, unit, source mode, selected ROI or atom IDs.
- `strain_table.head()` with core columns and QC columns.
- Strain component maps over the image for `strain_exx`, `strain_eyy`, `strain_exy`, rotation, and dilatation if available.
- Histograms for selected strain components and residual/condition metrics.
- QA table for fit status: neighbor count, valid fit count, rejected/NaN count, median residual, high condition-number count.
- Active-session/checkpoint/export paths when written.

### Plotting and export integration

- Reuse `plot_metric_map()` initially for scalar strain components.
- Add thin plotting helpers only if repeated notebook code becomes noisy, for example `plot_strain_component_map()` and `plot_strain_summary_panels()`.
- Save results back to the session with `save_active_session()` so downstream notebooks can load the strain stage.
- Extend `export_results()` to write `tables/strain.*` when `session.strain_table` is non-empty.
- Extend publication export to save separate strain component maps, not only the current single `strain_exx` map.
- Keep `RUN_EXPORT = False` by default in the notebook, following `05_publication_figures.ipynb`.

## 4. Testing plan

### API tests

Add tests for:

- Public imports in `src/em_atom_workbench/__init__.py` once new configs/functions exist.
- `AnalysisSession` pickle roundtrip with `reference_lattice` and `strain_table`.
- Legacy pickle backfill for sessions without those fields.
- `clear_downstream_results()` clearing strain outputs when upstream coordinates change.
- Manifest serialization reporting strain table size and reference lattice presence.
- No mutation of `candidate_points`, `refined_points`, or `curated_points` during strain computation.

### Reference lattice tests

Add tests for:

- Manual reference basis in px.
- Reference basis inferred from median local basis vectors.
- ROI-filtered reference basis selection.
- Atom-ID-filtered reference basis selection.
- Rejection of singular, collinear, or missing reference bases.
- Deterministic output independent of atom table row order when `atom_id` is stable.
- Correct unit metadata when pixel calibration is missing, in px, or calibrated.

### Local affine strain tests

Add tests for:

- Perfect square/rectangular grid returns near-zero strain.
- Known affine deformation returns expected `exx`, `eyy`, shear, and rotation within tolerance.
- Pure translation returns near-zero strain.
- Uniform rotation is separated from symmetric strain when rotation output is included.
- Too few neighbors yields NaNs plus clear QC/status columns instead of crashing.
- `keep == False` points can be excluded through the API.
- `column_role`/`seed_channel` filters work for HfO2 heavy/light tables.
- Existing `neighbor_graph` can be reused when valid, but rebuilt when absent.
- Output schema includes `atom_id`, `x_px`, `y_px`, strain components, deformation/rotation diagnostics, neighbor count, residual, and status.

### Notebook smoke tests

Add or update tests in `tests/test_notebook_smoke.py` for:

- New notebook file exists and JSON loads.
- Code cells compile with `ast.parse`.
- Markdown includes Chinese title and sections for connection, reference lattice, strain computation, QA, and checkpoint/export.
- Joined source includes `SESSION_PATH`, `USE_KEEP_POINTS`, `REFERENCE_MODE`, `DISPLAY_COMPONENT`, `SAVE_FINAL_CHECKPOINT`, and `RUN_EXPORT`.
- Joined source includes `load_or_connect_session`, `save_active_session`, plotting calls, and the future local affine strain API call.
- Default export and checkpoint switches remain `False`.
- Code cells stay compact, following the helper-driven style used by the merged HfO2 notebook.
