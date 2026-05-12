from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from em_atom_workbench.export import export_results
from em_atom_workbench.session import AnalysisSession, ExportConfig

matplotlib.use("Agg")


def test_export_results_writes_manifest_tables_and_figures(tmp_path):
    session = AnalysisSession(name="export_case", raw_image=np.zeros((32, 32), dtype=float))
    session.curated_points = pd.DataFrame(
        {
            "atom_id": [0, 1],
            "x_px": [10.0, 20.0],
            "y_px": [12.0, 18.0],
            "annotation_label": ["domain_a", "domain_b"],
        }
    )
    session.local_metrics = pd.DataFrame(
        {
            "atom_id": [0, 1],
            "mean_nn_distance_px": [3.2, 3.4],
            "strain_exx": [0.01, -0.02],
        }
    )
    session.annotations = {
        "records": [
            {"type": "domain", "label": "domain_a", "polygon": [[0, 0], [16, 0], [16, 16], [0, 16]]}
        ]
    }

    manifest_path = export_results(
        session,
        ExportConfig(
            output_dir=tmp_path,
            export_profile="publication",
            save_annotations=True,
            overwrite=True,
        ),
    )

    export_root = manifest_path.parent
    assert manifest_path.exists()
    assert any(export_root.glob("tables/atoms.*"))
    assert any(export_root.glob("tables/metrics.*"))
    assert (export_root / "annotations" / "annotations.json").exists()
    assert (export_root / "figures" / "atom_overlay.png").exists()
    assert (export_root / "figures" / "atom_overlay.pdf").exists()
    assert (export_root / "figures" / "atom_overlay.tiff").exists()

