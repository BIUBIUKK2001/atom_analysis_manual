from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from em_atom_workbench.simple_quant_plotting import (
    plot_line_spacing_blocks_on_image,
    plot_pair_blocks_on_image,
)


def test_plot_pair_blocks_on_image_returns_fig_ax_and_patches() -> None:
    image = np.zeros((20, 20), dtype=float)
    pair_table = pd.DataFrame(
        {
            "source_x_px": [2.0, 8.0],
            "source_y_px": [3.0, 3.0],
            "target_x_px": [8.0, 14.0],
            "target_y_px": [3.0, 3.0],
            "distance_px": [6.0, 6.0],
        }
    )

    fig, ax = plot_pair_blocks_on_image(image, pair_table, value_column="distance_px")

    assert fig is not None
    assert ax is not None
    assert ax.collections
    assert len(ax.collections[-1].get_paths()) == len(pair_table)
    plt.close(fig)


def test_plot_line_spacing_blocks_on_image_uses_valid_next_atoms() -> None:
    image = np.zeros((20, 20), dtype=float)
    line_spacing = pd.DataFrame(
        {
            "x_px": [2.0, 8.0],
            "y_px": [3.0, 3.0],
            "next_atom_id": [1, np.nan],
            "next_x_px": [8.0, np.nan],
            "next_y_px": [3.0, np.nan],
            "spacing_to_next_px": [6.0, np.nan],
        }
    )

    fig, ax = plot_line_spacing_blocks_on_image(image, line_spacing, value_column="spacing_to_next_px")

    assert fig is not None
    assert ax is not None
    assert ax.collections
    assert len(ax.collections[-1].get_paths()) == 1
    plt.close(fig)
