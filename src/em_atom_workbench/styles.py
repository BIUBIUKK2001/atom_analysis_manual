from __future__ import annotations

import matplotlib as mpl


PUBLICATION_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "image.cmap": "gray",
}


def apply_publication_style() -> None:
    mpl.rcParams.update(PUBLICATION_STYLE)

