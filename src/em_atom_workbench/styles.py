from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import matplotlib as mpl


@dataclass(frozen=True)
class FigureStyleConfig:
    """Shared figure style knobs for notebook publication exports."""

    fig_dpi: int = 600
    fig_formats: tuple[str, ...] = ("png", "pdf")
    font_family: tuple[str, ...] = ("Arial", "Helvetica", "DejaVu Sans")
    axis_label_size: float = 11.0
    tick_label_size: float = 9.0
    legend_size: float = 9.0
    line_width: float = 1.2
    marker_size: float = 12.0
    colorbar_label_size: float = 10.0
    panel_label_size: float = 16.0
    histogram_color: str = "#4c78a8"
    fit_color: str = "#d95f02"
    overlay_alpha: float = 0.70

    def normalized_formats(self) -> tuple[str, ...]:
        return tuple(str(fmt).lower().lstrip(".") for fmt in self.fig_formats)


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


def apply_publication_style(config: FigureStyleConfig | None = None) -> None:
    style = dict(PUBLICATION_STYLE)
    if config is not None:
        style.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": list(config.font_family),
                "axes.labelsize": config.axis_label_size,
                "xtick.labelsize": config.tick_label_size,
                "ytick.labelsize": config.tick_label_size,
                "legend.fontsize": config.legend_size,
                "savefig.dpi": config.fig_dpi,
            }
        )
    mpl.rcParams.update(style)


def coerce_figure_style(config: FigureStyleConfig | dict | None = None) -> FigureStyleConfig:
    if config is None:
        return FigureStyleConfig()
    if isinstance(config, FigureStyleConfig):
        return config
    payload = dict(config)
    if "fig_formats" in payload and not isinstance(payload["fig_formats"], tuple):
        payload["fig_formats"] = tuple(payload["fig_formats"])
    if "font_family" in payload and isinstance(payload["font_family"], str):
        payload["font_family"] = (payload["font_family"],)
    elif "font_family" in payload and isinstance(payload["font_family"], Iterable):
        payload["font_family"] = tuple(payload["font_family"])
    return FigureStyleConfig(**payload)
