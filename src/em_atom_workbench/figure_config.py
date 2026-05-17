from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FigureExportSpec:
    name: str
    save: bool = True
    title: str | None = None
    show_title: bool = True
    show_class_legend: bool = True
    show_roi_legend: bool = True
    show_task_legend: bool = True
    show_basis_vectors: bool = True
    show_roi_outlines: bool = True
    font_family: str = "Arial"
    font_size: int = 9
    color: str | None = None
    formats: tuple[str, ...] = ("pdf", "png", "svg")
    dpi: int = 600
    bbox_inches: str = "tight"


def normalize_figure_spec(spec: dict[str, Any] | FigureExportSpec, *, name: str | None = None) -> dict[str, Any]:
    if isinstance(spec, FigureExportSpec):
        payload = asdict(spec)
    else:
        payload = dict(spec or {})
    if name is not None:
        payload.setdefault("name", name)
    payload.setdefault("name", "figure")
    payload.setdefault("save", True)
    payload.setdefault("title", None)
    payload.setdefault("show_title", True)
    payload.setdefault("show_class_legend", True)
    payload.setdefault("show_roi_legend", True)
    payload.setdefault("show_task_legend", True)
    payload.setdefault("show_basis_vectors", True)
    payload.setdefault("show_roi_outlines", True)
    payload.setdefault("font_family", "Arial")
    payload.setdefault("font_size", 9)
    payload.setdefault("formats", ("pdf", "png", "svg"))
    payload.setdefault("dpi", 600)
    payload.setdefault("bbox_inches", "tight")
    payload["formats"] = tuple(str(fmt).lower().lstrip(".") for fmt in payload["formats"])
    payload["dpi"] = int(payload["dpi"])
    payload["font_size"] = int(payload["font_size"])
    payload["save"] = bool(payload["save"])
    return payload


def apply_figure_text_style(fig: Any, spec: dict[str, Any] | FigureExportSpec) -> Any:
    payload = normalize_figure_spec(spec)
    family = payload.get("font_family")
    size = payload.get("font_size")
    for axis in getattr(fig, "axes", []):
        if family:
            axis.title.set_fontfamily(family)
            axis.xaxis.label.set_fontfamily(family)
            axis.yaxis.label.set_fontfamily(family)
            for tick in axis.get_xticklabels() + axis.get_yticklabels():
                tick.set_fontfamily(family)
        if size:
            axis.title.set_fontsize(size)
            axis.xaxis.label.set_fontsize(size)
            axis.yaxis.label.set_fontsize(size)
            for tick in axis.get_xticklabels() + axis.get_yticklabels():
                tick.set_fontsize(size)
            legend = axis.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_fontsize(size)
                    if family:
                        text.set_fontfamily(family)
    return fig
