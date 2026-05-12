from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _maybe_ipywidgets() -> Any:
    try:
        import ipywidgets as widgets
    except ImportError:
        return None
    return widgets


def make_output_dir_widget(default_path: str | Path = "results") -> Any:
    widgets = _maybe_ipywidgets()
    if widgets is None:
        return str(default_path)
    return widgets.Text(value=str(default_path), description="OUTPUT_DIR")


def make_simple_markdown_hint(text: str) -> Any:
    widgets = _maybe_ipywidgets()
    if widgets is None:
        return text
    return widgets.HTML(f"<div style='padding:4px 0'>{text}</div>")


def dataframe_preview(table: pd.DataFrame, n_rows: int = 10) -> pd.DataFrame:
    return table.head(n_rows).copy()

