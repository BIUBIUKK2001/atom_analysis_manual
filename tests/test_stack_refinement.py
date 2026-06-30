from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from em_atom_workbench.session import RefinementConfig
from em_atom_workbench.stack_refinement import refine_points_on_image, refine_stack_point_table


def _gaussian(shape: tuple[int, int], x: float, y: float, *, amplitude: float = 1.0, sigma: float = 1.2) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=float)
    return amplitude * np.exp(-0.5 * (((xx - x) / sigma) ** 2 + ((yy - y) / sigma) ** 2))


def _seed_points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "point_id": ["p0"],
            "atom_id": [0],
            "x_px": [20.0],
            "y_px": [22.0],
            "class_id": [1],
            "class_name": ["Hf"],
            "class_color": ["#abcdef"],
            "roi_id": ["global"],
        }
    )


def test_refine_stack_point_table_recovers_slice_dependent_positions() -> None:
    truth = [(20.0, 22.0), (20.8, 22.3), (21.6, 22.6)]
    stack = np.stack([_gaussian((48, 48), x, y) for x, y in truth])
    points = _seed_points()

    refined = refine_stack_point_table(
        points,
        stack,
        RefinementConfig(mode="legacy", fit_half_window=6, initial_sigma_px=1.2, max_center_shift_px=4.0),
    ).sort_values("slice_index")

    assert refined["slice_index"].tolist() == [0, 1, 2]
    assert refined["x_seed_px"].tolist() == pytest.approx([20.0, 20.0, 20.0])
    assert refined["y_seed_px"].tolist() == pytest.approx([22.0, 22.0, 22.0])
    assert refined["x_px"].tolist() == pytest.approx([x for x, _ in truth], abs=0.35)
    assert refined["y_px"].tolist() == pytest.approx([y for _, y in truth], abs=0.35)
    assert refined["class_name"].tolist() == ["Hf", "Hf", "Hf"]
    assert refined["class_color"].tolist() == ["#abcdef", "#abcdef", "#abcdef"]
    assert set(refined["coordinate_mode"]) == {"slice_refined"}


def test_class_refinement_overrides_recorded() -> None:
    image = _gaussian((48, 48), 20.0, 22.0) + _gaussian((48, 48), 30.0, 22.0)
    points = pd.DataFrame(
        {
            "atom_id": [0, 1],
            "x_px": [20.0, 30.0],
            "y_px": [22.0, 22.0],
            "class_id": [0, 1],
            "class_name": ["A", "B"],
        }
    )

    refined = refine_points_on_image(
        points,
        image,
        RefinementConfig(mode="legacy", fit_half_window=5, max_center_shift_px=2.0),
        class_refinement_overrides={1: {"fit_half_window": 4}},
    ).sort_values("atom_id")

    assert refined["refinement_config_source"].tolist() == ["default", "class_1"]
    assert refined["nn_context_mode"].tolist() == ["all", "all"]


def test_max_center_shift_rejects_large_spurious_jump() -> None:
    stack = np.stack([_gaussian((48, 48), 18.0, 20.0, amplitude=2.0)])
    points = pd.DataFrame({"atom_id": [0], "x_px": [10.0], "y_px": [20.0], "class_id": [0]})

    refined = refine_stack_point_table(
        points,
        stack,
        RefinementConfig(mode="legacy", fit_half_window=10, max_center_shift_px=1.0, initial_sigma_px=1.2),
    )
    row = refined.iloc[0]

    assert row["x_px"] == pytest.approx(10.0)
    assert row["y_px"] == pytest.approx(20.0)
    assert bool(row["center_shift_rejected"]) is True
    assert row["attempted_center_shift_px"] > 1.0
    assert row["position_source"] == "candidate_shift_guard"


def test_input_points_are_not_modified() -> None:
    stack = np.stack([_gaussian((32, 32), 12.0, 12.0)])
    points = pd.DataFrame({"atom_id": [0], "x_px": [12.0], "y_px": [12.0]})
    original = points.copy(deep=True)

    refine_stack_point_table(points, stack, RefinementConfig(mode="legacy", fit_half_window=5))

    pd.testing.assert_frame_equal(points, original)


def test_invalid_image_or_stack_raises_clear_value_error() -> None:
    with pytest.raises(ValueError, match="2D"):
        refine_points_on_image(pd.DataFrame({"x_px": [1.0], "y_px": [1.0]}), np.zeros((2, 4, 4)), RefinementConfig())
    with pytest.raises(ValueError, match="3D"):
        refine_stack_point_table(pd.DataFrame({"x_px": [1.0], "y_px": [1.0]}), np.zeros((4, 4)), RefinementConfig())
