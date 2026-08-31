"""ARC grid utilities shared by the active layered-schema runners."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _simple_label(grid: np.ndarray, value: int) -> Tuple[np.ndarray, int]:
    """Label four-neighbour connected components for one cell value."""
    height, width = grid.shape
    mask = grid == value
    visited = np.zeros_like(mask, dtype=bool)
    labels = np.zeros_like(mask, dtype=int)
    label_id = 0

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            label_id += 1
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                current_y, current_x = stack.pop()
                labels[current_y, current_x] = label_id
                for delta_y, delta_x in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    next_y = current_y + delta_y
                    next_x = current_x + delta_x
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and mask[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
    return labels, label_id


def extract_grid_objects(grid: np.ndarray) -> List[Dict]:
    """Extract non-background connected components from an ARC grid."""
    objects: List[Dict] = []
    for value in sorted(set(int(cell) for cell in grid.flatten())):
        if value == 0:
            continue
        labels, component_count = _simple_label(grid, value)
        for component_id in range(1, component_count + 1):
            ys, xs = np.where(labels == component_id)
            if len(xs) == 0:
                continue
            objects.append(
                {
                    "color": value,
                    "top_left": (int(xs.min()), int(ys.min())),
                    "bottom_right": (int(xs.max()), int(ys.max())),
                    "area": int(len(xs)),
                    "centroid": (float(xs.mean()), float(ys.mean())),
                }
            )
    return objects


def color_name(color: int) -> str:
    """Map an ARC color index to a descriptive name."""
    palette = {
        0: "Black",
        1: "Blue",
        2: "Red",
        3: "Green",
        4: "Yellow",
        5: "Gray",
        6: "Pink",
        7: "Orange",
        8: "Cyan",
        9: "Maroon",
        10: "Beige",
        11: "Lime",
        12: "Indigo",
        13: "Brown",
        14: "Magenta",
        15: "White",
    }
    return palette.get(color, f"Color{color}")


def compute_grid_diff(
    pre_grid: np.ndarray | None,
    post_grid: np.ndarray | None,
) -> Tuple[bool, List[Dict]]:
    """Return whether the grid changed and a coarse changed-region summary."""
    if pre_grid is None or post_grid is None:
        return False, []
    if pre_grid.shape != post_grid.shape:
        return True, [
            {
                "top_left": (0, 0),
                "bottom_right": (
                    max(pre_grid.shape[1], post_grid.shape[1]) - 1,
                    max(pre_grid.shape[0], post_grid.shape[0]) - 1,
                ),
                "color_before": -1,
                "color_after": -1,
                "shape_before": list(pre_grid.shape),
                "shape_after": list(post_grid.shape),
            }
        ]

    changed_y, changed_x = np.where(pre_grid != post_grid)
    if len(changed_x) == 0:
        return False, []
    return True, [
        {
            "top_left": (int(changed_x.min()), int(changed_y.min())),
            "bottom_right": (int(changed_x.max()), int(changed_y.max())),
            "color_before": -1,
            "color_after": -1,
        }
    ]


__all__ = ["color_name", "compute_grid_diff", "extract_grid_objects"]
