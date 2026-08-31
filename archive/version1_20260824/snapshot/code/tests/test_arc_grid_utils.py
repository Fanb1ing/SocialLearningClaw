from __future__ import annotations

import unittest

import numpy as np

from socialclaw.schema.arc_agi3_parser import (
    color_name,
    compute_grid_diff,
    extract_grid_objects,
)


class ArcGridUtilityTests(unittest.TestCase):
    def test_extracts_four_neighbour_components(self) -> None:
        grid = np.array(
            [
                [0, 2, 0],
                [0, 2, 0],
                [3, 0, 3],
            ]
        )
        objects = extract_grid_objects(grid)
        self.assertEqual(len(objects), 3)
        red = next(item for item in objects if item["color"] == 2)
        self.assertEqual(red["top_left"], (1, 0))
        self.assertEqual(red["bottom_right"], (1, 1))
        self.assertEqual(red["area"], 2)

    def test_color_name_has_unknown_fallback(self) -> None:
        self.assertEqual(color_name(8), "Cyan")
        self.assertEqual(color_name(99), "Color99")

    def test_grid_diff_handles_content_and_shape_changes(self) -> None:
        before = np.zeros((2, 2), dtype=int)
        after = before.copy()
        after[1, 0] = 4
        changed, regions = compute_grid_diff(before, after)
        self.assertTrue(changed)
        self.assertEqual(regions[0]["top_left"], (0, 1))
        self.assertEqual(regions[0]["bottom_right"], (0, 1))

        changed, regions = compute_grid_diff(before, np.zeros((3, 2), dtype=int))
        self.assertTrue(changed)
        self.assertEqual(regions[0]["shape_before"], [2, 2])
        self.assertEqual(regions[0]["shape_after"], [3, 2])


if __name__ == "__main__":
    unittest.main()
