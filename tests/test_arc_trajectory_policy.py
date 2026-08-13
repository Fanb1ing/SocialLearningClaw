from __future__ import annotations

import unittest

import numpy as np

from socialclaw.trajectory.arc_policies import (
    CD82PaintOperation,
    apply_cd82_program,
    cd82_goal_mask,
    cd82_navigation_actions,
    detect_cd82_palette,
    solve_cd82_target,
)


class CD82TrajectoryPolicyTests(unittest.TestCase):
    def test_reverse_solver_recovers_half_and_detail_program(self) -> None:
        expected = apply_cd82_program(
            [
                CD82PaintOperation("paint", 4, 15),
                CD82PaintOperation("detail", 0, 12),
            ],
            include_detail=True,
        )
        program = solve_cd82_target(
            expected,
            palette_colors=[0, 12, 15],
            include_detail=True,
        )
        actual = apply_cd82_program(program, include_detail=True)
        relevant = cd82_goal_mask()
        np.testing.assert_array_equal(actual[relevant], expected[relevant])

    def test_palette_detection_uses_visible_five_by_five_buttons(self) -> None:
        grid = np.full((64, 64), 5, dtype=np.int16)
        for left, color in ((20, 0), (26, 15), (32, 12)):
            grid[2:7, left : left + 5] = 4
            grid[3:6, left + 1 : left + 4] = color
        self.assertEqual(
            detect_cd82_palette(grid),
            {0: (22, 4), 15: (28, 4), 12: (34, 4)},
        )

    def test_navigation_path_reaches_target(self) -> None:
        self.assertEqual(
            [item.name for item in cd82_navigation_actions(0, 4)],
            ["ACTION3", "ACTION2", "ACTION2", "ACTION4"],
        )


if __name__ == "__main__":
    unittest.main()
