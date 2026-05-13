from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class ARCAGI3EnvWrapper:
    """Wrapper around ARC-AGI-3 interactive environment.

    Manages a single game across multiple levels.
    Each step: agent observes grid state, chooses an action, env returns new state.
    """

    def __init__(self, game_id: str, render_mode: Optional[str] = None):
        import arc_agi
        from arcengine import GameState

        self.arc = arc_agi.Arcade()
        self.env = self.arc.make(game_id, render_mode=render_mode)
        self.game_id = game_id
        self.GameState = GameState

        if self.env is None:
            raise RuntimeError(f"Failed to create environment for {game_id}")

    def reset(self):
        """Reset the current environment and return initial observation."""
        return self.env.reset()

    def get_available_actions(self, obs=None) -> List:
        """Return list of available GameAction objects.

        If obs is provided, returns only actions currently available
        according to the observation. Otherwise returns all actions
        in the action space.
        """
        import arcengine

        if obs is not None and hasattr(obs, "available_actions"):
            return [arcengine.GameAction.from_id(aid) for aid in obs.available_actions]
        return list(self.env.action_space)

    def step(self, action, data: Optional[Dict] = None):
        """Execute one action and return observation."""
        return self.env.step(action, data=data or {})

    def get_scorecard(self):
        """Return current scorecard."""
        return self.arc.get_scorecard()

    @staticmethod
    def grid_to_text(grid: np.ndarray, max_size: int = 16) -> str:
        """Convert a grid slice to compact text for LLM prompt.

        Truncates to max_size x max_size centered region to save tokens.
        """
        h, w = grid.shape
        if h > max_size or w > max_size:
            cy, cx = h // 2, w // 2
            y0 = max(0, cy - max_size // 2)
            x0 = max(0, cx - max_size // 2)
            grid = grid[y0 : y0 + max_size, x0 : x0 + max_size]

        lines = []
        for row in grid:
            lines.append(" ".join(str(int(v)) for v in row))
        return "\n".join(lines)

    @staticmethod
    def extract_objects(grid: np.ndarray) -> List[Dict]:
        """Extract connected-component objects from the grid.

        Returns list of dicts: {color, top_left, bottom_right, area, centroid}.
        """
        from scipy import ndimage

        objects = []
        visited = np.zeros_like(grid, dtype=bool)
        unique_vals = sorted(set(grid.flatten()))

        for val in unique_vals:
            if val == 0:  # Background often 0
                continue
            mask = grid == val
            labeled, num_features = ndimage.label(mask)
            for i in range(1, num_features + 1):
                component = labeled == i
                ys, xs = np.where(component)
                if len(xs) == 0:
                    continue
                obj = {
                    "color": int(val),
                    "top_left": (int(xs.min()), int(ys.min())),
                    "bottom_right": (int(xs.max()), int(ys.max())),
                    "area": int(component.sum()),
                    "centroid": (float(xs.mean()), float(ys.mean())),
                }
                objects.append(obj)
        return objects
