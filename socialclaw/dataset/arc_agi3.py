from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


_ARC_COLOR_RGB = {
    0: (0, 0, 0),
    1: (0, 0, 255),
    2: (255, 0, 0),
    3: (0, 255, 0),
    4: (255, 255, 0),
    5: (128, 128, 128),
    6: (255, 192, 203),
    7: (255, 165, 0),
    8: (0, 255, 255),
    9: (128, 0, 0),
    10: (245, 245, 220),
    11: (50, 205, 50),
    12: (75, 0, 130),
    13: (165, 42, 42),
    14: (255, 0, 255),
    15: (255, 255, 255),
}


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
    def grid_to_text(grid: np.ndarray) -> str:
        """Convert a grid to compact text for LLM prompt.

        No truncation — the full grid is preserved because all regions may
        carry useful information.
        """
        lines = []
        for row in grid:
            lines.append(" ".join(str(int(v)) for v in row))
        return "\n".join(lines)

    @staticmethod
    def grid_to_image(grid: np.ndarray, cell_size: int = 8, grid_step: int = 8) -> "Image.Image":
        """Render a grid as a colour image for visual LLM input.

        Each cell is cell_size x cell_size pixels. Sparse gridlines are drawn
        every grid_step cells to help the LLM and human locate positions.
        Coordinate system: col (x) increases left→right from 0;
        row (y) increases top→bottom from 0.
        """
        from PIL import Image, ImageDraw

        h, w = grid.shape
        img_h, img_w = h * cell_size, w * cell_size
        img = Image.new("RGB", (img_w, img_h))
        pixels = img.load()
        for row in range(h):
            for col in range(w):
                colour = _ARC_COLOR_RGB.get(int(grid[row, col]), (128, 128, 128))
                for dy in range(cell_size):
                    for dx in range(cell_size):
                        pixels[col * cell_size + dx, row * cell_size + dy] = colour

        # Draw sparse gridlines every grid_step cells
        draw = ImageDraw.Draw(img)
        line_color = (80, 80, 80)
        for col in range(0, w + 1, grid_step):
            x_px = min(col * cell_size, img_w - 1)
            draw.line([(x_px, 0), (x_px, img_h - 1)], fill=line_color, width=1)
        for row in range(0, h + 1, grid_step):
            y_px = min(row * cell_size, img_h - 1)
            draw.line([(0, y_px), (img_w - 1, y_px)], fill=line_color, width=1)
        return img

    @staticmethod
    def extract_objects(grid: np.ndarray) -> List[Dict]:
        """Extract connected-component objects from the grid.

        Returns list of dicts: {color, top_left, bottom_right, area, centroid}.
        """
        # Keep one implementation of ARC object extraction. The schema parser
        # uses a dependency-free four-neighbour BFS and is the canonical path.
        from ..schema.arc_agi3_parser import extract_grid_objects

        return extract_grid_objects(grid)
