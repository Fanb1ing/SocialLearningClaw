"""Approach-agnostic agent interface.

Every approach subclasses `Agent` and implements `choose_action` (and optionally
`is_done` / `reset`). The harness owns the engine loop and scoring; an approach
only sees frames and returns actions. This mirrors the official ARC-AGI-3
`Agent` contract so an approach can be carried to another official runtime with minimal change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from arcengine import FrameData, GameAction


@dataclass
class ActionChoice:
    """An action plus optional ACTION6 coordinates and free-form reasoning.

    `reasoning` is recorded in the trace and passed to the engine, but does NOT
    count as an action under RHAE — only the committed move does.
    """

    action: GameAction
    x: Optional[int] = None
    y: Optional[int] = None
    #: Free-form per-step internals recorded in the trace. Analysis tools may interpret
    #: `predicted_frame`, `world_model_code`, `plan`, `goal`, and `cells_highlight`;
    #: unrecognized keys remain ordinary JSON data.
    reasoning: Optional[dict] = None

    @property
    def data(self) -> Optional[dict]:
        if self.action == GameAction.ACTION6:
            if self.x is None or self.y is None:
                raise ValueError("ACTION6 requires x and y in 0-63")
            return {"x": int(self.x), "y": int(self.y)}
        return None


class Agent:
    """Base class for all approaches.

    The harness calls, per environment: `reset()` once, then `choose_action`
    each turn until WIN / GAME_OVER / action budget / `is_done`.
    """

    #: Stable identifier used in result records; override per approach.
    name: str = "base"

    def reset(self, game_id: str, available_actions: list[GameAction]) -> None:
        """Called once at the start of each environment. Clear per-env state here."""

    def choose_action(
        self,
        frames: list[FrameData],
        latest_frame: FrameData,
        available_actions: list[GameAction],
    ) -> ActionChoice:
        raise NotImplementedError

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Optional early stop. Default: never stop early (harness enforces budget)."""
        return False
