"""Tycho actor using the EFPS-aware workspace without changing actor policy."""

from arcengine import GameState

from tycho.agent.agent import TychoAgent

from .tools import EFPSToolExecutor
from .workspace import EFPSGameWorkspace


class EFPSTychoAgent(TychoAgent):
    name = "tycho_efps"
    workspace_class = EFPSGameWorkspace
    executor_class = EFPSToolExecutor

    def note_final_observation(self, latest_frame, available_actions) -> None:
        """Persist the last nonterminal action outcome when the harness stops at a hard budget.

        Ordinary outcomes are recorded when ``choose_action`` receives the next frame. A bounded
        experiment has no next decision after its final committed action, so without this callback
        EFPS loses exactly the evidence most likely to explain a failed five-action probe. Winning
        and fatal frames keep using Tycho's dedicated terminal/death evidence channels.
        """
        if latest_frame.state in (GameState.WIN, GameState.GAME_OVER):
            return
        grid = latest_frame.frame[-1]
        available = [getattr(action, "name", str(action)) for action in available_actions]
        state = str(latest_frame.state).replace("GameState.", "")
        self.ws.record(
            grid,
            level=self.level,
            turn_in_level=self.turn_in_level,
            action=getattr(self, "_last_action", None),
            row=getattr(self, "_last_row", None),
            col=getattr(self, "_last_col", None),
            state=state,
            available=available,
        )


def build_agent() -> EFPSTychoAgent:
    return EFPSTychoAgent()
