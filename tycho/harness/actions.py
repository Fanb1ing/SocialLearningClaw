"""Action-list helpers for ARC-AGI-3 engine frames.

The public API exposes two related but distinct action sets:

* game actions: the game-defined ACTION1..ACTION7 entries from frame metadata;
* actor actions: game actions plus RESET, the API recovery control.

RESET is handled specially by the engine and is often not present in
FrameData.available_actions. Keep it out of world_model.actions() seeding, but
allow the actor to choose it on playable turns.
"""

from __future__ import annotations

from typing import Iterable

from arcengine import GameAction


def normalize_game_actions(actions: Iterable[object] | None) -> list[GameAction]:
    """Return the frame-declared game actions, excluding RESET."""
    out: list[GameAction] = []
    seen: set[GameAction] = set()
    for action in actions or []:
        ga = _as_game_action(action)
        if ga is GameAction.RESET or ga in seen:
            continue
        out.append(ga)
        seen.add(ga)
    return out


def actor_actions(actions: Iterable[object] | None) -> list[GameAction]:
    """Return legal actor actions: RESET plus the frame-declared game actions."""
    return [GameAction.RESET, *normalize_game_actions(actions)]


def _as_game_action(action: object) -> GameAction:
    if isinstance(action, GameAction):
        return action
    if isinstance(action, str):
        return getattr(GameAction, action)
    return GameAction.from_id(int(action))
