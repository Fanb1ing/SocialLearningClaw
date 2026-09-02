"""Event vocabulary and agent specifications for event-driven orchestration.

  - single   = one agent triggered on {GAME_START, ACTION_SUBMITTED} (the actor; writes world_model.py).
  - orchestrator = an actor that invokes a builder when useful.
  - trigger  = a BUILDER agent triggered on (ACTION_SUBMITTED ∧ world model has ≥1 inaccuracy) + the
               ACTOR triggered on AGENT_DONE("builder"); the harness fires the builder, not the actor.

Agents may be LLM-backed or Python-backed. The level-boundary consolidator uses the stable agent id
`scribe` for configuration compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class Event(str, Enum):
    """The lifecycle + agent-completion events a trigger condition can fire on. str-valued so they
    serialize cleanly into journals and snapshots."""

    GAME_START = "game_start"            # the game's first frame (after reset)
    LEVEL_START = "level_start"          # the first frame of any level (incl. level 0)
    LEVEL_COMPLETE = "level_complete"    # a level was just solved (the scribe's trigger)
    GAME_OVER_RESET = "game_over_reset"  # same-level reset after API GAME_OVER
    GAME_COMPLETE = "game_complete"      # the final level solved / game won
    WM_UPDATED = "wm_updated"            # world_model.py was edited (by any agent)
    ACTION_SUBMITTED = "action_submitted"  # the actor committed a scored action this turn
    AGENT_DONE = "agent_done"            # another agent finished — carries the finished agent's id


# A trigger condition: given the fired Event and a context dict, decide whether THIS agent runs now.
# Kept as a plain predicate (not a DSL) so conditions compose in Python and stay debuggable — e.g.
# the builder's condition is `lambda ev, ctx: ev is Event.ACTION_SUBMITTED and ctx["wm_has_inaccuracy"]`.
# For AGENT_DONE the predicate also inspects ctx["done_agent"] (the id that just completed).
TriggerFn = Callable[[Event, dict], bool]


class AgentKind(str, Enum):
    LLM = "llm"          # a prompt run over an LLM (actor, builder, scribe, …)
    PYTHON = "python"    # a non-LLM agent: a learned model, a solver, a QA pass — pure code


@dataclass(frozen=True)
class AgentSpec:
    """One agent in the orchestration graph.

    id              unique name; referenced by AGENT_DONE triggers (e.g. the actor triggers on the
                    builder's id).
    trigger         predicate over (Event, ctx) deciding when this agent runs.
    kind            LLM or PYTHON.
    sees_full_history  True → the agent appends to and reads the actor conversation; False →
                    it runs a fresh, bounded conversation seeded with a context SLICE (recent
                    frames/diffs from disk + relevant beliefs + the triggering payload). At most ONE
                    durable-conversation agent (the actor); all others ephemeral with disk-backed outputs.
    run             the callable that executes the agent's work given a context dict; returns a result
                    (e.g. the builder's advisory report, or an ActionChoice for the actor).
    subagents       ids this agent may PULL synchronously (the subagent toggle): the actor in subagent
                    mode has subagents=("builder",).
    """

    id: str
    trigger: TriggerFn
    kind: AgentKind = AgentKind.LLM
    sees_full_history: bool = True
    run: Optional[Callable[[dict], object]] = None
    subagents: tuple[str, ...] = field(default_factory=tuple)


# --- canonical trigger predicates (named, reusable, unit-testable) -------------------------------
def on_events(*events: Event) -> TriggerFn:
    """Fire whenever the event is one of `events` (the simple lifecycle triggers)."""
    wanted = frozenset(events)
    return lambda ev, ctx: ev in wanted


def on_agent_done(agent_id: str) -> TriggerFn:
    """Fire when a SPECIFIC agent completes (AGENT_DONE carrying ctx['done_agent'] == agent_id)."""
    return lambda ev, ctx: ev is Event.AGENT_DONE and ctx.get("done_agent") == agent_id


def builder_trigger() -> TriggerFn:
    """Canonical trigger-mode condition for a detected model divergence."""
    return lambda ev, ctx: ev is Event.ACTION_SUBMITTED and bool(ctx.get("wm_inaccuracy_fire"))
