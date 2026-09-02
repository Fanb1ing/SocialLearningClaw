"""Declarative specifications for Tycho's orchestration policies and agent graphs.

A `ModeSpec` controls the actor's world-model instructions, model ownership, actor-invoked
delegation, and harness-triggered repair independently. This keeps the four paper policies explicit
without scattering correlated booleans across the actor and runner.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModeSpec:
    """A Tycho orchestration mode.

    name              canonical mode name (== the TYCHO_MODE value and the manifest label).
    wm_variant        which world-model paragraph the ACTOR's system prompt carries:
                        "single"  -> _WM_PARAGRAPH ("you write world_model.py")
                        "orch"    -> _WM_PARAGRAPH_ORCH ("a builder owns it; you PULL it via invoke_builder")
                        "trigger" -> _WM_PARAGRAPH_TRIGGER ("a builder is maintained for you automatically;
                                     read its latest report; you do NOT call it")
                        "none"    -> no executable world-model paragraph (direct-reasoning baseline)
                      DECOUPLED from the builder switches so each combination is expressible.
    include_builder   the actor gets the `invoke_builder` TOOL (actor-pull). Orchestrator only.
    harness_fires_builder  the HARNESS fires the builder on a trigger condition (no actor tool). Trigger only.
                      A WorldModelBuilder OBJECT is constructed whenever EITHER builder switch is set
                      (see `needs_builder`).
    actor_writes_wm   whether the actor owns world_model.py; builder modes isolate those edits.
    """

    name: str
    wm_variant: str
    include_builder: bool
    actor_writes_wm: bool
    harness_fires_builder: bool = False

    @property
    def needs_builder(self) -> bool:
        """A WorldModelBuilder object is needed whether the actor PULLS it (include_builder) or the
        HARNESS pushes it (harness_fires_builder)."""
        return self.include_builder or self.harness_fires_builder


# The registry — the single source of truth for valid TYCHO_MODE values.
#   no_world_model direct-reasoning baseline: no executable world-model loop.
#   single         world model written by the actor inline.
#   orchestrator   world-model builder subagent pulled by the actor.
#   trigger        world-model builder fired automatically by the harness on verifier/outcome triggers.
MODES: dict[str, ModeSpec] = {
    "no_world_model": ModeSpec("no_world_model", wm_variant="none", include_builder=False, actor_writes_wm=False),
    "single":       ModeSpec("single", wm_variant="single", include_builder=False, actor_writes_wm=True),
    "orchestrator": ModeSpec("orchestrator", wm_variant="orch", include_builder=True, actor_writes_wm=False),
    "trigger":      ModeSpec("trigger", wm_variant="trigger", include_builder=False, actor_writes_wm=False,
                             harness_fires_builder=True),
}


def resolve_mode(mode: str) -> ModeSpec:
    """Look up a ModeSpec; reject unknown values instead of silently changing policy."""
    spec = MODES.get(mode)
    if spec is None:
        raise ValueError(
            f"TYCHO_MODE={mode!r} is not a recognized mode {sorted(MODES)}. "
            "Choose one of the listed policies.")
    return spec


# --------------------------------------------------------------------------------------------------
# Declarative orchestration graph: the config file's `orchestration.agents` block.
# A multi-agent topology can't round-trip through env vars (env has no place for "agent B fires on
# agent A's event"), so it's expressed as a list of agent declarations and validated HERE into a
# structure consumed by the agent's _build_agent_graph. It builds on the AgentSpec graph in
# events.py rather than replacing it.
#
# Schema (each list item):
#   {id: <name>, type: actor|builder|scribe, trigger: {...}, sees_full_history: bool}
# Known agent TYPES (a type ties an id to a known run() in the agent; unknown type → loud error, no
# half-built generality): 'actor' (implicit — the choose_action loop itself), 'builder' (the
# world-model builder, harness-fired), 'scribe' (the level-boundary consolidator). trigger forms:
#   {on: level_complete}                              — a lifecycle event
#   {on: action_submitted, when: wm_mispredicts}      — event + the cheap inaccuracy gate
#   {on: agent_done, agent: <id>}                     — fire after another agent completes
_KNOWN_AGENT_TYPES = frozenset({"actor", "builder", "scribe"})
_KNOWN_TRIGGER_EVENTS = frozenset({
    "game_start", "level_start", "level_complete", "game_complete",
    "wm_updated", "action_submitted", "game_over_reset", "agent_done"})
_KNOWN_TRIGGER_WHEN = frozenset({"wm_mispredicts"})


@dataclass(frozen=True)
class AgentDecl:
    """One validated agent declaration from the config file's orchestration.agents list."""
    id: str
    type: str                       # actor | builder | scribe
    on: str | None                  # a _KNOWN_TRIGGER_EVENTS value (None only for the implicit actor)
    when: str | None = None         # optional gate (wm_mispredicts) for action_submitted
    after_agent: str | None = None  # for on=agent_done: the id to fire after
    sees_full_history: bool = True


def resolve_agent_graph(orchestration: dict | None) -> list[AgentDecl] | None:
    """Validate the config file's `orchestration.agents` block into a list of AgentDecl, or return
    None if no declarative graph is given (→ the agent falls back to the MODES preset). Fails LOUD on
    an unknown type / event / gate (a typo in a multi-agent config must not silently drop an agent)."""
    if not orchestration:
        return None
    agents = orchestration.get("agents")
    if not agents:
        return None
    if not isinstance(agents, list):
        raise ValueError("orchestration.agents must be a list of agent declarations")
    out: list[AgentDecl] = []
    seen_ids: set[str] = set()
    for i, a in enumerate(agents):
        if not isinstance(a, dict) or "id" not in a or "type" not in a:
            raise ValueError(f"orchestration.agents[{i}] needs at least {{id, type}}")
        aid, atype = a["id"], a["type"]
        if atype not in _KNOWN_AGENT_TYPES:
            raise ValueError(f"orchestration.agents[{i}].type={atype!r} unknown "
                             f"(known: {sorted(_KNOWN_AGENT_TYPES)})")
        if aid in seen_ids:
            raise ValueError(f"orchestration.agents: duplicate id {aid!r}")
        seen_ids.add(aid)
        trig = dict(a.get("trigger") or {})
        # YAML 1.1 GOTCHA: a bare `on:` key parses to the boolean True (likewise off/yes/no). So
        # `trigger: {on: action_submitted}` arrives as {True: "action_submitted"}. Normalize it back
        # to the string key "on" so the user can write the natural `on:` without quoting.
        if True in trig and "on" not in trig:
            trig["on"] = trig.pop(True)
        on = trig.get("on", "action_submitted" if atype == "actor" else None)
        if atype != "actor" and on is None:
            raise ValueError(f"orchestration.agents[{i}] ({aid}): non-actor agents need trigger.on")
        if on is not None and on not in _KNOWN_TRIGGER_EVENTS:
            raise ValueError(f"orchestration.agents[{i}].trigger.on={on!r} unknown "
                             f"(known: {sorted(_KNOWN_TRIGGER_EVENTS)})")
        when = trig.get("when")
        if when is not None and when not in _KNOWN_TRIGGER_WHEN:
            raise ValueError(f"orchestration.agents[{i}].trigger.when={when!r} unknown "
                             f"(known: {sorted(_KNOWN_TRIGGER_WHEN)})")
        after = trig.get("agent") if on == "agent_done" else None
        if on == "agent_done" and after is None:
            raise ValueError(f"orchestration.agents[{i}] ({aid}): trigger.on=agent_done needs trigger.agent")
        out.append(AgentDecl(id=aid, type=atype, on=on, when=when, after_agent=after,
                             sees_full_history=bool(a.get("sees_full_history", atype in ("actor", "scribe")))))
    actor_count = sum(1 for d in out if d.type == "actor")
    if actor_count != 1:
        raise ValueError("orchestration.agents must include exactly one agent of type 'actor'")
    return out
