"""Trigger-mode helpers: when to fire the auto-builder, and what reason text to hand it.

NO RATE LIMIT. The builder fires whenever the world model mispredicts an observed transition; once
the model is correct, the inaccuracy probe returns ACCURATE and the builder doesn't fire — that
IS the rate limit. A turn-cap (the old `max_per_level=3` + `cooldown=5`) was wrong: it would stop
firing exactly when the builder needs more attempts (early-level evidence accumulating toward a
breakthrough). The natural circuit breaker is the game's overall LLM budget — `self.max_calls` —
which agent.py checks before each builder run.

Only `trigger` mode constructs a `TriggerDispatcher`; other policies do not enter this path.
"""

from __future__ import annotations

from tycho.agent.wm_signal import wm_signal


class TriggerDispatcher:
    """Tracks the per-level fire count (for the kickoff-message header) and runs the cheap divergence
    probe — no rate limiting; the builder fires whenever the model is detectably wrong on EITHER the
    dynamics channel (simulation_accuracy < 1.0) or the outcome channel (level_complete too loose,
    level_complete too tight, GAME_OVER missed/too loose, or terminal-render mismatch; see wm_signal).
    The latest probe verdict is cached so the kickoff can be tailored to the cause. Terminal-status
    issues are often first knowable once a terminal is observed, so LEVEL_START gives the builder an
    immediate repair pass after a solved level."""

    def __init__(self):
        # per-level fire count; used only in the kickoff to label the message
        self._fires_this_level: dict[int, int] = {}
        # cause of the most recent should_fire()==True, so trigger_reason() can tailor the kickoff:
        # "dynamics" | outcome divergence labels from wm_signal.
        self._last_cause: str | None = None

    def should_fire(self, level: int, executor) -> bool:  # noqa: ARG002 — level reserved for future per-level state
        """True iff the current world model is detectably wrong: dynamics mispredict OR outcome()
        diverges. One sandboxed subprocess records the cause for trigger_reason(). Level-start gives
        the builder a prompt after a solved level, but the per-action gate also keeps re-engaging while
        terminal status is wrong; once outcome() is consistent it returns ok and stops firing."""
        sig = wm_signal(executor)
        if sig["dynamics_inaccurate"]:
            self._last_cause = "dynamics"          # dynamics first: a wrong δ/ρ undermines outcome evidence
            return True
        od = sig["outcome_divergence"]
        if od in ("level_complete_too_loose", "level_complete_too_tight",
                  "game_over_missed", "game_over_too_loose", "terminal_render", "invalid"):
            self._last_cause = f"outcome_{od}"
            return True
        self._last_cause = None
        return False

    def note_fired(self, level: int) -> None:
        """Record that the builder fired on `level`."""
        self._fires_this_level[level] = self._fires_this_level.get(level, 0) + 1

    def trigger_reason(self, level: int) -> str:
        """A WHY string for the builder kickoff, tailored to the cause recorded by should_fire().
        Detail (first divergence / outcome misfires) is in the auto-attached [verify state] block."""
        n = self._fires_this_level.get(level, 0) + 1
        if self._last_cause == "outcome_level_complete_too_loose":
            return (f"[auto-trigger #{n} this level — OUTCOME] Dynamics look right, but outcome() returns "
                    f'"level_complete" on a decision frame where the level did NOT end. The planner will '
                    f"steer toward that phantom terminal. The [verify state] block shows "
                    f"level_complete_on_nonterminal>0 and the misfiring frames — tighten outcome() so "
                    f"only true level-ending states return level_complete.")
        if self._last_cause == "outcome_level_complete_too_tight":
            return (f"[auto-trigger #{n} this level — OUTCOME] Dynamics look right, but outcome() does "
                    f'not return "level_complete" for a level that was actually solved. The planner will '
                    f"not recognise the win. The [verify state] block shows level_complete_on_terminal<1.0 "
                    f"and which solved level it misses — widen outcome() to cover that terminal.")
        if self._last_cause == "outcome_game_over_too_loose":
            return (f"[auto-trigger #{n} this level — OUTCOME] Dynamics look right, but outcome() returns "
                    f'"game_over" on an ordinary decision frame. The [verify state] block shows '
                    f"game_over_on_nonterminal>0 — reserve game_over for observed API GAME_OVER states.")
        if self._last_cause == "outcome_game_over_missed":
            return (f"[auto-trigger #{n} this level — OUTCOME] Dynamics look right, but outcome() misses "
                    f'an observed API GAME_OVER transition. Mark the modeled post-fatal-action state as '
                    f'"game_over" so planning can avoid known-fatal edges.')
        if self._last_cause == "outcome_terminal_render":
            return (f"[auto-trigger #{n} this level — TERMINAL RENDER] Dynamics look right and outcome() "
                    f"recognises the solved terminal, but the modeled post-winning-action state does NOT "
                    f"render the observed winning grid. The [verify state] block shows "
                    f"terminal_render_exact<1.0. Fix transition()/render() at the winning action so the "
                    f"terminal model is grounded in the actual frame the API returned.")
        if self._last_cause == "outcome_invalid":
            return (f"[auto-trigger #{n} this level — OUTCOME] Dynamics look right, but outcome() raised "
                    f"or returned a value outside 'ongoing', 'level_complete', 'game_over'. Fix the "
                    f"world-model status function, then re-check verify.py.")
        return (f"[auto-trigger #{n} this level] A committed action's outcome diverged from the "
                f"current world_model.py prediction. Find the first graded transition the model "
                f"simulates wrong (run python verify.py for simulation_accuracy + first divergence) and "
                f"fix init_state/transition so simulation_accuracy climbs.")

    def level_start_reason(self, level: int) -> str:
        """A WHY string for a NEW-LEVEL kickoff. The level just changed: a new level can introduce
        colors, layout, or decode rules the prior-level model doesn't handle. Tell the builder to
        CHECK the new init frame against the current model rather than chase a divergence — AND to
        check the outcome channel, since the level that just ended is now an observed terminal that
        falsifies a too-tight level_complete condition."""
        return (f"[auto-trigger: level {level} just started] A new level began. It may introduce "
                f"new colors, objects, or rules the current world_model.py wasn't built for. If "
                f"notes/level_{level-1}_insights.md exists, read it. If the builder has derived a "
                f"reliable world model for the game, use it before reporting. Check the new init frame "
                f"and whether init_state/render/outcome still hold — refine them if the new level "
                f"breaks the model, or confirm they generalize and stop. The just-finished level is "
                f"now an observed terminal: check the outcome channel in "
                f"the [verify state] block — if outcome() did NOT return level_complete on it "
                f"(level_complete_on_terminal<1.0), widen it to cover that win. No observed transitions on this level yet, "
                f"so simulation_accuracy is n/a until the actor acts.")

    def game_over_reset_reason(self, level: int) -> str:
        """A WHY string for a same-level reset after API GAME_OVER."""
        n = self._fires_this_level.get(level, 0) + 1
        return (f"[auto-trigger #{n} this level — GAME_OVER RESET] The previous action caused API "
                f"GAME_OVER and the framework reset level {level}. This is a fresh start of the SAME "
                "level, not a new level. Read the death_*.json/death_*_diff evidence and update "
                "hazards/outcome/planning constraints before the actor retries; avoid treating the "
                "reset frame as the fatal action's normal next state.")

    # --- resume -------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {"fires_this_level": {str(k): v for k, v in self._fires_this_level.items()}}

    def restore(self, s: dict) -> None:
        self._fires_this_level = {int(k): v for k, v in (s.get("fires_this_level") or {}).items()}
