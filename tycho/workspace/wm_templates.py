"""Templates for the files seeded into each game workspace so the agent can model + plan.
Kept here (not as inline strings in workspace.py) so they're easy to maintain. The text
in these templates is written FOR THE AGENT — terse, task-focused, no project history.

Seeded files:
  world_model.py   the agent edits this (init_state/transition/render/goal + optional subgoals/
                   heuristic; actions() is PRE-FILLED with this game's available actions)
  wmlib.py         helpers (from tycho.workspace/wmlib_template.py) — imported, not edited
  verify.py        `python verify.py`         -> simulation_accuracy + first divergence + goal check
  plan.py          `python plan.py [auto|astar|bfs|subgoals]` -> canonically validated action
                   sequence plus a start-anchored handoff in notes/validated_plan.json
"""
from pathlib import Path

# These are PYTHON FILES seeded into the workspace (world_model.py / verify.py / plan.py), NOT LLM
# prompts — they live in tycho/workspace/templates/*.py.tmpl (moved out of tycho/prompts/, which is
# now strictly LLM prompts). Read verbatim; world_model keeps its {ACTIONS_LINE} placeholder, filled
# per game when written to disk (workspace._seed_worldmodel).
_TMPL_DIR = Path(__file__).resolve().parent / "templates"


def _load_template(name: str) -> str:
    return (_TMPL_DIR / f"{name}.py.tmpl").read_text()


WORLD_MODEL = _load_template("seed_world_model")   # {ACTIONS_LINE} filled per game
VERIFY = _load_template("seed_verify")
PLAN = _load_template("seed_plan")
