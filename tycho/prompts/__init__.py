"""LLM prompts for Tycho — Jinja2 templates, one <agent>.<role>.j2 per prompt surface.

Each agent's prompt is a pair of templates: <agent>.system.j2 (the system prompt) and
<agent>.user.j2 (the user message). The actor's per-turn message is actor.user.j2; the builder's
kickoff is builder.user.j2; the level-boundary consolidator injects scribe.user.j2 into the actor's
conversation (it has no own system prompt). All conditionals (perception fidelity, the world-model-paragraph variant, the
meta-reflect addendum, the per-turn grid/diff/no-op toggles) are VISIBLE {% if %}/{% include %} in the
templates — edit them there. Shared fragments live in partials/ (the wm_* world-model paragraphs and
meta_reflect, {% include %}d by actor.system.j2).

Render via tycho.prompts.render.render_prompt(name, **ctx). Code TEMPLATES (the Python seeded into the
workspace: world_model.py / verify.py / plan.py / the wm-feedback probe) are NOT prompts and live in
tycho/workspace/templates/*.py.tmpl.
"""
