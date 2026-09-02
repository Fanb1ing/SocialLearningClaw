"""Jinja2 prompt rendering — the single entry point for assembling Tycho's agent prompts.

WHY JINJA (replacing load_prompt + str.replace + the scattered _perception_sys/_wm_variant swaps in
agent.py): the prompt CONDITIONALS — which world-model paragraph the actor carries (single/orch/
trigger), whether the image is described as lossless (per-model vision encoder), whether the grid text
is inlined, whether the dev-only meta-reflection is appended, and the whole per-turn frame message —
used to live as Python `.replace()`/`if` logic in agent.py, INVISIBLE to anyone editing the .md. Now
they are `{% if %}` blocks IN the template, where you edit them. One template per agent surface.

StrictUndefined: an undefined variable raises at render (parity with the old load_prompt, which raised
on a leftover {KEY}) — a typo'd/renamed context var fails LOUD, never ships a blank.

trim_blocks/lstrip_blocks: so `{% if %}` tag lines don't leave stray blank lines — lets the templates
read naturally while producing tight prose. (The equivalence test pins exact output regardless.)
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_DIR = Path(__file__).resolve().parent

_ENV = Environment(
    # search the prompts dir AND partials/ so a template can {% include "wm_single.j2" %} the
    # mode-specific world-model paragraph / meta-reflection fragment by bare name.
    loader=FileSystemLoader([str(_DIR), str(_DIR / "partials")]),
    undefined=StrictUndefined,        # undefined var -> error (fail-loud, like the old load_prompt)
    trim_blocks=True,                 # a block tag's trailing newline is removed
    lstrip_blocks=True,               # whitespace before a block tag is stripped
    keep_trailing_newline=True,       # don't silently drop a file's final newline
    autoescape=False,                 # these are LLM prompts, not HTML — never escape
)


def render_prompt(name: str, **ctx) -> str:
    """Render tycho/prompts/<name>.j2 with the given context. `name` has no extension.
    Raises jinja2.UndefinedError on a missing context variable (catches a renamed var before it
    reaches the model), TemplateNotFound if the .j2 is missing.

    Strips ONE trailing newline: template files conventionally end with a newline (POSIX text-file
    hygiene), but the assembled prompts they replace were Python strings with no trailing newline, so
    we drop exactly one to stay byte-identical. (Interior blank lines are preserved.)"""
    out = _ENV.get_template(f"{name}.j2").render(**ctx)
    return out[:-1] if out.endswith("\n") else out
