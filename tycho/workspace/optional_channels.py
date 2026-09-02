"""Optional observation channels supplied by local deployments.

The paper artifact uses rendered images and exact text grids. Research deployments may add other
channels without changing the actor or the public workspace interface.
"""

from __future__ import annotations

import importlib


def _extension():
    try:
        return importlib.import_module("tycho.workspace._workspace_extension")
    except ModuleNotFoundError:
        return None


def grid_embeds_enabled() -> bool:
    extension = _extension()
    return bool(extension and extension.grid_embeds_enabled())


def grid_embeds_part(grid, model_dir: str | None = None) -> dict | None:
    extension = _extension()
    if extension is None:
        return None
    return extension.grid_embeds_part(grid, model_dir)
