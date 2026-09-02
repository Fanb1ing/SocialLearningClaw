"""Shared executable-model analysis used by viewer overlays.

The implementation remains in the harness because result generation and offline diagnostics use the
same replay logic.
"""

from tycho.harness.model_replay import (
    find_workspaces,
    predict_for_workspace,
    predict_with_model_src,
)

__all__ = ["find_workspaces", "predict_for_workspace", "predict_with_model_src"]
