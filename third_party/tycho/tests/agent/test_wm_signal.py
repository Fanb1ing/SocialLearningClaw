from __future__ import annotations

import ast

from tycho.agent.wm_signal import _WM_SIGNAL_PROBE


def test_wm_signal_probe_script_compiles() -> None:
    ast.parse(_WM_SIGNAL_PROBE)
