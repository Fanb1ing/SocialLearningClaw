"""Viewer contract for level-boundary terminal frames.

The harness trace stores frame[-1] after a level-completing action, which is the
next playable level, not the solved level's terminal frame. The viewer must use
workspace terminal.json when labelling a boundary panel as terminal/final.
"""

from __future__ import annotations

from pathlib import Path


VIZ = Path(__file__).resolve().parents[2] / "tycho" / "viewer" / "viz.py"


def _check_boundary_terminal_source() -> dict[str, bool]:
    src = VIZ.read_text()
    return {
        "has_terminal_grid_helper": "function terminalGridForBoundary(s)" in src,
        "helper_reads_previous_level_terminal_json": (
            "level_${prevLevel}/terminal.json" in src
            and "JSON.parse(body)" in src
            and "obj.terminal_grid" in src
        ),
        "helper_prefers_embedded_terminal_txt": (
            "function parseGridText(txt)" in src
            and "level_${prevLevel}/terminal.txt" in src
            and "parseGridText(resolveContent(txtKey,c[txtKey]))" in src
        ),
        "helper_accepts_server_injected_terminal_grid": "isGrid(s.boundary_terminal_grid)" in src,
        "has_completed_step_terminal_helper": "function terminalGridForCompletedStep(s)" in src,
        "winning_step_shows_terminal_grid": (
            "const completedTerm = s.just_completed ? terminalGridForCompletedStep(s) : null;" in src
            and "completed level (L${s.level}) observed TERMINAL frame" in src
        ),
        "boundary_prefers_terminal_grid": (
            "const term=terminalGridForBoundary(s);" in src
            and "const prev=term || DATA()[i-1].grid;" in src
        ),
        "terminal_label_requires_terminal_evidence": "observed TERMINAL frame" in src,
        "legacy_fallback_not_labelled_final": (
            "terminal evidence not captured in this legacy record" in src
            and "const prev=DATA()[i-1].grid;   // the prior level's terminal frame" not in src
        ),
    }


def test_level_boundary_terminal_panel() -> None:
    failed = [name for name, passed in _check_boundary_terminal_source().items() if not passed]
    assert not failed


def main() -> int:
    ok = True
    checks = _check_boundary_terminal_source()
    print("=== level_boundary_terminal_panel ===")
    for name, passed in checks.items():
        print(f"  {name}: {passed}")
        ok = ok and passed
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
