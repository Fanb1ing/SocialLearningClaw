"""Source-level contracts for viewer JavaScript that Python tests cannot execute directly."""

from __future__ import annotations

from pathlib import Path


VIZ = Path(__file__).resolve().parents[2] / "tycho" / "viewer" / "viz.py"


def _checks() -> dict[str, bool]:
    src = VIZ.read_text()
    return {
        "path_args_are_bound_not_interpolated": (
            "function bindArg(v)" in src
            and "toggleDirArg(${bindArg(fp)})" in src
            and "selFileArg(${bindArg(f.path)})" in src
            and "selDiskArg(${bindArg(rel)})" in src
            and "toggleDir('${esc(fp)}')" not in src
            and "selFile('${esc(f.path)}')" not in src
            and "selDisk('${esc(rel)}')" not in src
        ),
        "attribute_escape_available": (
            "function attr(t)" in src
            and 'title="${attr(fmtA(p))}"' in src
        ),
        "prediction_diff_uses_same_step_observation": (
            "const pred=_predGrid(s); const obs=s.grid;" in src
            and "const nxt=(i+1<DATA().length)?DATA()[i+1].grid:null;" not in src
            and "DATA()[i+1].grid" not in src
        ),
        "grid_diff_rejects_row_width_mismatch": "if(a[r].length!==b[r].length) return null;" in src,
        "agent_filter_resets_on_context_switch": (
            "loadGame(){G=document.getElementById('game').value;i=0;window._wsSel=null;window._agentF='all';" in src
            and "loadRun(){R=document.getElementById('run').value;MANIFEST=MAN(R);i=0;window._wsSel=null;window._agentF='all';" in src
        ),
        "workspace_history_uses_version_manifests": (
            "function resolveVersions(v)" in src
            and "const curVersions=_wsVersions(D[i]), prevVersions=_wsVersions(D[i-1]);" in src
            and "open exact historical blob" in src
        ),
        "responsive_layout_has_narrow_breakpoints": (
            "@media(max-width:1600px)" in src
            and "@media(max-width:900px)" in src
            and "@media(max-width:560px)" in src
            and "width:min(576px,100%)" in src
        ),
        "dynamic_text_and_paths_can_wrap": (
            ".frow{cursor:pointer;padding:1px 5px;border-radius:4px;white-space:normal;"
            in src
            and ".call{" in src
            and "overflow-wrap:anywhere" in src
        ),
        "reasoning_labels_do_not_guess_providers": (
            "if(opts && opts.kind) return opts.kind;" in src
            and "do not infer it from provider names" in src
            and "opts.backend" not in src
            and "const b=(opts&&opts.backend" not in src
        ),
    }


def test_viewer_template_contracts() -> None:
    failed = [name for name, passed in _checks().items() if not passed]
    assert not failed


def main() -> int:
    ok = True
    print("=== viewer_template_contracts ===")
    for name, passed in _checks().items():
        print(f"  {name}: {passed}")
        ok = ok and passed
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
