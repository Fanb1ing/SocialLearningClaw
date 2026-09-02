"""Terminal-frame selection at ARC-AGI level boundaries."""

from __future__ import annotations

from tycho.harness.harness import _grid_to_first, _grid_to_list


def _check_level_completion_frame_list() -> dict[str, bool]:
    terminal = [[1, 1], [2, 2]]
    animation = [[3, 3], [3, 3]]
    next_playable = [[4, 4], [5, 5]]
    frames = [terminal, animation, next_playable]
    return {
        "terminal_uses_first_frame": _grid_to_first(frames) == terminal,
        "playable_uses_last_frame": _grid_to_list(frames) == next_playable,
    }


def _check_empty_frame_list() -> dict[str, bool]:
    return {
        "empty_first_is_none": _grid_to_first([]) is None,
        "empty_list_is_none": _grid_to_list([]) is None,
        "none_first_is_none": _grid_to_first(None) is None,
        "none_list_is_none": _grid_to_list(None) is None,
    }


def main() -> int:
    ok = True
    for group, checks in (
        ("level_completion_frame_list", _check_level_completion_frame_list()),
        ("empty_frame_list", _check_empty_frame_list()),
    ):
        print(f"=== {group} ===")
        for name, passed in checks.items():
            print(f"  {name}: {passed}")
            ok = ok and passed
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
