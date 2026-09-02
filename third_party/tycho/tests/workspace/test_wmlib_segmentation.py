from __future__ import annotations

import numpy as np

from tycho.workspace import wmlib_template as wmlib


def test_segment_defaults_to_compact_metadata_without_cells() -> None:
    grid = np.array(
        [
            [1, 1, 2],
            [1, 0, 2],
            [3, 3, 2],
        ],
        dtype=np.int16,
    )

    parts = wmlib.segment(grid, background=0)

    assert [p["color"] for p in parts] == [1, 2, 3]
    assert "cells" not in parts[0]
    assert parts[0]["size"] == 3
    assert parts[0]["bbox"] == (0, 0, 1, 1)
    assert parts[0]["shape"] == "0:0-1;1:0"
    assert parts[0]["adjacent"] == [1, 2]
    assert parts[1]["adjacent"] == [0, 2]
    assert parts[2]["adjacent"] == [0, 1]

    cell_parts = wmlib.segment(grid, background=0, include_cells=True)
    assert cell_parts[0]["cells"] == [(0, 0), (0, 1), (1, 0)]

    restored = wmlib.composite(cell_parts, 3, 3, background=0)
    np.testing.assert_array_equal(restored, grid)


def test_segment_keeps_exact_relative_shape_without_cells() -> None:
    grid = np.array(
        [
            [4, 4, 0, 5, 5],
            [4, 0, 0, 5, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.int16,
    )

    parts = wmlib.segment(grid, background=0)

    assert len(parts) == 2
    assert all("cells" not in p for p in parts)
    assert parts[0]["shape"] == parts[1]["shape"] == "0:0-1;1:0"
    assert parts[0]["bbox"] == (0, 0, 1, 1)
    assert parts[1]["bbox"] == (0, 3, 1, 4)


def test_segment_summary_is_compact_and_bounded() -> None:
    grid = np.array(
        [
            [1, 0, 2],
            [0, 0, 0],
            [3, 0, 4],
        ],
        dtype=np.int16,
    )

    summary = wmlib.segment_summary(grid, background=0, max_components=2)

    assert "grid=3x3 background=0 components=4" in summary
    assert "#0 color=1 size=1 bbox=(0,0)-(0,0)" in summary
    assert "#1 color=2 size=1 bbox=(0,2)-(0,2)" in summary
    assert "shape=rect" in summary
    assert "2 more components omitted" in summary
    assert "cells" not in summary
