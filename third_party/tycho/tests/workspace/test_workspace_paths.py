from __future__ import annotations

import pytest

from tycho.workspace.workspace import GameWorkspace


def test_file_tools_reject_sibling_with_workspace_name_prefix(tmp_path) -> None:
    workspace = GameWorkspace("game01", root=str(tmp_path), render=False)
    sibling = tmp_path / "game01_backup"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("outside")

    with pytest.raises(ValueError, match="path escapes workspace"):
        workspace.read_file("../game01_backup/secret.txt")
    with pytest.raises(ValueError, match="path escapes workspace"):
        workspace.write_file("../game01_backup/new.txt", "outside")

    assert not (sibling / "new.txt").exists()
