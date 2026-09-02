from __future__ import annotations

import os
from pathlib import Path

import pytest

from tycho.workspace.sandbox import PythonSandbox, _runtime_usable


@pytest.mark.skipif(not _runtime_usable("bwrap"), reason="bubblewrap is unavailable")
def test_bwrap_live_policy_and_secret_scrubbing(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-enter-model-workspace")
    runner = PythonSandbox(runtime="bwrap")

    policy = runner.check(require_isolation=True)
    result = runner.run_source(
        tmp_path,
        "import os\nprint(os.environ.get('OPENROUTER_API_KEY', '<unset>'))\n",
        timeout=5,
    )

    assert policy["outside_readable"] is False
    assert policy["network"] is False
    assert policy["workspace_write"] is True
    assert policy["root_read_only"] is True
    assert policy["cap_eff"] == "0000000000000000"
    assert policy["no_new_privs"] == "1"
    assert result.returncode == 0
    assert result.stdout.strip() == "<unset>"
    assert not list(Path(tmp_path).glob(".tycho-run-*.py"))
