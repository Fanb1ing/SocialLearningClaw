"""EFPS audit feedback layered onto Tycho's normal world-model verifier."""

from pathlib import Path

from tycho.workspace.agent_tools import ToolExecutor


class EFPSToolExecutor(ToolExecutor):
    def _wm_feedback(self, path: str, before_ast=None) -> str:
        feedback = super()._wm_feedback(path, before_ast)
        if Path(path).name not in self._WM_FILES or not self.wm_feedback_enabled:
            return feedback
        audit_path = self.ws.dir / "efps_audit.py"
        if not audit_path.exists():
            return feedback
        audit = self._run_python(
            "exec(compile(open('efps_audit.py').read(), 'efps_audit.py', 'exec'))",
            timeout=20,
        )
        return feedback + "\n\n[auto EFPS audit]\n" + audit.strip()
