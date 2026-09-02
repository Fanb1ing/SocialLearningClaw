from __future__ import annotations

import os
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from arcengine import GameAction


def _set_env(*, cap: int = 5, evidence_keep: int | None = None,
             summary: bool = True, vision: bool = True) -> dict:
    old = dict(os.environ)
    os.environ.update({
        "LLM_MODEL": "dummy",
        "LLM_BACKEND": "openai",
        "LLM_BASE_URL": "http://127.0.0.1:9",
        "LLM_API_KEY": "dummy",
        "MPLCONFIGDIR": "/private/tmp",
        "TYCHO_MODE": "single",
        "TYCHO_VISION": "1" if vision else "0",
        "TYCHO_RENDER_SCALE": "1",
        "TYCHO_TEXT_GRID": "full",
        "TYCHO_HISTORY": "tail_evict",
        "TYCHO_GRID_KEEP": "8",
        "TYCHO_IMAGE_KEEP": "8",
        "TYCHO_IMAGE_CAP": "8",
        "TYCHO_MAX_TOOL_STEPS": "1",
        "TYCHO_MAX_LLM_CALLS": "20",
        "TYCHO_ANIMATION_SUMMARY": "1" if summary else "0",
        "TYCHO_ANIMATION_SUMMARY_MAX_PER_LEVEL": str(cap),
        "TYCHO_ANIMATION_EVIDENCE_KEEP_PER_LEVEL": str(evidence_keep if evidence_keep is not None else cap),
        "TYCHO_ANIMATION_SUMMARY_MAX_KEYFRAMES": "5",
    })
    return old


def _restore_env(old: dict) -> None:
    os.environ.clear()
    os.environ.update(old)


def _frame(frames, levels_completed=0, state="NOT_FINISHED"):
    return SimpleNamespace(frame=frames, levels_completed=levels_completed, state=state)


def _text(history: list[dict]) -> str:
    chunks = []
    for msg in history:
        content = msg.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            chunks.extend(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
    return "\n".join(chunks)


def _image_count(history: list[dict]) -> int:
    n = 0
    for msg in history:
        content = msg.get("content")
        if isinstance(content, list):
            n += sum(1 for part in content if isinstance(part, dict) and part.get("image_png") is not None)
    return n


def _event_frames(kind: int = 0) -> list[list[list[int]]]:
    frames = []
    base = np.zeros((64, 64), dtype=np.int16)
    if kind == 0:
        base[10:45, 20:44] = 3
        for i in range(6):
            g = np.roll(base, shift=-i * 4, axis=0)
            g[: i + 1, :] = 9
            frames.append(g.tolist())
    else:
        base[20:44, 8:48] = 4
        for i in range(6):
            g = np.roll(base, shift=i * 5, axis=1)
            g[:, -i - 1:] = 8
            frames.append(g.tolist())
    return frames


def test_animation_summary_is_text_only_and_reuses_same_level_signature():
    old = _set_env(cap=5)
    try:
        from tycho.agent.agent import TychoAgent

        class StubTycho(TychoAgent):
            def __init__(self):
                super().__init__()
                self.summary_calls = 0
                self.actor_prompts: list[str] = []
                self.actor_image_counts: list[int] = []

            def _summarize_animation_with_llm(self, frames, decision, *, action: str, terminal: str) -> str:
                self.summary_calls += 1
                return f"summary {self.summary_calls}: viewport shifts and reveals new border content."

            def _chat(self, call_type, max_tokens=None):
                self.actor_prompts.append(_text(self.history))
                self.actor_image_counts.append(_image_count(self.history))
                return {"text": "act", "tool_calls": [
                    {"id": f"a{len(self.actor_prompts)}", "name": "take_action",
                     "input": {"action": "ACTION1"}},
                ]}

        with TemporaryDirectory() as td:
            agent = StubTycho()
            agent.reset("anim_summary", [GameAction.ACTION1], ws_root=td)

            agent.choose_action([], _frame([_event_frames()[0]], 0), [GameAction.ACTION1])
            agent.choose_action([], _frame(_event_frames(), 0), [GameAction.ACTION1])
            agent.choose_action([], _frame(_event_frames(), 0), [GameAction.ACTION1])

            assert agent.summary_calls == 1
            assert "summary 1: viewport shifts" in agent.actor_prompts[-2]
            assert "exact cached summary reused for the same selected keyframes" in agent.actor_prompts[-1]
            assert "Heuristic visual caption" in agent.actor_prompts[-1]
            assert "not from every saved frame and not from the game engine" in agent.actor_prompts[-1]
            assert "wmlib.animation_index()" in agent.actor_prompts[-1]
            assert "wmlib.animation_grids(...)" in agent.actor_prompts[-1]
            assert agent.actor_image_counts[-1] <= 3  # current-frame images only; contact sheet is transient
    finally:
        _restore_env(old)


def test_animation_summary_cap_blocks_new_uncached_signatures():
    old = _set_env(cap=1, evidence_keep=5)
    try:
        from tycho.agent.agent import TychoAgent

        class StubTycho(TychoAgent):
            def __init__(self):
                super().__init__()
                self.summary_calls = 0

            def _summarize_animation_with_llm(self, frames, decision, *, action: str, terminal: str) -> str:
                self.summary_calls += 1
                return f"summary {self.summary_calls}"

            def _chat(self, call_type, max_tokens=None):
                return {"text": "act", "tool_calls": [
                    {"id": "a", "name": "take_action", "input": {"action": "ACTION1"}},
                ]}

        with TemporaryDirectory() as td:
            agent = StubTycho()
            agent.reset("anim_summary", [GameAction.ACTION1], ws_root=td)
            agent.choose_action([], _frame([_event_frames()[0]], 0), [GameAction.ACTION1])
            first = agent._animation_summary_parts(_event_frames(0), action="ACTION1", terminal="nonterminal")
            second = agent._animation_summary_parts(_event_frames(1), action="ACTION2", terminal="nonterminal")

            assert first
            assert second
            assert agent.summary_calls == 1
            assert agent._turn_animation_summaries[-1]["skipped"] == "per-level animation-summary cap reached"
            assert "Animation frames are saved" in second[0]["text"]
            note = agent.ws.read_file("notes/animation_evidence.md")
            assert "ACTION1" in note
            assert "ACTION2" in note
            assert "no summary: per-level animation-summary cap reached" in note
    finally:
        _restore_env(old)


def test_animation_evidence_frames_are_saved_and_readable():
    old = _set_env(cap=2)
    try:
        from tycho.agent.agent import TychoAgent

        class StubTycho(TychoAgent):
            def _summarize_animation_with_llm(self, frames, decision, *, action: str, terminal: str) -> str:
                return "summary: broad motion."

        with TemporaryDirectory() as td:
            agent = StubTycho()
            agent.reset("anim_summary", [GameAction.ACTION1], ws_root=td)
            parts = agent._animation_summary_parts(_event_frames(), action="ACTION1", terminal="nonterminal")

            assert parts
            spec = importlib.util.spec_from_file_location("wmlib_under_test", agent.ws.dir / "wmlib.py")
            wmlib = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(wmlib)
            events = wmlib.animation_index(str(agent.ws.dir))
            assert len(events) == 1
            assert events[0]["original_frame_count"] == 6
            assert "frames" not in events[0]
            assert "all_frame_files" not in events[0]
            assert "selected_frame_files" not in events[0]
            frames = wmlib.animation_grids(events[0], root=str(agent.ws.dir))
            keyframes = wmlib.animation_grids(events[0], root=str(agent.ws.dir), keyframes=True)
            assert len(frames) == 6
            assert frames[0].shape == (64, 64)
            assert keyframes[0].shape == (64, 64)
            files = agent.ws.ls("level_0")
            assert "animation_" in files
            note = agent.ws.read_file("notes/animation_evidence.md")
            assert "frames: level_0/animation_" in note
            assert "heuristic selected-keyframe caption, not hard evidence" in note
    finally:
        _restore_env(old)


def test_game_over_animation_is_saved_with_failed_attempt():
    old = _set_env(cap=2)
    try:
        from tycho.agent.agent import TychoAgent

        class StubTycho(TychoAgent):
            def _summarize_animation_with_llm(self, frames, decision, *, action: str, terminal: str) -> str:
                return "summary: fatal transient."

        with TemporaryDirectory() as td:
            agent = StubTycho()
            agent.reset("anim_game_over", [GameAction.ACTION1], ws_root=td)
            agent.ws.record([[0]], level=0, turn_in_level=0, available=["ACTION1"])
            event = {
                "level": 0,
                "turn": 1,
                "action": "ACTION1",
                "frames": _event_frames(),
                "stem": "death_001",
            }

            agent.note_external_reset(event)
            parts = agent._game_over_evidence_parts()

            spec = importlib.util.spec_from_file_location("wmlib_game_over", agent.ws.dir / "wmlib.py")
            wmlib = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(wmlib)
            attempt = wmlib.attempts(str(agent.ws.dir))[0]
            archived = wmlib.animation_index(attempt["root"])
            assert wmlib.animation_index(str(agent.ws.dir)) == []
            assert len(archived) == 1
            assert archived[0]["terminal"] == "game_over"
            assert len(wmlib.animation_grids(archived[0], root=attempt["root"])) == 6
            assert "attempts/level_0_attempt_000/level_0/animation_001_ACTION1" in parts[-1]["text"]
    finally:
        _restore_env(old)


def test_animation_evidence_persists_when_visual_summary_unavailable():
    for vision, summary in ((True, False), (False, True)):
        old = _set_env(cap=2, vision=vision, summary=summary)
        try:
            from tycho.agent.agent import TychoAgent

            class StubTycho(TychoAgent):
                def __init__(self):
                    super().__init__()
                    self.summary_calls = 0

                def _summarize_animation_with_llm(self, frames, decision, *, action: str, terminal: str) -> str:
                    self.summary_calls += 1
                    return "should not be called"

            with TemporaryDirectory() as td:
                agent = StubTycho()
                agent.reset("anim_persist_no_summary", [GameAction.ACTION1], ws_root=td)
                parts = agent._animation_summary_parts(_event_frames(), action="ACTION1", terminal="nonterminal")

                spec = importlib.util.spec_from_file_location("wmlib_under_test", agent.ws.dir / "wmlib.py")
                wmlib = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(wmlib)
                events = wmlib.animation_index(str(agent.ws.dir))

                assert agent.summary_calls == 0
                assert len(events) == 1
                assert len(wmlib.animation_grids(events[0], root=str(agent.ws.dir))) == 6
                assert "frames saved" in parts[0]["text"]
        finally:
            _restore_env(old)


def test_animation_summary_uses_copied_history_without_actor_tools(monkeypatch):
    old = _set_env(cap=5)
    try:
        from tycho.agent import agent as agent_mod
        from tycho.harness.animation_evidence import analyze_frame_sequence

        captured: dict = {}

        def fake_chat_tools(history, tools, cfg, **kwargs):
            captured["history"] = history
            captured["tools"] = tools
            captured["system"] = kwargs.get("system", "")
            return {
                "text": "The animation reveals a viewport shift and newly exposed cells at the top.",
                "tool_calls": [],
                "usage": {"in": 10, "out": 12, "cache_read": 0, "cache_write": 0},
                "latency_ms": 1,
            }

        monkeypatch.setattr(agent_mod, "chat_tools", fake_chat_tools)

        with TemporaryDirectory() as td:
            agent = agent_mod.TychoAgent()
            agent.reset("anim_summary", [GameAction.ACTION1], ws_root=td)
            agent.cache_image_cap = 3
            agent.history = [
                {"role": "user", "content": [{"text": "old frame"}, {"image_png": b"old-a"}]},
                {"role": "user", "content": [
                    {"text": "newer frame"},
                    {"image_png": b"old-b"},
                ]},
                {"role": "assistant", "content": "noted", "raw_reasoning": {"reasoningContent": {}}},
            ]
            original_images = _image_count(agent.history)
            original_len = len(agent.history)
            decision = analyze_frame_sequence(_event_frames())

            summary = agent._summarize_animation_with_llm(
                _event_frames(),
                decision,
                action="ACTION1",
                terminal="nonterminal",
            )

            assert summary.startswith("The animation reveals")
            assert len(agent.history) == original_len
            assert _image_count(agent.history) == original_images
            assert _image_count(captured["history"]) == original_images + 1
            assert captured["history"][-1]["content"][-1].get("image_png")
            assert captured["history"][-1] not in agent.history
            assert captured["tools"] == []
            assert "transient side task" in captured["system"]
            assert captured["history"][-2].get("raw_reasoning")
    finally:
        _restore_env(old)


def test_animation_summary_falls_back_when_context_image_cap_would_break_cache(monkeypatch):
    old = _set_env(cap=5)
    try:
        from tycho.agent import agent as agent_mod
        from tycho.harness.animation_evidence import analyze_frame_sequence

        captured: dict = {}

        def fake_chat_tools(history, tools, cfg, **kwargs):
            captured["history"] = history
            captured["tools"] = tools
            captured["system"] = kwargs.get("system", "")
            return {
                "text": "The smaller contact sheet still shows a broad viewport shift.",
                "tool_calls": [],
                "usage": {"in": 5, "out": 9, "cache_read": 0, "cache_write": 0},
                "latency_ms": 1,
            }

        monkeypatch.setattr(agent_mod, "chat_tools", fake_chat_tools)

        with TemporaryDirectory() as td:
            agent = agent_mod.TychoAgent()
            agent.reset("anim_summary", [GameAction.ACTION1], ws_root=td)
            agent.cache_image_cap = 2
            agent.history = [
                {"role": "user", "content": [{"text": "old frame"}, {"image_png": b"old-a"}]},
                {"role": "user", "content": [{"text": "newer frame"}, {"image_png": b"old-b"}]},
            ]
            decision = analyze_frame_sequence(_event_frames())

            summary = agent._summarize_animation_with_llm(
                _event_frames(),
                decision,
                action="ACTION1",
                terminal="nonterminal",
            )

            assert summary.startswith("The smaller contact sheet")
            assert len(captured["history"]) == 1
            assert _image_count(captured["history"]) == 1
            assert captured["tools"] == []
            assert "transient side task" in captured["system"]
    finally:
        _restore_env(old)


def test_animation_summarizer_input_does_not_include_heuristic_labels(monkeypatch):
    from tycho.agent import agent as agent_mod
    from tycho.harness.animation_evidence import analyze_frame_sequence

    agent = agent_mod.TychoAgent.__new__(agent_mod.TychoAgent)
    decision = analyze_frame_sequence(_event_frames())
    user_text = agent._animation_summary_user_text(
        decision,
        action="ACTION1",
        terminal="nonterminal",
    )

    assert "Action: ACTION1" in user_text
    assert "Selected original frame indices:" in user_text
    assert "Filter category:" not in user_text
    assert "Filter reasons:" not in user_text
    assert decision.category not in user_text

    captured = {}

    def fake_sheet(frames, selected_indices, *, scale, title):
        captured["title"] = title
        return b"png"

    monkeypatch.setattr(agent_mod, "contact_sheet_png_bytes", fake_sheet)
    assert agent._animation_summary_png(
        _event_frames(),
        decision.selected_original_indices,
        action="ACTION1",
        terminal="nonterminal",
        scale=4,
    ) == b"png"
    assert captured["title"] == "nonterminal ACTION1"
