from __future__ import annotations

import json
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from socialclaw.v2.model import (
    ModelImage,
    ModelResult,
    ModelTool,
    ModelToolResult,
    OpenAICompatibleVisionModel,
    TextModelResult,
)
from socialclaw.v2.agents.prompts import (
    EXPLORATION_INSTRUCTIONS,
    MAIN_INSTRUCTIONS,
    UPDATE_INSTRUCTIONS,
)
from socialclaw.v2.runtime import run_arc_online


class _ActionType:
    @staticmethod
    def model_json_schema():
        return {
            "type": "object",
            "properties": {"game_id": {"type": "string"}},
        }


class _FakeAction:
    name = "ACTION1"
    action_type = _ActionType


class _FakeEnvironment:
    game_id = "unseen-fixture"

    def __init__(self) -> None:
        self.levels_completed = 0

    def _observation(self):
        grid = np.zeros((4, 4), dtype=np.int16)
        if self.levels_completed:
            grid[0, 0] = 1
        return SimpleNamespace(
            frame=[grid],
            state=SimpleNamespace(value="NOT_FINISHED"),
            levels_completed=self.levels_completed,
            available_actions=[1],
        )

    def reset(self):
        self.levels_completed = 0
        return self._observation()

    def get_available_actions(self, observation):
        return [_FakeAction()]

    def step(self, action, data=None):
        self.levels_completed = 1
        return self._observation()


class _FakeMultiLevelEnvironment(_FakeEnvironment):
    def step(self, action, data=None):
        self.levels_completed += 1
        return self._observation()


class _TwoActionsPerLevelEnvironment(_FakeEnvironment):
    def __init__(self, *, levels: int = 2) -> None:
        super().__init__()
        self.level_count = levels
        self.progress = 0

    def _observation(self):
        grid = np.full(
            (4, 4), self.levels_completed * 10 + self.progress, dtype=np.int16
        )
        state = "WIN" if self.levels_completed >= self.level_count else "NOT_FINISHED"
        return SimpleNamespace(
            frame=[grid],
            state=SimpleNamespace(value=state),
            levels_completed=self.levels_completed,
            available_actions=[1],
        )

    def reset(self):
        self.levels_completed = 0
        self.progress = 0
        return self._observation()

    def step(self, action, data=None):
        self.progress += 1
        if self.progress == 2:
            self.levels_completed += 1
            self.progress = 0
        return self._observation()


class _NeverCompletesEnvironment(_TwoActionsPerLevelEnvironment):
    def step(self, action, data=None):
        self.progress += 1
        return self._observation()


class _GameOverRecoveryEnvironment(_FakeEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.started = False
        self.failed_once = False
        self.state = "NOT_FINISHED"

    def _observation(self):
        value = 9 if self.state == "GAME_OVER" else self.levels_completed
        grid = np.full((4, 4), value, dtype=np.int16)
        return SimpleNamespace(
            frame=[grid],
            state=SimpleNamespace(value=self.state),
            levels_completed=self.levels_completed,
            available_actions=[1],
        )

    def reset(self):
        if not self.started:
            self.started = True
            self.levels_completed = 0
            self.failed_once = False
        self.state = "NOT_FINISHED"
        return self._observation()

    def step(self, action, data=None):
        if not self.failed_once:
            self.failed_once = True
            self.state = "GAME_OVER"
        else:
            self.levels_completed = 1
            self.state = "WIN"
        return self._observation()


class _FakeModel:
    model_name = "fake-structured-vision-model"

    def __init__(self) -> None:
        self.responses = [
            {
                "scene_summary": "A uniform public grid is visible.",
                "transition_analysis": None,
                "entities": [
                    {
                        "ref": "region_1",
                        "entity_id": None,
                        "label": "uniform square region",
                        "bbox": [0, 0, 3, 3],
                        "features": [
                            {
                                "name": "uniform appearance",
                                "kind": "intrinsic",
                                "value": True,
                                "confidence": 0.9,
                                "description": "All visible cells currently match.",
                            }
                        ],
                    }
                ],
                "prototypes": [
                    {
                        "name": "visual region",
                        "prototype_id": None,
                        "member_refs": ["region_1"],
                        "defining_feature_names": ["uniform appearance"],
                    }
                ],
                "schema_updates": [],
                "discarded_inferences": ["No action effect is yet evidenced."],
            },
            {
                "goal_hypotheses": [
                    {
                        "text": "A visible transformation may be required.",
                        "confidence": 0.2,
                        "evidence_ids": [],
                    }
                ],
                "decision_mode": "explore",
                "selected_action": {"name": "ACTION1", "arguments": {}},
                "schemas_used": [],
                "schema_prediction": None,
                "exploration_hypothesis": "The public action may change the visible grid.",
                "rationale": "Test the only unobserved public action.",
            },
            {
                "scene_summary": "One public transition changed a visible cell.",
                "transition_analysis": {
                    "summary": "The known uniform region changed at one cell.",
                    "entity_changes": [
                        {
                            "entity_ref": "",
                            "entity_id": None,
                            "label": "uniform square region",
                            "change_type": "feature_changed",
                            "before": "all cells matched",
                            "after": "one cell differs",
                            "description": "One cell in the region changed value.",
                            "confidence": 0.95,
                        }
                    ],
                    "unassigned_visual_changes": [],
                },
                "entities": [],
                "prototypes": [],
                "schema_updates": [
                    {
                        "operation": "create",
                        "schema_id": None,
                        "name": "ACTION1 changes part of the visible grid",
                        "role_bindings": [
                            {"role": "affected_region", "prototype": "visual region"}
                        ],
                        "preconditions": ["the observed grid state"],
                        "action": {"name": "ACTION1", "arguments": {}},
                        "expected_changes": ["at least one visible cell changes"],
                        "invariants": [],
                        "boundary_conditions": ["only one starting state is evidenced"],
                        "confidence": 0.55,
                        "reason": "The before/after images ground this action effect.",
                    }
                ],
                "discarded_inferences": [],
            },
            {
                "goal_hypotheses": [],
                "decision_mode": "explore",
                "selected_action": {"name": "ACTION1", "arguments": {}},
                "schemas_used": [],
                "schema_prediction": None,
                "exploration_hypothesis": "Test whether ACTION1 changes anything when repeated.",
                "rationale": "A repeated probe checks the boundary of the observed effect.",
            },
            {
                "scene_summary": "Repeating ACTION1 produced no public grid change.",
                "transition_analysis": {
                    "summary": "No visible Entity changed after the repeated action.",
                    "entity_changes": [],
                    "unassigned_visual_changes": [],
                },
                "entities": [],
                "prototypes": [],
                "schema_updates": [],
                "discarded_inferences": [],
            },
        ]
        self.text_responses = [
            "The effect of ACTION1 is unknown; use ACTION1 as a low-risk probe and compare which visible Entity changes.",
            "The first Evidence says one Entity changed; repeating ACTION1 can test whether that effect is conditional.",
        ]
        self.structured_payloads = []
        self.structured_instructions = []
        self.text_payloads = []
        self.tools_seen = []

    def generate(self, *, instructions, payload, images, tools=None):
        if not self.responses:
            raise AssertionError("Unexpected model call")
        self.structured_payloads.append(payload)
        self.structured_instructions.append(instructions)
        self.tools_seen.append(list(tools or []))
        return ModelResult(
            data=self.responses.pop(0),
            model=self.model_name,
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    def generate_text(self, *, instructions, payload, images, tools=None):
        if not self.text_responses:
            raise AssertionError("Unexpected text model call")
        self.text_payloads.append(payload)
        self.tools_seen.append(list(tools or []))
        return TextModelResult(
            text=self.text_responses.pop(0),
            model=self.model_name,
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )


class _LoopingFakeModel:
    model_name = "looping-fake-structured-vision-model"

    def __init__(self) -> None:
        self.structured_payloads = []
        self.text_payloads = []

    @staticmethod
    def _usage():
        return {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

    def generate(self, *, instructions, payload, images, tools=None):
        self.structured_payloads.append(payload)
        if instructions.startswith(MAIN_INSTRUCTIONS):
            data = {
                "goal_hypotheses": [],
                "decision_mode": "explore",
                "selected_action": {"name": "ACTION1", "arguments": {}},
                "schemas_used": [],
                "schema_prediction": None,
                "exploration_hypothesis": "Probe the only public action.",
                "rationale": "Use the available action within the current level budget.",
            }
        elif instructions.startswith(UPDATE_INSTRUCTIONS):
            is_action_transition = "phase=public_transition" in payload or (
                "phase=public_level_boundary" in payload
            )
            data = {
                "scene_summary": "The current public grid was recorded.",
                "transition_analysis": (
                    {
                        "summary": "The public grid changed after ACTION1.",
                        "entity_changes": [],
                        "unassigned_visual_changes": [
                            "The visible grid changed without a stored Entity attribution."
                        ],
                    }
                    if is_action_transition
                    else None
                ),
                "entities": [],
                "prototypes": [],
                "schema_updates": [],
                "discarded_inferences": [],
            }
        else:  # pragma: no cover - protects the fixture protocol
            raise AssertionError("Unexpected structured instruction profile")
        return ModelResult(
            data=data,
            model=self.model_name,
            usage=self._usage(),
        )

    def generate_text(self, *, instructions, payload, images, tools=None):
        self.text_payloads.append(payload)
        return TextModelResult(
            text="Use the only public action and observe the result.",
            model=self.model_name,
            usage=self._usage(),
        )


class _MultiLevelFakeModel(_FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.responses[2]["scene_summary"] = (
            "The first level completed and a different next-level region is now visible."
        )
        self.responses[2]["transition_analysis"] = {
            "summary": (
                "The public level counter increased; the old scene ended and the attached "
                "after image is the next level rather than an ordinary action-effect frame."
            ),
            "entity_changes": [
                {
                    "entity_ref": "",
                    "entity_id": None,
                    "label": "uniform square region",
                    "change_type": "disappeared",
                    "before": "visible in the completed level",
                    "after": "not part of the new level scene",
                    "description": "The prior-level region ended at the public boundary.",
                    "confidence": 0.95,
                },
                {
                    "entity_ref": "level_two_region",
                    "entity_id": None,
                    "label": "next level patterned region",
                    "change_type": "appeared",
                    "before": "absent from the completed level",
                    "after": "visible in the new level scene",
                    "description": "A new current scene is visible after level completion.",
                    "confidence": 0.95,
                },
            ],
            "unassigned_visual_changes": [],
        }
        self.responses[2]["entities"] = [
            {
                "ref": "level_two_region",
                "entity_id": None,
                "label": "next level patterned region",
                "bbox": [0, 0, 3, 3],
                "status": "active",
                "features": [
                    {
                        "name": "patterned appearance",
                        "kind": "state",
                        "value": True,
                        "confidence": 0.9,
                        "description": "The next level has a visibly different pattern.",
                    }
                ],
            }
        ]
        self.responses[2]["prototypes"] = [
            {
                "name": "visual region",
                "prototype_id": None,
                "member_refs": ["level_two_region"],
                "defining_feature_names": ["patterned appearance"],
            }
        ]


class _FailingSecondMainModel(_FakeModel):
    def generate(self, *, instructions, payload, images, tools=None):
        if len(self.structured_payloads) >= 3:
            raise RuntimeError("simulated provider failure")
        return super().generate(
            instructions=instructions, payload=payload, images=images, tools=tools
        )


class _InvalidThenCorrectedUpdateModel(_FakeModel):
    def __init__(self) -> None:
        super().__init__()
        corrected = self.responses[2]
        self.responses[2] = {
            "scene_summary": "The grid changed, but no difference was attributed.",
            "transition_analysis": {
                "summary": "Incorrect empty attribution.",
                "entity_changes": [],
                "unassigned_visual_changes": [],
            },
            "entities": [],
            "prototypes": [],
            "schema_updates": [],
            "discarded_inferences": [],
        }
        self.responses.insert(3, corrected)


class GenericARCOnlineTests(unittest.TestCase):
    def test_openai_compatible_model_executes_read_only_tool_loop(self) -> None:
        responses = [
            {
                "model": "test-model",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "test_read_tool",
                                        "arguments": '{"query":"schema"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
            {
                "model": "test-model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"ok":true}'},
                    }
                ],
                "usage": {"prompt_tokens": 15, "completion_tokens": 3, "total_tokens": 18},
            },
        ]
        requests = []

        class _Response:
            def __init__(self, value):
                self.value = value

            def raise_for_status(self):
                return None

            def json(self):
                return self.value

        class _Client:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, url, headers, json):
                requests.append(copy.deepcopy(json))
                return _Response(responses.pop(0))

        model = OpenAICompatibleVisionModel(
            base_url="https://unused.invalid",
            api_key="test-key",
            model="test-model",
        )
        tool = ModelTool(
            name="test_read_tool",
            description="read memory",
            parameters={"type": "object", "properties": {}},
            execute=lambda arguments: ModelToolResult(
                text="schema result",
                images=[
                    ModelImage(
                        label="stored observation",
                        artifact_id="artifact_test",
                        sha256="abc123",
                        relative_path="images/test.png",
                        data_url="data:image/png;base64,AA==",
                    )
                ],
            ),
        )
        with patch("socialclaw.v2.model.httpx.Client", _Client):
            result = model.generate(
                instructions="Return JSON.",
                payload="compact prose",
                images=[],
                tools=[tool],
            )

        self.assertEqual(result.data, {"ok": True})
        self.assertEqual(result.usage["input_tokens"], 25)
        self.assertEqual(result.usage["total_tokens"], 30)
        self.assertEqual(len(result.tool_trace), 1)
        self.assertEqual(result.tool_trace[0]["result"], "schema result")
        self.assertEqual(
            result.tool_trace[0]["returned_images"][0]["artifact_id"],
            "artifact_test",
        )
        self.assertEqual(len(result.usage_rounds), 2)
        self.assertEqual(requests[0]["messages"][1]["content"][0]["text"], "compact prose")
        self.assertEqual(requests[1]["messages"][-2]["role"], "tool")
        self.assertEqual(requests[1]["messages"][-1]["role"], "user")
        self.assertEqual(
            requests[1]["messages"][-1]["content"][-1]["image_url"]["url"],
            "data:image/png;base64,AA==",
        )

    def test_structured_model_retries_invalid_json_once(self) -> None:
        class _RepairingModel(OpenAICompatibleVisionModel):
            def __init__(self) -> None:
                super().__init__(
                    base_url="https://unused.invalid",
                    api_key="test-key",
                    model="test-model",
                )
                self.instructions_seen = []

            def _request(self, *, instructions, payload, images, json_mode, tools):
                self.instructions_seen.append(instructions)
                if len(self.instructions_seen) == 1:
                    usage = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
                    return '{"scene_summary": "unterminated', "test-model", usage, [], [usage]
                usage = {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7}
                return '{"scene_summary": "repaired"}', "test-model", usage, [], [usage]

        model = _RepairingModel()
        result = model.generate(instructions="Return JSON.", payload={}, images=[])

        self.assertEqual(result.data, {"scene_summary": "repaired"})
        self.assertEqual(result.usage["total_tokens"], 12)
        self.assertEqual(len(model.instructions_seen), 2)
        self.assertIn("previous response could not be parsed", model.instructions_seen[1])

    def test_structured_model_has_a_final_tool_free_transport_repair(self) -> None:
        class _TwiceInvalidModel(OpenAICompatibleVisionModel):
            def __init__(self) -> None:
                super().__init__(
                    base_url="https://unused.invalid",
                    api_key="test-key",
                    model="test-model",
                )
                self.tools_seen = []

            def _request(self, *, instructions, payload, images, json_mode, tools):
                self.tools_seen.append(list(tools))
                usage = {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
                if len(self.tools_seen) < 3:
                    return "", "test-model", usage, [], [usage]
                return '{"ok":true}', "test-model", usage, [], [usage]

        model = _TwiceInvalidModel()
        marker_tool = ModelTool(
            name="marker",
            description="unused",
            parameters={"type": "object", "properties": {}},
            execute=lambda arguments: "unused",
        )
        result = model.generate(
            instructions="Return JSON.", payload="payload", images=[], tools=[marker_tool]
        )

        self.assertEqual(result.data, {"ok": True})
        self.assertEqual(result.usage["total_tokens"], 9)
        self.assertEqual([len(items) for items in model.tools_seen], [1, 1, 0])

    def test_cognition_tool_budget_closes_tools_and_forces_final_answer(self) -> None:
        def tool_message(call_id):
            return {
                "model": "test-model",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": "test_read_tool",
                                        "arguments": '{"query":"more"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "total_tokens": 6,
                },
            }

        responses = [
            tool_message("call_1"),
            tool_message("call_2"),
            {
                "model": "test-model",
                "choices": [
                    {"finish_reason": "stop", "message": {"content": '{"ok":true}'}}
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "total_tokens": 10,
                },
            },
        ]
        requests = []

        class _Response:
            def __init__(self, value):
                self.value = value

            def raise_for_status(self):
                return None

            def json(self):
                return self.value

        class _Client:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, url, headers, json):
                requests.append(copy.deepcopy(json))
                return _Response(responses.pop(0))

        model = OpenAICompatibleVisionModel(
            base_url="https://unused.invalid", api_key="test-key", model="test-model"
        )
        tool = ModelTool(
            name="test_read_tool",
            description="read memory",
            parameters={"type": "object", "properties": {}},
            execute=lambda arguments: "memory result",
        )
        with patch("socialclaw.v2.model.httpx.Client", _Client):
            result = model.generate(
                instructions="Return JSON.", payload="compact prose", images=[], tools=[tool]
            )

        self.assertEqual(result.data, {"ok": True})
        self.assertEqual(len(result.tool_trace), 2)
        self.assertEqual(len(result.usage_rounds), 3)
        self.assertIn("tools", requests[0])
        self.assertIn("tools", requests[1])
        self.assertNotIn("tools", requests[2])
        self.assertIn("budget is now exhausted", requests[2]["messages"][-1]["content"])

    def test_update_agent_retries_evidence_constraint_violation_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = _InvalidThenCorrectedUpdateModel()
            summary = run_arc_online(
                Path(directory) / "run",
                game_id="unseen-fixture",
                model=model,
                max_steps=1,
                env=_FakeEnvironment(),
                replay_fn=lambda root, episode: {
                    "episode_id": episode.episode_id,
                    "status": "passed",
                    "steps_replayed": len(episode.steps),
                    "errors": [],
                },
            )

            self.assertTrue(summary["success"])
            self.assertEqual(len(model.structured_payloads), 4)
            correction = model.structured_instructions[3]
            self.assertIn("No new observation or external fact", correction)
            self.assertNotIn("authoritative", correction)

    def test_zero_prior_agent_inputs_and_compact_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            model = _FakeModel()
            summary = run_arc_online(
                output,
                game_id="unseen-fixture",
                model=model,
                max_steps=2,
                env=_FakeEnvironment(),
                replay_fn=lambda root, episode: {
                    "episode_id": episode.episode_id,
                    "status": "passed",
                    "steps_replayed": len(episode.steps),
                    "errors": [],
                },
            )
            self.assertTrue(summary["success"])
            self.assertEqual(summary["actions"], 1)
            self.assertEqual(summary["model_calls"], 4)
            self.assertEqual(summary["final_cognition"]["schemas"], 1)

            timeline = json.loads((output / "timeline.json").read_text())
            first_action = timeline["events"][1]
            received = first_action["shared_decision_input"]
            self.assertNotIn("game_id", json.dumps(received))
            self.assertNotIn("goal_description", json.dumps(received))
            self.assertIn("current_observation_ref", received)
            self.assertIn("cognition_ref", received)
            self.assertIn("input_catalog", timeline)
            cognition_ref = received["cognition_ref"].split(".")[-1]
            self.assertIn(
                cognition_ref, timeline["input_catalog"]["cognition_views"]
            )
            cognition_receipt = timeline["input_catalog"]["cognition_views"][
                cognition_ref
            ]
            self.assertIn("entities_sent", cognition_receipt)
            self.assertIn("evidence_sent", cognition_receipt)
            self.assertEqual(first_action["decision"]["schema_ids"], [])
            self.assertIsNone(first_action["decision"]["schema_prediction"])
            self.assertTrue(first_action["decision"]["exploration_hypothesis"])
            self.assertEqual(
                first_action["agent_calls"]["main_agent"]["received_refs"],
                [
                    "event.shared_decision_input",
                    "event.agent_calls.exploration_agent.output",
                ],
            )
            self.assertTrue(
                first_action["agent_calls"]["main_agent"]["image_inputs"]
            )
            self.assertIsInstance(
                first_action["agent_calls"]["exploration_agent"]["output"], str
            )
            self.assertIn("evidence", timeline["input_catalog"])
            transition = first_action["environment_transition"]
            self.assertEqual(
                transition["semantic_result"]["entity_changes"][0]["change_type"],
                "feature_changed",
            )
            evidence_ref = transition["evidence_ref"].split(".")[-1]
            resolved = timeline["input_catalog"]["evidence"][evidence_ref]
            self.assertEqual(resolved["evidence_id"], transition["evidence_id"])
            self.assertEqual(
                resolved["entity_changes"][0]["label"], "uniform square region"
            )
            self.assertTrue(resolved["artifacts"])

            observation_ref = received["current_observation_ref"].split(".")[-1]
            observation = timeline["input_catalog"]["observations"][
                observation_ref
            ]
            agent_view = next(
                item for item in observation["artifacts"] if item["role"] == "agent_view"
            )
            review_view = next(
                item for item in observation["artifacts"] if item["role"] == "review_view"
            )
            self.assertEqual(agent_view["metadata"]["width"], 32)
            self.assertFalse(agent_view["metadata"]["grid_overlay"])
            self.assertEqual(review_view["metadata"]["width"], 32)
            self.assertTrue(review_view["metadata"]["grid_overlay"])
            self.assertEqual(
                first_action["agent_calls"]["main_agent"]["image_inputs"][0][
                    "artifact_id"
                ],
                agent_view["artifact_id"],
            )
            self.assertNotIn("logical_grid_sha256", observation["public_state"])

            contract_ref = received["available_actions_ref"].split(".")[-1]
            contracts = timeline["input_catalog"]["action_contracts"][contract_ref]
            self.assertEqual(contracts[0]["name"], "ACTION1")

            self.assertTrue((output / "report.md").is_file())
            self.assertTrue((output / "token_usage.md").is_file())
            self.assertTrue((output / "token_usage.json").is_file())
            process = (output / "process.md").read_text()
            self.assertIn("Recent public transitions", process)
            self.assertNotIn("先解释：什么是", process)
            self.assertIn("Step 0", process)
            self.assertIn("Step 1", process)
            self.assertIn("动作前公开画面", process)
            self.assertIn("Exploration Agent 输出", process)
            self.assertIn("Update Agent 输出", process)
            self.assertIn("本次实际 System Instructions", process)
            self.assertIn("The runtime maintains a provisional cognition graph", process)
            self.assertIn("read_cognition tool is an exact store reader", process)
            self.assertTrue((output / "cognition" / "graph.json").is_file())
            self.assertFalse((output / "summary.json").exists())
            self.assertFalse((output / "evidence.json").exists())
            self.assertFalse((output / "manifest.json").exists())
            self.assertFalse((output / "cognition" / "snapshots").exists())
            self.assertFalse((output / "timeline.partial.json").exists())
            self.assertFalse((output / "process.partial.md").exists())
            self.assertFalse((output / "cognition" / "graph.partial.json").exists())
            self.assertTrue(all(model.tools_seen))
            prompt = model.text_payloads[0]
            self.assertIsInstance(prompt, str)
            self.assertIn("uniform square region", prompt)
            self.assertIn("important features", prompt)
            self.assertIn("All visible cells currently match.", prompt)
            self.assertNotIn("artifact_", prompt)
            self.assertNotIn("logical_grid_sha256", prompt)
            graph_payload = json.loads(
                (output / "cognition" / "graph.json").read_text()
            )
            entity_id = next(iter(graph_payload["entities"]))
            tool = model.tools_seen[0][0]
            self.assertEqual(tool.name, "read_cognition")
            tool_result = tool.execute(
                {"command": "get_entity", "id": entity_id}
            )
            self.assertIn("uniform square region", tool_result)
            unrelated = tool.execute(
                {"command": "get_entity", "id": "entity_missing"}
            )
            self.assertEqual(json.loads(unrelated)["error"], "not_found")
            transition_evidence = next(
                value
                for value in graph_payload["evidence"].values()
                if value["kind"] == "public_transition"
            )
            evidence_result = json.loads(
                tool.execute(
                    {
                        "command": "get_evidence",
                        "id": transition_evidence["evidence_id"],
                    }
                )
            )
            self.assertEqual(
                [item["phase"] for item in evidence_result["record"]["observation_refs"]],
                ["before", "after"],
            )
            artifact_id = evidence_result["record"]["artifact_ids"][0]
            artifact_result = tool.execute(
                {"command": "get_artifact", "id": artifact_id}
            )
            self.assertIsInstance(artifact_result, ModelToolResult)
            self.assertEqual(len(artifact_result.images), 1)
            self.assertEqual(artifact_result.images[0].artifact_id, artifact_id)
            usage_report = json.loads((output / "token_usage.json").read_text())
            self.assertEqual(usage_report["totals"]["logical_calls"], 4)
            self.assertEqual(
                usage_report["by_request_phase"][
                    "first_request_per_logical_call"
                ]["provider_requests"],
                4,
            )
            self.assertIn(
                "Current learned cognition", usage_report["by_input_section"]
            )
            assertion = next(iter(graph_payload["feature_assertions"].values()))
            self.assertEqual(
                assertion["description"], "All visible cells currently match."
            )

    def test_next_decision_receives_semantic_transition_and_resolved_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = _FakeModel()
            summary = run_arc_online(
                Path(directory) / "run",
                game_id="unseen-fixture",
                model=model,
                max_steps=1,
                stop_after_levels=2,
                env=_FakeEnvironment(),
                replay_fn=lambda root, episode: {
                    "episode_id": episode.episode_id,
                    "status": "passed",
                    "steps_replayed": len(episode.steps),
                    "errors": [],
                },
            )
            self.assertEqual(summary["actions"], 2)
            recent_text = model.text_payloads[1]
            self.assertIn("## Recent public transitions", recent_text)
            self.assertIn("known uniform region changed", recent_text)
            self.assertIn("Evidence=evidence_", recent_text)
            self.assertNotIn("artifact_", recent_text)
            run = Path(directory) / "run"
            timeline = json.loads((run / "timeline.json").read_text())
            second = timeline["events"][2]
            attached = second["shared_decision_input"][
                "attached_schema_evidence_images"
            ]
            self.assertEqual(attached, [])
            self.assertEqual(
                len(second["agent_calls"]["main_agent"]["image_inputs"]), 1
            )

    def test_level_boundary_continues_into_next_level_with_clean_entity_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            model = _MultiLevelFakeModel()
            summary = run_arc_online(
                output,
                game_id="unseen-fixture",
                model=model,
                max_steps=2,
                stop_after_levels=2,
                env=_FakeMultiLevelEnvironment(),
                replay_fn=lambda root, episode: {
                    "episode_id": episode.episode_id,
                    "status": "passed",
                    "steps_replayed": len(episode.steps),
                    "errors": [],
                },
            )

            self.assertTrue(summary["success"])
            self.assertEqual(summary["actions"], 2)
            self.assertEqual(summary["public_levels_completed"], 2)
            second_main_input = model.structured_payloads[3]
            self.assertIn("levels_completed=1", second_main_input)
            cognition_section = second_main_input.split(
                "## Current learned cognition\n", 1
            )[1].split("\n\n## On-demand memory", 1)[0]
            self.assertIn("next level patterned region", cognition_section)
            self.assertNotIn("uniform square region", cognition_section)

            timeline = json.loads((output / "timeline.json").read_text())
            self.assertEqual(
                timeline["events"][1]["update_input"]["phase"],
                "public_level_boundary",
            )
            self.assertEqual(
                timeline["events"][2]["update_input"]["phase"],
                "public_level_boundary",
            )
            first_boundary_kinds = timeline["events"][1]["cognitive_update"][
                "transaction"
            ]["operation_kinds"]
            self.assertIn("update_entity", first_boundary_kinds)
            self.assertIn("add_entity", first_boundary_kinds)

    def test_max_step_is_a_fresh_budget_for_each_level(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = run_arc_online(
                Path(directory) / "run",
                game_id="unseen-fixture",
                model=_LoopingFakeModel(),
                max_steps=2,
                stop_after_levels=None,
                env=_TwoActionsPerLevelEnvironment(levels=2),
                replay_fn=lambda root, episode: {
                    "episode_id": episode.episode_id,
                    "status": "passed",
                    "steps_replayed": len(episode.steps),
                    "errors": [],
                },
            )

            self.assertTrue(summary["success"])
            self.assertTrue(summary["game_won"])
            self.assertEqual(summary["actions"], 4)
            self.assertEqual(summary["levels_passed"], 2)
            self.assertEqual(summary["levels_attempted"], 2)
            self.assertEqual(
                [item["actions"] for item in summary["level_results"]], [2, 2]
            )

    def test_level_fails_when_its_own_action_budget_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = run_arc_online(
                Path(directory) / "run",
                game_id="unseen-fixture",
                model=_LoopingFakeModel(),
                max_steps=2,
                stop_after_levels=None,
                env=_NeverCompletesEnvironment(levels=1),
                replay_fn=lambda root, episode: {
                    "episode_id": episode.episode_id,
                    "status": "passed",
                    "steps_replayed": len(episode.steps),
                    "errors": [],
                },
            )

            self.assertFalse(summary["success"])
            self.assertEqual(summary["termination_reason"], "level_step_limit")
            self.assertEqual(summary["actions"], 2)
            self.assertEqual(summary["levels_passed"], 0)
            self.assertEqual(summary["levels_attempted"], 1)
            self.assertIn("did not complete within 2", summary["failure_reason"])

    def test_default_game_over_reset_preserves_same_level_budget_and_compact_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            summary = run_arc_online(
                output,
                game_id="unseen-fixture",
                model=_LoopingFakeModel(),
                max_steps=2,
                stop_after_levels=None,
                compact_process=True,
                env=_GameOverRecoveryEnvironment(),
                replay_fn=lambda root, episode: {
                    "episode_id": episode.episode_id,
                    "status": "passed",
                    "steps_replayed": len(episode.steps),
                    "errors": [],
                },
            )

            self.assertTrue(summary["game_won"])
            self.assertEqual(summary["actions"], 2)
            self.assertEqual(summary["trajectory_steps"], 3)
            self.assertEqual(summary["runtime_resets"], 1)
            self.assertEqual(summary["level_results"][0]["actions"], 2)
            timeline = json.loads((output / "timeline.json").read_text())
            reset = timeline["events"][1]["environment_reset"]
            self.assertEqual(
                reset["update_input"]["level_budget"]["actions_used"], 1
            )
            self.assertIn(
                "recovery_update_agent", timeline["events"][1]["agent_calls"]
            )
            process = (output / "process.md").read_text()
            self.assertNotIn("先解释：什么是", process)
            self.assertIn("Prompt 记录策略", process)
            self.assertNotIn("本次实际 System Instructions", process)
            self.assertNotIn("### Main Agent 实际收到\n\n- 实际文本输入", process)
            self.assertIn("Entity 输入", process)
            self.assertIn("Prototype 输入", process)
            self.assertIn("Schema 输入", process)
            self.assertIn("GAME_OVER 后恢复当前 Level", process)
            self.assertIn("actions_used': 1", process)

    def test_provider_failure_preserves_last_complete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with self.assertRaisesRegex(RuntimeError, "simulated provider failure"):
                run_arc_online(
                    output,
                    game_id="unseen-fixture",
                    model=_FailingSecondMainModel(),
                    max_steps=2,
                    stop_after_levels=2,
                    env=_FakeEnvironment(),
                    replay_fn=lambda root, episode: {},
                )
            partial = json.loads((output / "timeline.partial.json").read_text())
            self.assertEqual(partial["summary"]["checkpoint_status"], "INCOMPLETE")
            self.assertEqual(partial["summary"]["actions"], 1)
            self.assertEqual([item["step"] for item in partial["events"]], [0, 1])
            self.assertTrue((output / "process.partial.md").is_file())
            self.assertTrue((output / "cognition" / "graph.partial.json").is_file())

    def test_v2_has_no_game_specific_or_privileged_imports(self) -> None:
        root = Path(__file__).resolve().parents[2] / "socialclaw" / "v2"
        text = "\n".join(path.read_text() for path in root.rglob("*.py"))
        forbidden = [
            "trajectory.arc_policies",
            "schema.gold_loader",
            "arc_environment_fingerprint",
            "arc_environment_files",
            "third_party/arc_agi3_games",
        ]
        for value in forbidden:
            self.assertNotIn(value, text)
        self.assertNotRegex(text, r"(?i)cd82|sk48|tu93")

    def test_exploration_is_prose_and_requested_prompt_lines_are_absent(self) -> None:
        self.assertIn("Do not return JSON", EXPLORATION_INSTRUCTIONS)
        self.assertIn("Exploration is only", EXPLORATION_INSTRUCTIONS)
        self.assertIn("primary objective is to complete", MAIN_INSTRUCTIONS)
        self.assertIn("A predictable action effect is not", MAIN_INSTRUCTIONS)
        self.assertNotIn("You have no game", EXPLORATION_INSTRUCTIONS)
        self.assertNotIn("No goal or action meaning is supplied", MAIN_INSTRUCTIONS)


if __name__ == "__main__":
    unittest.main()
