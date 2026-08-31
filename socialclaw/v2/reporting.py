from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List


def _short(value: Any, limit: int = 360) -> str:
    """Keep the audit faithful without reproducing bulky JSON-shaped prose."""
    if value is None:
        return "无"
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _resolve(payload: Dict[str, Any], reference: str) -> Any:
    parts = reference.split(".")
    if len(parts) != 3 or parts[0] != "input_catalog":
        raise ValueError(f"Unsupported timeline reference: {reference}")
    return payload["input_catalog"][parts[1]][parts[2]]


def _action(value: Dict[str, Any] | None) -> str:
    if not value:
        return "无"
    name = str(value.get("name") or "unknown")
    arguments = value.get("arguments") or {}
    return name if not arguments else f"{name}({json.dumps(arguments, ensure_ascii=False, sort_keys=True)})"


def _image_link(observation: Dict[str, Any], label: str) -> str:
    image = next(
        (
            item
            for item in observation.get("artifacts") or []
            if item.get("media_type") == "image/png"
            and item.get("role") == "review_view"
        ),
        None,
    )
    if image is None:  # Legacy runs stored only one guided PNG.
        image = next(
            (
                item
                for item in observation.get("artifacts") or []
                if item.get("media_type") == "image/png"
            ),
            None,
        )
    if image is None:
        return f"{label}：无 PNG"
    path = f"trajectory/assets/{image['relative_path']}"
    return f"[{label} · {image['artifact_id']}]({path})"


def _agent_image_link(observation: Dict[str, Any], label: str) -> str:
    image = next(
        (
            item
            for item in observation.get("artifacts") or []
            if item.get("media_type") == "image/png"
            and item.get("role") == "agent_view"
        ),
        None,
    )
    if image is None:
        return f"{label}：无 Agent PNG"
    path = f"trajectory/assets/{image['relative_path']}"
    overlay = (image.get("metadata") or {}).get("grid_overlay")
    note = "原始无辅助线" if overlay is False else "旧版图片"
    return f"[{label} · {note} · {image['artifact_id']}]({path})"


def _compact_result(result: Dict[str, Any]) -> str:
    return (
        f"changed={result.get('grid_changed')}, "
        f"cells={result.get('changed_cells')}, "
        f"bounds={result.get('changed_bounds')}, "
        f"level_delta={result.get('level_delta')}, "
        f"status={result.get('environment_status')}"
    )


def _recent_history(
    values: Iterable[Dict[str, Any]], timeline: Dict[str, Any] | None = None
) -> str:
    items = []
    for value in values:
        semantic = value.get("semantic_result") or {}
        changes = semantic.get("entity_changes") or []
        entity_text = ", ".join(
            f"{item.get('entity_id')}:{item.get('label')} {item.get('change_type')} "
            f"({_short(item.get('description') or (str(item.get('before')) + ' -> ' + str(item.get('after'))), 120)})"
            for item in changes
        )
        unassigned = semantic.get("unassigned_visual_changes") or []
        semantic_text = (
            entity_text
            or (
                "未归属变化="
                + ", ".join(_short(item, 100) for item in unassigned)
                if unassigned
                else ""
            )
            or "旧版记录未提供 Entity 语义变化"
        )
        evidence = value.get("evidence") or {}
        if not evidence and timeline is not None and value.get("evidence_ref"):
            evidence = _resolve(timeline, value["evidence_ref"])
        items.append(
            f"S{value.get('step')} {_action(value.get('action'))} → "
            f"{_compact_result(value.get('public_result') or {})}; "
            f"Entity语义=[{semantic_text}]; "
            f"evidence={value.get('evidence_id')}"
            + (
                f"(kind={evidence.get('kind')}, artifacts={evidence.get('artifact_ids')})"
                if evidence
                else ""
            )
        )
    return "；".join(items) if items else "空（此前没有公开交互）"


def _schema_surface(cognition: Dict[str, Any]) -> str:
    values = cognition.get("schemas_sent") or []
    if not values:
        return "空（Schema=0）"
    return "；".join(
        f"{item.get('schema_id')}:"
        f"{item.get('prototype_id')} → {_short(item.get('action'), 80)} → {_short(item.get('output'), 100)}"
        for item in values
    )


def _cognition_lines(cognition: Dict[str, Any]) -> List[str]:
    entities = cognition.get("entities_sent")
    prototypes = cognition.get("prototypes_sent")
    evidence = cognition.get("evidence_sent")
    if entities is None:  # Legacy compact receipt.
        return [
            f"- Entity IDs：`{cognition.get('entity_ids') or []}`（旧版只保存输入哈希与 ID）",
            f"- Prototype IDs：`{cognition.get('prototype_ids') or []}`",
        ]
    entity_text = [
        {
            "entity_id": item.get("entity_id"),
            "label": item.get("label"),
            "bbox": item.get("bbox"),
            "status": item.get("status"),
            "features": [
                {
                    "name": feature.get("feature_name"),
                    "value": feature.get("value"),
                    "confidence": feature.get("confidence"),
                }
                for feature in item.get("features") or []
            ],
            "evidence_ids": item.get("evidence_ids"),
        }
        for item in entities
    ]
    return [
        f"- Entity 输入（{len(entity_text)}）：`{entity_text}`",
        f"- Prototype 输入（{len(prototypes or [])}）：`{prototypes or []}`",
        f"- Schema 输入（{len(cognition.get('schemas_sent') or [])}）：`{cognition.get('schemas_sent') or []}`",
        f"- 全局 Insight/Rule 输入（{len(cognition.get('insights_sent') or [])}）：`{cognition.get('insights_sent') or []}`",
        f"- Evidence 摘要输入（{len(evidence or [])}）：`{evidence or []}`",
    ]


def _evidence_text(evidence: Any) -> str:
    if not isinstance(evidence, dict):
        return str(evidence or "无")
    return str(
        {
            "evidence_id": evidence.get("evidence_id"),
            "kind": evidence.get("kind"),
            "step_index": evidence.get("step_index"),
            "action": evidence.get("action"),
            "result": evidence.get("result"),
            "semantic_summary": evidence.get("semantic_summary"),
            "entity_changes": evidence.get("entity_changes"),
            "unassigned_visual_changes": evidence.get(
                "unassigned_visual_changes"
            ),
            "observation_fingerprints": evidence.get(
                "observation_fingerprints"
            ),
            "artifacts": evidence.get("artifacts")
            or evidence.get("artifact_ids"),
        }
    )


def _goal_lines(values: Iterable[Dict[str, Any]]) -> List[str]:
    result = []
    for item in values:
        result.append(
            f"  - `{float(item.get('confidence') or 0):.2f}` {_short(item.get('text'), 160)} "
            f"（evidence={item.get('evidence_ids') or []}）"
        )
    return result or ["  - 无"]


def _exploration_lines(output: Any) -> List[str]:
    if isinstance(output, str):
        return [f"- 给 Main 的文本建议：{output}"]
    if not isinstance(output, dict):
        return ["- 无有效文本建议"]
    lines = ["- 不确定项："]
    uncertainties = output.get("uncertainties") or []
    lines.extend([f"  - {item}" for item in uncertainties] or ["  - 无"])
    lines.append("- 候选 probe：")
    proposals = output.get("proposals") or []
    if not proposals:
        lines.append("  - 无")
    for item in proposals:
        priority = (
            float(item.get("expected_information_gain") or 0)
            - float(item.get("irreversible_risk") or 0)
            - float(item.get("action_cost") or 0)
            - float(item.get("repeated_probe_penalty") or 0)
        )
        lines.append(
            f"  - `{_action(item.get('action'))}`，priority≈{priority:.2f}："
            f"{_short(item.get('hypothesis'), 140)}；预期："
            f"{_short(item.get('expected_observation'), 100)}"
        )
    return lines


def _update_lines(output: Dict[str, Any]) -> List[str]:
    lines = [f"- scene summary：{_short(output.get('scene_summary'), 300)}"]
    entities = output.get("entities") or []
    entity_names = [
        f"{'更新已有' if entity.get('entity_id') else '新候选'} "
        f"{entity.get('entity_id') or entity.get('ref')}:{_short(entity.get('label'), 60)}"
        for entity in entities
    ]
    lines.append(
        f"- Entity observation/upsert：{len(entities)} 个"
        + (f"（{'；'.join(entity_names)}）" if entity_names else "")
    )
    prototypes = output.get("prototypes") or []
    prototype_names = [
        {
            "mode": "update_existing" if item.get("prototype_id") else "new_candidate",
            "prototype_id": item.get("prototype_id"),
            "name": item.get("name"),
            "member_refs": item.get("member_refs") or [],
            "defining_feature_names": item.get("defining_feature_names") or [],
        }
        for item in prototypes
    ]
    lines.append(
        f"- Prototype proposal：{len(prototypes)} 个"
        + (f"（{prototype_names}）" if prototype_names else "")
    )
    schemas = output.get("schema_updates") or []
    schema_names = [
        f"{item.get('operation')}:{item.get('schema_id') or _short(item.get('prototype'), 60)}"
        for item in schemas
    ]
    lines.append(
        f"- Schema proposal：{len(schemas)} 个"
        + (f"（{'；'.join(schema_names)}）" if schema_names else "")
    )
    insights = output.get("insight_updates") or []
    insight_names = [
        f"{item.get('operation')}:{item.get('insight_id') or _short(item.get('statement'), 100)}"
        for item in insights
    ]
    lines.append(
        f"- Insight/Rule proposal：{len(insights)} 个"
        + (f"（{'；'.join(insight_names)}）" if insight_names else "")
    )
    discarded = output.get("discarded_inferences") or []
    lines.append(
        "- 主动放弃的推断："
        + ("；".join(_short(item, 180) for item in discarded) if discarded else "无")
    )
    transition = output.get("transition_analysis")
    if isinstance(transition, dict):
        lines.append(f"- Entity 级 transition：{_short(transition.get('summary'), 300)}")
        changes = transition.get("entity_changes") or []
        if changes:
            for item in changes:
                lines.append(
                    f"  - `{item.get('entity_id') or item.get('entity_ref')}` "
                    f"{item.get('label')} / {item.get('change_type')}："
                    f"{_short(item.get('description') or (str(item.get('before')) + ' → ' + str(item.get('after'))), 220)}"
                )
        else:
            lines.append("  - 没有观察到 Entity 级变化")
        unassigned = transition.get("unassigned_visual_changes") or []
        if unassigned:
            lines.append(
                "  - 尚未归属到 Entity 的视觉变化："
                + "；".join(_short(item, 180) for item in unassigned)
            )
    return lines


def _transaction_line(update: Dict[str, Any]) -> str:
    transaction = update["transaction"]
    return (
        f"`{transaction.get('transaction_id')}`：revision={transaction.get('revision')}，"
        f"mode={transaction.get('mode')}，applied={transaction.get('applied_operations')}，"
        f"skipped={transaction.get('skipped_operations')}"
    )


def _contract_line(item: Dict[str, Any]) -> str:
    schema = item.get("arguments_schema") or {}
    properties = schema.get("properties") or {}
    if not properties:
        return f"`{item.get('name')}`（无参数）"
    arguments = []
    for name, definition in properties.items():
        bounds = ""
        if "minimum" in definition or "maximum" in definition:
            bounds = f"[{definition.get('minimum')},{definition.get('maximum')}]"
        description = str(definition.get("description") or "").strip()
        arguments.append(
            f"{name}:{definition.get('type', 'unknown')}{bounds}"
            + (f"—{description}" if description else "")
        )
    return f"`{item.get('name')}`（{', '.join(arguments)}）"


def _actual_call_input_lines(
    call: Dict[str, Any], *, include_text: bool = True
) -> List[str]:
    text = str(call.get("input_text") or "").strip()
    usage = call.get("usage") or {}
    sections = call.get("input_sections") or []
    section_text = "；".join(
        f"{item.get('section')}={item.get('characters')} chars"
        for item in sections
    ) or "无"
    traces = call.get("tool_trace") or []
    lines = [
        f"- provider usage：input={usage.get('input_tokens', 0)}，"
        f"output={usage.get('output_tokens', 0)}，total={usage.get('total_tokens', 0)}",
        f"- 输入区段体积：{section_text}",
        f"- `read_cognition` 调用数：{len(traces)}",
    ]
    for item in traces:
        lines.append(
            f"  - command：`{item.get('arguments')}` → "
            f"{_short(item.get('result'), 500)}"
        )
        returned_images = item.get("returned_images") or []
        if returned_images:
            lines.append(f"    - 返回并附加的保存图片：`{returned_images}`")
    if include_text and text:
        lines.extend(["- 实际文本输入：", "", "```text", text, "```"])
    elif text:
        lines.append(
            "- 完整 prompt：按本次精简报告设置省略；EFPS 与公开输入在本节的结构化摘要中列出，"
            "机器审计原文仍保存在 `timeline.json`。"
        )
    return lines


def build_process_markdown(timeline: Dict[str, Any]) -> str:
    """Render one chronological, human-readable audit from compact timeline JSON."""
    summary = timeline["summary"]
    include_prompt_text = timeline.get("process_prompt_detail", "full") == "full"
    if summary.get("checkpoint_status") == "INCOMPLETE":
        result_text = "未完成 checkpoint，不可作为最终实验结果"
    else:
        result_text = str(summary.get("termination_reason") or (
            "完成请求的 level boundary" if summary["success"] else "未过关"
        ))
    first_action_event = next(
        (event for event in timeline["events"] if int(event["step"]) > 0), None
    )
    first_contracts = (
        _resolve(
            timeline,
            first_action_event["shared_decision_input"]["available_actions_ref"],
        )
        if first_action_event is not None
        else []
    )
    lines = [
        "# Agent 完整执行流程（按时间顺序）",
        "",
        "这份文档是 `timeline.json` 的人类可读投影。JSON 仍是机器审计源；这里不增加任何事后游戏规则。",
        "",
        "## 运行概览",
        "",
        f"- game harness：`{summary['game_id']}`",
        f"- model：`{summary['model']}`",
        f"- result：`{result_text}`",
        f"- actions：{summary['actions']}；model calls：{summary['model_calls']}",
        f"- levels：{summary.get('levels_passed', 0)}/{summary.get('levels_attempted', 0)}；"
        f"pass rate={float(summary.get('level_pass_rate', 0.0)):.2%}；"
        f"per-level max={summary.get('max_steps_per_level', 'checkpoint unknown')}；"
        f"runtime resets={summary.get('runtime_resets', 0)}",
        f"- failure reason：{summary.get('failure_reason') or '无'}",
        f"- provider requests：{summary.get('provider_requests', summary['model_calls'])}；"
        f"read_cognition calls：{summary.get('cognition_tool_calls', 0)}",
        f"- final cognition：`{summary['final_cognition']}`",
        "- 本次公开 action 合同："
        + "；".join(_contract_line(item) for item in first_contracts),
        "",
        "每个动作 step 的固定触发链是：",
        "",
        "```text",
        "公开画面 → Explore 子 Agent 生成 probe → Main Agent 选动作",
        "→ 环境执行 → 产生 before/action/after Evidence",
            "→ Update 子 Agent 提图更新 → validator 提交 EFPS/Insight transaction",
        "```",
        "",
    ]

    if include_prompt_text:
        lines.extend(["## 本次实际 System Instructions", ""])
        for profile, instruction in (timeline.get("instruction_profiles") or {}).items():
            lines.extend(
                [
                    f"### `{profile}`",
                    "",
                    "```text",
                    str(instruction).strip(),
                    "```",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "## Prompt 记录策略",
                "",
                "本报告不重复全量 system/user prompt。实际 instruction profile 为：`"
                + "`、`".join((timeline.get("instruction_profiles") or {}).keys())
                + "`；完整机器审计输入仍在 `timeline.json`。下面逐步保留 Agent 实际获得的"
                "公开状态、动作合同、预算、最近 transition，以及 EFPS 与全局 Insight 摘要。",
                "",
            ]
        )

    lines.extend(
        [
            "## 一眼看完整条时间线",
            "",
            "| 时刻 | 触发与 Agent 输出 | 执行动作 / 公开结果 | 观察图片 |",
            "|---:|---|---|---|",
        ]
    )
    for event in timeline["events"]:
        step = int(event["step"])
        if step == 0:
            observation = _resolve(
                timeline, event["update_input"]["observation_ref"]
            )
            output = event["agent_calls"]["update_agent"].get("output") or {}
            lines.append(
                f"| 0 | reset → Update：{len(output.get('entities') or [])} Entity，"
                f"{len(output.get('prototypes') or [])} Prototype，"
                f"{len(output.get('insight_updates') or [])} Insight proposals | 无动作 | "
                f"{_image_link(observation, '初始图')} |"
            )
            continue
        decision = event["decision"]
        transition = event["environment_transition"]
        update_output = event["agent_calls"]["update_agent"].get("output") or {}
        before = _resolve(
            timeline, event["update_input"]["before_observation_ref"]
        )
        after = _resolve(
            timeline, event["update_input"]["after_observation_ref"]
        )
        exploration_value = event["agent_calls"]["exploration_agent"].get(
            "output"
        )
        exploration_label = (
            "文本建议"
            if isinstance(exploration_value, str)
            else f"{len((exploration_value or {}).get('proposals', []))} probes"
        )
        lines.append(
            f"| {step} | Explore：{exploration_label} → Main："
            f"{decision.get('decision_mode')} → Update："
            f"{len(update_output.get('schema_updates') or [])} Schema，"
            f"{len(update_output.get('insight_updates') or [])} Insight proposals | "
            f"`{_action(decision.get('action'))}` / "
            f"{_compact_result(transition.get('result') or {})} | "
            f"{_image_link(before, '前')} / {_image_link(after, '后')} |"
        )
        if event.get("environment_reset"):
            lines[-1] = lines[-1][:-1] + "；GAME_OVER 后同关 reset（不返还步数） |"
    lines.append("")

    for event in timeline["events"]:
        step = int(event["step"])
        if step == 0:
            update_input = event["update_input"]
            observation = _resolve(timeline, update_input["observation_ref"])
            cognition = _resolve(timeline, update_input["cognition_before_ref"])
            initial_evidence = (
                _resolve(timeline, update_input["evidence_ref"])
                if update_input.get("evidence_ref")
                else "旧版记录仅有 ID"
            )
            call = event["agent_calls"]["update_agent"]
            update = event["cognitive_update"]
            if include_prompt_text and call.get("input_text"):
                received_lines = [
                    f"- Agent 实际图片：{_agent_image_link(observation, '初始公开画面')}",
                    f"- 人类审查辅助（未发送给 Agent）：{_image_link(observation, '带坐标辅助线版本')}",
                    *_actual_call_input_lines(call, include_text=True),
                ]
            else:
                received_lines = [
                    f"- Agent 实际图片：{_agent_image_link(observation, '初始公开画面')}",
                    f"- 人类审查辅助（未发送给 Agent）：{_image_link(observation, '带坐标辅助线版本')}",
                    f"- public state：`{observation.get('public_state')}`",
                    f"- Evidence：`{update_input.get('evidence_id')}`",
                    f"- Evidence 可解引用内容：`{_evidence_text(initial_evidence)}`",
                    f"- EFPS/Insight：revision={cognition.get('revision')}，counts={cognition.get('counts')}",
                    *_cognition_lines(cognition),
                    *_actual_call_input_lines(call, include_text=False),
                ]
            lines.extend(
                [
                    "## Step 0 — reset 后的第一次观察",
                    "",
                    "### 触发",
                    "",
                    "Python runtime 在创建 session 时调用环境 `reset()`，随后按固定初始化流程调用 Update。"
                    "此时 Main 和 Exploration 都没有被调用、也没有输入输出；由于还没有 action transition，"
                    "validator 禁止创建动作 Schema。",
                    "",
                    "### Update Agent 收到",
                    "",
                    *received_lines,
                    f"- instruction profile：`{call.get('instruction_profile')}`",
                    "",
                    "### Update Agent 输出",
                    "",
                    *_update_lines(call.get("output") or {}),
                    "",
                    "### 提交结果",
                    "",
                    f"- transaction：{_transaction_line(update)}",
                    f"- counts：`{update['counts_before']}` → `{update['counts_after']}`",
                    f"- warnings：`{update.get('warnings') or []}`",
                    "",
                ]
            )
            continue

        shared = event["shared_decision_input"]
        update_input = event["update_input"]
        current = _resolve(timeline, shared["current_observation_ref"])
        contracts = _resolve(timeline, shared["available_actions_ref"])
        cognition = _resolve(timeline, shared["cognition_ref"])
        before = _resolve(timeline, update_input["before_observation_ref"])
        after = _resolve(timeline, update_input["after_observation_ref"])
        calls = event["agent_calls"]
        exploration_output = calls["exploration_agent"].get("output") or {}
        decision = event["decision"]
        update_output = calls["update_agent"].get("output") or {}
        transition = event["environment_transition"]
        resolved_transition_evidence = (
            _resolve(timeline, transition["evidence_ref"])
            if transition.get("evidence_ref")
            else transition.get("evidence")
        )
        update = event["cognitive_update"]
        action_names = [item.get("name") for item in contracts]
        supporting = shared.get("attached_schema_evidence_images") or []
        prose_context = include_prompt_text and bool(
            calls["exploration_agent"].get("input_text")
        )
        if prose_context:
            shared_lines = [
                f"- Agent 实际图片：{_agent_image_link(current, '动作前公开画面')}",
                f"- 人类审查辅助（未发送给 Agent）：{_image_link(current, '带坐标辅助线版本')}",
                "- 默认不附加历史 Evidence 图片；Agent 可通过 `read_cognition` 精确读取记录或保存图片。",
                "",
                "### Exploration Agent 实际收到",
                "",
                *_actual_call_input_lines(
                    calls["exploration_agent"], include_text=True
                ),
            ]
        else:
            shared_lines = [
                f"- Agent 实际图片：{_agent_image_link(current, '动作前公开画面')}",
                f"- 人类审查辅助（未发送给 Agent）：{_image_link(current, '带坐标辅助线版本')}",
                f"- public state：`{current.get('public_state')}`",
                f"- 当时可用 action：`{action_names}`（参数合同见运行概览）",
                f"- EFPS 输入：revision={cognition.get('revision')}，counts={cognition.get('counts')}",
                *_cognition_lines(cognition),
                f"- 最近公开 transition：{_recent_history(shared.get('recent_public_transitions') or [], timeline)}",
                f"- 当前关卡动作预算：`{shared.get('level_budget') or {}}`",
                *_actual_call_input_lines(
                    calls["exploration_agent"], include_text=False
                ),
            ]
        lines.extend(
            [
                f"## Step {step} — Main 执行 `{_action(decision['action'])}`",
                "",
                "### 1. 触发 Explore 子 Agent",
                "",
                "这里的 runtime 是确定性的 Python 调度代码，不是另一个 Agent。它按循环调用 "
                "Exploration，再把文本建议交给 Main；Main 是唯一动作决策者，但不是这次函数调用的发起者。",
                "",
                "### Explore/Main 共同收到",
                "",
                *shared_lines,
                "",
                "### 2. Exploration Agent 输出",
                "",
                *_exploration_lines(exploration_output),
                "",
                "### Main Agent 实际收到",
                "",
                *_actual_call_input_lines(
                    calls["main_agent"], include_text=include_prompt_text
                ),
                "",
                "### 3. Main Agent 输出并触发环境动作",
                "",
                "- goal hypotheses：",
                "  - 说明：这些是 Main 自己输出的、可审计且可修正的工作假设，用来比较动作价值；"
                "不是环境提供的目标，也不是隐藏思维链。",
                *_goal_lines(decision.get("goal_hypotheses") or []),
                f"- decision mode：`{decision.get('decision_mode')}`",
                f"- selected action：`{_action(decision.get('action'))}`",
                f"- schemas used：`{decision.get('schema_ids') or []}`",
                f"- schema prediction：{_short(decision.get('schema_prediction'), 400)}",
                f"- insights used：`{decision.get('insight_ids') or []}`",
                f"- insight application：{_short(decision.get('insight_application'), 400)}",
                f"- exploration hypothesis：{_short(decision.get('exploration_hypothesis'), 280)}",
                f"- rationale：{_short(decision.get('rationale'), 320)}",
                "",
                "### 4. 环境返回公开 transition",
                "",
                f"- 动作前：{_image_link(before, 'before PNG')}",
                f"- 动作后：{_image_link(after, 'after PNG')}",
                f"- action：`{_action(transition.get('action'))}`",
                f"- result：`{_compact_result(transition.get('result') or {})}`",
                f"- Entity 语义结果：`{transition.get('semantic_result') or '旧版记录缺失'}`",
                f"- Evidence ID：`{transition.get('evidence_id')}`",
                f"- Evidence 可解引用内容：`{_evidence_text(resolved_transition_evidence or '旧版记录仅有 ID')}`",
                "- Artifact 边界：上面是 Evidence 的持久化清单，不是模型输入清单；"
                "只有 `agent_view` 会作为 Agent 图片输入，`review_view` 仅供人类审查，"
                "`environment_state` 仅供 diff、校验和 replay。",
                "",
                "### 5. 触发 Update 子 Agent",
                "",
                "Python runtime 在环境返回后调用 Update。Update 收到无辅助线的 before/after 原始图像对、"
                "实际动作、公开 result、Main decision 和更新前 EFPS。Update 只提出 operations；"
                "确定性 translator/validator 校验引用和 Evidence 后，由 `EFPSGraph.apply_transaction()` "
                "原子提交图修改。",
                "",
                "### Update Agent 实际收到",
                "",
                *_actual_call_input_lines(
                    calls["update_agent"], include_text=include_prompt_text
                ),
                "",
                "### Update Agent 输出",
                "",
                *_update_lines(update_output),
                "",
                "### 6. Validator 提交结果",
                "",
                f"- transaction：{_transaction_line(update)}",
                f"- counts：`{update['counts_before']}` → `{update['counts_after']}`",
                f"- warnings：`{update.get('warnings') or []}`",
                "",
            ]
        )
        reset = event.get("environment_reset")
        if reset:
            reset_input = reset.get("update_input") or {}
            reset_observation = _resolve(
                timeline, reset_input["observation_ref"]
            )
            reset_cognition = _resolve(
                timeline, reset_input["cognition_before_ref"]
            )
            reset_evidence = (
                _resolve(timeline, reset["evidence_ref"])
                if reset.get("evidence_ref")
                else reset.get("evidence")
            )
            reset_call = calls.get("recovery_update_agent") or {}
            reset_update = reset.get("cognitive_update") or {}
            lines.extend(
                [
                    f"### 7. GAME_OVER 后恢复当前 Level（Step {step}R）",
                    "",
                    "runtime 依据公开 `GAME_OVER` 状态调用环境 reset；它不是 Agent 动作，"
                    "不增加 action 计数，也不返还本关已经消耗的 action。reset 后仅调用 Update "
                    "重新对齐当前公开画面，Main/Exploration 不参与这次恢复。",
                    "",
                    f"- reset 后公开图：{_agent_image_link(reset_observation, '恢复后的 Agent 画面')}",
                    f"- 人类审查图：{_image_link(reset_observation, '恢复后的审查画面')}",
                    f"- public result：`{reset.get('result') or {}}`",
                    f"- 本关剩余预算：`{reset_input.get('level_budget') or {}}`",
                    f"- Evidence：`{_evidence_text(reset_evidence)}`",
                    f"- 恢复前 EFPS：revision={reset_cognition.get('revision')}，counts={reset_cognition.get('counts')}",
                    *_cognition_lines(reset_cognition),
                    *_actual_call_input_lines(
                        reset_call, include_text=include_prompt_text
                    ),
                    "",
                    "#### Recovery Update 输出",
                    "",
                    *_update_lines(reset_call.get("output") or {}),
                    "",
                    f"- transaction：{_transaction_line(reset_update)}",
                    f"- counts：`{reset_update.get('counts_before')}` → `{reset_update.get('counts_after')}`",
                    f"- warnings：`{reset_update.get('warnings') or []}`",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["build_process_markdown"]
