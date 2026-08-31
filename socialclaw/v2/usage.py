from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List


def build_usage_report(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    by_agent: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "logical_calls": 0,
            "provider_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "image_attachments": 0,
            "tool_calls": 0,
            "tool_result_characters": 0,
        }
    )
    by_step: List[Dict[str, Any]] = []
    by_section: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"appearances": 0, "characters": 0, "utf8_bytes": 0}
    )
    calls: List[Dict[str, Any]] = []
    request_phases: Dict[str, Dict[str, int]] = {
        "first_request_per_logical_call": {
            "provider_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
        "additional_requests": {
            "provider_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }
    for event in events:
        step_row = {
            "step": int(event.get("step") or 0),
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "provider_requests": 0,
            "tool_calls": 0,
        }
        for agent, audit in (event.get("agent_calls") or {}).items():
            usage = audit.get("usage") or {}
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            total_tokens = int(
                usage.get("total_tokens") or input_tokens + output_tokens
            )
            rounds = list(audit.get("usage_rounds") or [])
            provider_requests = len(rounds) or (1 if total_tokens else 0)
            trace = list(audit.get("tool_trace") or [])
            images = list(audit.get("image_inputs") or [])
            tool_images = [
                image
                for item in trace
                for image in item.get("returned_images") or []
            ]
            row = by_agent[agent]
            row["logical_calls"] += 1
            row["provider_requests"] += provider_requests
            row["input_tokens"] += input_tokens
            row["output_tokens"] += output_tokens
            row["total_tokens"] += total_tokens
            row["image_attachments"] += len(images) + len(tool_images)
            row["tool_calls"] += len(trace)
            row["tool_result_characters"] += sum(
                int(item.get("result_characters") or 0) for item in trace
            )
            step_row["input_tokens"] += input_tokens
            step_row["output_tokens"] += output_tokens
            step_row["total_tokens"] += total_tokens
            step_row["provider_requests"] += provider_requests
            step_row["tool_calls"] += len(trace)
            if rounds:
                for round_index, round_usage in enumerate(rounds):
                    phase = (
                        "first_request_per_logical_call"
                        if round_index == 0
                        else "additional_requests"
                    )
                    phase_row = request_phases[phase]
                    round_input = int(round_usage.get("input_tokens") or 0)
                    round_output = int(round_usage.get("output_tokens") or 0)
                    round_total = int(
                        round_usage.get("total_tokens")
                        or round_input + round_output
                    )
                    phase_row["provider_requests"] += 1
                    phase_row["input_tokens"] += round_input
                    phase_row["output_tokens"] += round_output
                    phase_row["total_tokens"] += round_total
            elif total_tokens:
                phase_row = request_phases["first_request_per_logical_call"]
                phase_row["provider_requests"] += 1
                phase_row["input_tokens"] += input_tokens
                phase_row["output_tokens"] += output_tokens
                phase_row["total_tokens"] += total_tokens
            for section in audit.get("input_sections") or []:
                name = str(section.get("section") or "unknown")
                value = by_section[name]
                value["appearances"] += 1
                value["characters"] += int(section.get("characters") or 0)
                value["utf8_bytes"] += int(section.get("utf8_bytes") or 0)
            calls.append(
                {
                    "step": step_row["step"],
                    "agent": agent,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "provider_requests": provider_requests,
                    "usage_rounds": rounds,
                    "image_attachments": len(images) + len(tool_images),
                    "image_artifact_ids": [
                        item.get("artifact_id") for item in [*images, *tool_images]
                    ],
                    "tool_calls": trace,
                    "input_characters": len(str(audit.get("input_text") or "")),
                    "input_sections": list(audit.get("input_sections") or []),
                }
            )
        by_step.append(step_row)

    totals = {
        "logical_calls": sum(item["logical_calls"] for item in by_agent.values()),
        "provider_requests": sum(
            item["provider_requests"] for item in by_agent.values()
        ),
        "input_tokens": sum(item["input_tokens"] for item in by_agent.values()),
        "output_tokens": sum(item["output_tokens"] for item in by_agent.values()),
        "total_tokens": sum(item["total_tokens"] for item in by_agent.values()),
        "image_attachments": sum(
            item["image_attachments"] for item in by_agent.values()
        ),
        "tool_calls": sum(item["tool_calls"] for item in by_agent.values()),
        "tool_result_characters": sum(
            item["tool_result_characters"] for item in by_agent.values()
        ),
    }
    total_characters = sum(item["characters"] for item in by_section.values()) or 1
    sections = {
        name: {
            **value,
            "character_share": round(value["characters"] / total_characters, 6),
        }
        for name, value in sorted(
            by_section.items(), key=lambda item: item[1]["characters"], reverse=True
        )
    }
    return {
        "measurement_note": (
            "Provider token counts are exact per request. Section allocation uses exact "
            "characters/UTF-8 bytes because the provider does not return field-level tokens; "
            "character shares must not be presented as exact token shares."
        ),
        "totals": totals,
        "by_request_phase": request_phases,
        "by_agent": dict(sorted(by_agent.items())),
        "by_step": by_step,
        "by_input_section": sections,
        "calls": calls,
    }


def usage_markdown(report: Dict[str, Any]) -> str:
    totals = report["totals"]
    lines = [
        "# Token 与认知工具消耗统计",
        "",
        report["measurement_note"],
        "",
        "## 总量",
        "",
        f"- logical Agent calls：{totals['logical_calls']}",
        f"- provider requests（包含 tool continuation/重试）：{totals['provider_requests']}",
        f"- input tokens：{totals['input_tokens']:,}",
        f"- output tokens：{totals['output_tokens']:,}",
        f"- total tokens：{totals['total_tokens']:,}",
        f"- image attachments：{totals['image_attachments']}",
        f"- read_cognition calls：{totals['tool_calls']}",
        f"- tool result characters：{totals['tool_result_characters']:,}",
        "",
        "## 按 Agent",
        "",
        "| Agent | Logical calls | Provider requests | Input | Output | Total | Images | Tools |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for agent, item in report["by_agent"].items():
        lines.append(
            f"| {agent} | {item['logical_calls']} | {item['provider_requests']} | "
            f"{item['input_tokens']:,} | {item['output_tokens']:,} | "
            f"{item['total_tokens']:,} | {item['image_attachments']} | {item['tool_calls']} |"
        )
    lines.extend(
        [
            "",
            "## 首轮与追加 provider request",
            "",
            "每个逻辑 Agent 调用的第一轮是默认输入；后续轮包含认知工具续轮、JSON 修复或语义纠错。"
            "这是 provider 返回的精确 token，不是字符估算。",
            "",
            "| Request phase | Requests | Input | Output | Total | Total share |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for phase, item in report["by_request_phase"].items():
        share = item["total_tokens"] / (totals["total_tokens"] or 1)
        lines.append(
            f"| {phase} | {item['provider_requests']} | {item['input_tokens']:,} | "
            f"{item['output_tokens']:,} | {item['total_tokens']:,} | {share * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 按步骤",
            "",
            "| Step | Provider requests | Input | Output | Total | Tool calls |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["by_step"]:
        lines.append(
            f"| {item['step']} | {item['provider_requests']} | {item['input_tokens']:,} | "
            f"{item['output_tokens']:,} | {item['total_tokens']:,} | {item['tool_calls']} |"
        )
    lines.extend(
        [
            "",
            "## 默认输入区段体积",
            "",
            "这里是精确字符/字节统计，不是 provider 字段级 token。",
            "",
            "| Section | Appearances | Characters | UTF-8 bytes | Character share |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in report["by_input_section"].items():
        lines.append(
            f"| {name.replace('|', '/')} | {item['appearances']} | "
            f"{item['characters']:,} | {item['utf8_bytes']:,} | "
            f"{item['character_share'] * 100:.2f}% |"
        )
    cognition = report["by_input_section"].get("Current learned cognition", {})
    lines.extend(
        [
            "",
            "## 认知图输入/输出的归因边界",
            "",
            f"- 默认 prompt 中 `Current learned cognition` 共 "
            f"{int(cognition.get('characters') or 0):,} 字符，占所有默认输入区段字符的 "
            f"{float(cognition.get('character_share') or 0) * 100:.2f}%。这是字符占比，不是精确 token 占比。",
            f"- `read_cognition` 共返回 {totals['tool_result_characters']:,} 个结果字符；它们进入后续 "
            "provider request，且可能随会话再次计费，因此不能与默认区段字符简单相加后换算 token。",
            f"- Update Agent 的全部输出为 {report['by_agent'].get('update_agent', {}).get('output_tokens', 0):,} "
            "tokens；其中同时包含场景摘要、transition 语义、Entity/Prototype/Schema proposal，provider "
            "没有返回字段级 token，所以不能把这项冒充为纯认知图输出 token。",
            "",
            "逐调用 usage、tool 参数/结果、图片 ID 和输入区段见 `token_usage.json`。",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["build_usage_report", "usage_markdown"]
