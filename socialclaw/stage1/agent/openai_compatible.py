from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .base import AgentAttempt, ReasoningTrace, Usage


# Regex patterns for parsing reasoning trace
_CONCEPTS_RE = re.compile(
    r"(?:使用的概念|概念|concepts)[：:]\s*(.+?)(?=\n\s*(?:推理路径|路径|relations|解释|explanation)|\n\s*\[最终答案\]|$)",
    re.IGNORECASE | re.DOTALL,
)
_RELATIONS_RE = re.compile(
    r"(?:推理路径|路径|relations)[：:]\s*(.+?)(?=\n\s*(?:解释|explanation)|\n\s*\[最终答案\]|$)",
    re.IGNORECASE | re.DOTALL,
)
_EXPLANATION_RE = re.compile(
    r"(?:解释|explanation)[：:]\s*(.+?)(?=\n\s*\[最终答案\]|$)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_block(text: str, start_marker: str, end_marker: Optional[str] = None) -> str:
    """Extract content between start_marker and end_marker (or end of string)."""
    idx = text.find(start_marker)
    if idx == -1:
        return ""
    start = idx + len(start_marker)
    if end_marker:
        end = text.find(end_marker, start)
        if end == -1:
            end = len(text)
    else:
        end = len(text)
    return text[start:end].strip()


def _clean_braces(text: str) -> str:
    """Remove surrounding braces/parentheses and strip."""
    text = text.strip()
    for prefix in ["{", "(", "[", "<"]:
        if text.startswith(prefix):
            text = text[1:]
    for suffix in ["}", ")", "]", ">"]:
        if text.endswith(suffix):
            text = text[:-1]
    return text.strip()


def _parse_concepts(text: str) -> List[str]:
    concepts: List[str] = []
    m = _CONCEPTS_RE.search(text)
    if m:
        raw = m.group(1)
        # Split by newlines first
        for line in raw.split("\n"):
            line = line.strip().lstrip("-–•1234567890. ").strip()
            if not line:
                continue
            line = _clean_braces(line)
            if not line:
                continue
            # Split by commas if present
            if "," in line:
                for part in line.split(","):
                    part = _clean_braces(part)
                    if part and len(part) < 80:
                        concepts.append(part)
            elif len(line) < 120:
                concepts.append(line)
    return concepts


def _parse_relations(text: str) -> List[Tuple[str, str, str]]:
    relations: List[Tuple[str, str, str]] = []
    m = _RELATIONS_RE.search(text)
    if not m:
        return relations
    raw = m.group(1)
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = _clean_braces(line)
        if not line:
            continue
        # Normalize arrow
        if "->" in line or "→" in line:
            normalized = line.replace("→", "->")
            parts = [p.strip() for p in normalized.split("->") if p.strip()]
            if len(parts) >= 2:
                # Chain: A -> rel1 -> B -> rel2 -> C produces (A,rel1,B) and (B,rel2,C)
                i = 0
                while i + 2 < len(parts):
                    src = parts[i]
                    rel_type = parts[i + 1]
                    tgt = parts[i + 2]
                    relations.append((src, tgt, rel_type))
                    i += 2
                # If only two parts, single relation
                if len(parts) == 2:
                    relations.append((parts[0], parts[1], "related"))
    return relations


def _parse_explanation(text: str) -> str:
    m = _EXPLANATION_RE.search(text)
    if m:
        return m.group(1).strip()
    return ""


def _parse_reasoning_trace(content: str) -> ReasoningTrace:
    reasoning_block = _extract_block(content, "[推理过程]", "[最终答案]")
    if not reasoning_block:
        # Fallback: try to find any section that looks like reasoning
        reasoning_block = content

    return ReasoningTrace(
        concepts=_parse_concepts(reasoning_block),
        relations=_parse_relations(reasoning_block),
        explanation=_parse_explanation(reasoning_block),
    )


def _parse_answer_text(content: str) -> str:
    ans = _extract_block(content, "[最终答案]")
    if ans:
        return ans

    # If content looks like a markdown code block, extract inner content
    stripped = content.strip()
    if stripped.startswith("```"):
        # Remove opening ```lang and closing ```
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        inner = "\n".join(lines).strip()
        if inner:
            return inner

    # Fallback: return the last non-empty, non-bracket line
    for line in reversed(content.split("\n")):
        line_stripped = line.strip()
        if line_stripped and not line_stripped.startswith("["):
            return line_stripped
    return content.strip()


def _try_load_image_data_url(*, meta: Dict[str, Any]) -> Optional[str]:
    problem_meta = meta.get("problem_meta") or {}
    if not isinstance(problem_meta, dict):
        return None
    image_rel = problem_meta.get("image")
    if not isinstance(image_rel, str) or not image_rel:
        return None

    prepared_dataset_path = meta.get("prepared_dataset_path")
    if not isinstance(prepared_dataset_path, str) or not prepared_dataset_path:
        return None

    prepared_dir = os.path.dirname(prepared_dataset_path)
    image_abs = os.path.join(prepared_dir, image_rel)
    if not os.path.exists(image_abs):
        return None

    try:
        with open(image_abs, "rb") as f:
            b = f.read()
    except Exception:
        return None

    if not b:
        return None

    b64 = base64.b64encode(b).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@dataclass
class OpenAICompatibleAgent:
    base_url: str
    api_key: str
    model: str
    timeout_s: int = 120
    temperature: float = 0.2

    def answer(self, *, prompt: str, meta: Dict[str, Any]) -> AgentAttempt:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        image_data_url = _try_load_image_data_url(meta=meta)
        if image_data_url:
            messages: List[Dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()

        usage = data.get("usage") or {}
        total_prompt_tokens = int(usage.get("prompt_tokens") or 0)
        total_completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (total_prompt_tokens + total_completion_tokens))

        choice0 = (data.get("choices") or [{}])[0]
        msg = choice0.get("message") or {}
        content = msg.get("content") or ""

        trace = _parse_reasoning_trace(content)
        answer_text = _parse_answer_text(content)

        return AgentAttempt(
            answer_text=answer_text,
            reasoning_trace=trace,
            usage=Usage(
                input_tokens=total_prompt_tokens,
                output_tokens=total_completion_tokens,
                total_tokens=total_tokens,
            ),
            raw={"response": data, "meta": meta, "messages": messages},
        )
