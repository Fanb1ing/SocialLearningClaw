from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


class ToolError(Exception):
    pass


def tool_noop(_: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True}


def tool_calculator(args: Dict[str, Any]) -> Dict[str, Any]:
    """Very small and safe calculator.

    Accepts: {"expression": "1+2*(3-4)"}
    Only allows digits, operators, parentheses, dot, and whitespace.
    """

    expr = str(args.get("expression") or "").strip()
    if not expr:
        raise ToolError("expression is required")

    allowed = set("0123456789+-*/(). %")
    if any(ch not in allowed for ch in expr):
        raise ToolError("expression contains unsupported characters")

    # eval in restricted env
    val = eval(expr, {"__builtins__": {}}, {"sqrt": math.sqrt})
    return {"value": val}


@dataclass
class ToolRegistry:
    tools: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]

    @classmethod
    def safe_default(cls) -> "ToolRegistry":
        return cls(tools={"noop": tool_noop, "calculator": tool_calculator})

    def call(self, name: str, arguments: Any) -> Dict[str, Any]:
        if name not in self.tools:
            raise ToolError(f"unknown tool: {name}")

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments else {}
            except Exception as e:
                raise ToolError(f"invalid tool arguments json: {e}") from e
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ToolError("tool arguments must be an object")

        return self.tools[name](arguments)
