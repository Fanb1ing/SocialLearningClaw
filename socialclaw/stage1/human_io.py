from __future__ import annotations

from typing import List, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.tree import Tree

    _RICH_AVAILABLE = True
except Exception:
    _RICH_AVAILABLE = False

from .agent.base import AgentAttempt
from .dataset.base import EvalResult, Problem


class HumanIO:
    def __init__(self, auto_yes: bool = False):
        self.auto_yes = auto_yes
        self.console = Console() if _RICH_AVAILABLE else None

    def ask(self, question: str, context: str, hint: str) -> str:
        if self.auto_yes:
            return ""

        if self.console:
            self.console.print()
            self.console.print(
                Panel.fit(
                    f"[bold yellow]主动提问[/bold yellow]\n"
                    f"[bold]问题：[/bold]{question}\n\n"
                    f"[bold]上下文：[/bold]{context[:300]}...\n\n"
                    f"[bold cyan]提示：[/bold cyan]{hint}",
                    title="Schema 初始化",
                    border_style="yellow",
                )
            )
        else:
            print("\n" + "─" * 50)
            print("[主动提问] 当前问题缺少必要的概念：")
            print(f"问题：{question}")
            print(f"上下文：{context[:300]}...")
            print(f"提示：{hint}")
            print("─" * 50)

        print("请输入你的回答（多行，空行结束）：")
        lines: List[str] = []
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            lines.append(line)
        return "\n".join(lines)

    def ask_correction(
        self,
        problem: Problem,
        attempt: AgentAttempt,
        reasoning_confidence: float,
        eval: EvalResult,
    ) -> str:
        if self.auto_yes:
            return ""

        trace = attempt.reasoning_trace
        if self.console:
            self.console.print()
            tree = Tree("[bold red]高自信错误 — 需要纠错[/bold red]")
            tree.add(f"题目：{problem.id} ({problem.problem_type})")
            tree.add(f"预测：{eval.pred}")
            tree.add(f"标准答案：{eval.gold}")
            tree.add(f"Schema reasoning confidence：{reasoning_confidence:.3f}")

            if trace.concepts:
                concepts_node = tree.add("使用的概念")
                for c in trace.concepts:
                    concepts_node.add(c)
            if trace.relations:
                rels_node = tree.add("推理路径")
                for src, tgt, rel_type in trace.relations:
                    rels_node.add(f"{src} → [{rel_type}] → {tgt}")
            if trace.explanation:
                tree.add(f"解释：{trace.explanation}")

            self.console.print(
                Panel.fit(
                    tree,
                    title="Schema 纠错",
                    border_style="red",
                )
            )
        else:
            print("\n" + "─" * 50)
            print("[高自信错误] Schema 推理置信度很高，但答案错误。")
            print(f"题目：{problem.id}")
            print(f"预测：{eval.pred}, 标准答案：{eval.gold}")
            print(f"Confidence：{reasoning_confidence:.3f}")
            print(f"使用的概念：{trace.concepts}")
            print(f"推理路径：{trace.relations}")
            print("─" * 50)

        print(
            "\n请指出 schema 中的问题（如概念定义错误、关系方向错误、缺少概念等）：\n"
            "（多行输入，空行结束）"
        )
        lines: List[str] = []
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            lines.append(line)
        return "\n".join(lines)
