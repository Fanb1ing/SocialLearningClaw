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
                    f"[bold yellow]Proactive Question[/bold yellow]\n"
                    f"[bold]Question:[/bold]{question}\n\n"
                    f"[bold]Context:[/bold]{context[:300]}...\n\n"
                    f"[bold cyan]Hint:[/bold cyan]{hint}",
                    title="Schema Initialization",
                    border_style="yellow",
                )
            )
        else:
            print("\n" + "─" * 50)
            print("[Proactive Question] The current problem is missing necessary concepts:")
            print(f"Question: {question}")
            print(f"Context: {context[:300]}...")
            print(f"Hint: {hint}")
            print("─" * 50)

        print("Please enter your answer (multiple lines, empty line to finish):")
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
        problem_preview = problem.prompt[:800] + "..." if len(problem.prompt) > 800 else problem.prompt
        answer_preview = attempt.answer_text[:600] + "..." if len(attempt.answer_text) > 600 else attempt.answer_text

        if self.console:
            self.console.print()
            tree = Tree("[bold red]High-Confidence Error — Correction Needed[/bold red]")
            tree.add(f"Problem ID: {problem.id} ({problem.problem_type})")
            tree.add(f"Evaluation: {eval.details or ('Correct' if eval.correct else 'Wrong')}")
            tree.add(f"Schema reasoning confidence: {reasoning_confidence:.3f}")

            prob_node = tree.add("[bold]Original Problem (first 800 chars)[/bold]")
            prob_node.add(problem_preview)

            ans_node = tree.add("[bold]LLM Answer (first 600 chars)[/bold]")
            ans_node.add(answer_preview)

            gold_node = tree.add("[bold]Ground Truth (first 600 chars)[/bold]")
            gold_node.add(eval.gold[:600] + ("..." if len(eval.gold) > 600 else ""))

            if trace.concepts:
                concepts_node = tree.add("[bold]Concepts Used[/bold]")
                for c in trace.concepts:
                    concepts_node.add(c)
            if trace.relations:
                rels_node = tree.add("[bold]Reasoning Path[/bold]")
                for src, tgt, rel_type in trace.relations:
                    rels_node.add(f"{src} -> [{rel_type}] -> {tgt}")
            if trace.explanation:
                tree.add(f"[bold]Explanation:[/bold]{trace.explanation}")

            self.console.print(
                Panel.fit(
                    tree,
                    title="Schema Correction",
                    border_style="red",
                )
            )
        else:
            print("\n" + "─" * 50)
            print("[High-Confidence Error] Schema reasoning confidence is high, but the answer is wrong.")
            print(f"Problem ID: {problem.id}")
            print(f"Evaluation: {eval.details or ('Correct' if eval.correct else 'Wrong')}")
            print(f"Confidence: {reasoning_confidence:.3f}")
            print(f"\nOriginal Problem (first 800 chars):\n{problem_preview}")
            print(f"\nLLM Answer (first 600 chars):\n{answer_preview}")
            print(f"\nGround Truth (first 600 chars):\n{eval.gold[:600] + ('...' if len(eval.gold) > 600 else '')}")
            print(f"\nConcepts Used: {trace.concepts}")
            print(f"Reasoning Path: {trace.relations}")
            print("─" * 50)

        print(
            "\nPlease point out problems in the schema (e.g. incorrect concept definitions, wrong relation direction, missing concepts):\n"
            "(Multiple lines, empty line to finish)"
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
