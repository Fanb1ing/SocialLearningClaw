from __future__ import annotations

from typing import List

from .types import Episode


def ask_key_points_cli(episode: Episode) -> List[str]:
    """Ask human for key knowledge points via CLI."""

    print("\n=== Proactive Question to Human ===")
    print("Problem:")
    print(episode.problem.prompt)
    if episode.problem.choices:
        print("Choices:")
        for c in episode.problem.choices:
            print("-", c)

    if episode.attempts:
        last = episode.attempts[-1]
        print("\nAgent answer:")
        print(last.get("answer_text", ""))
        print("\nEval:")
        if episode.evals:
            ev = episode.evals[-1]
            print(f"correct={ev.correct}, {ev.details}")

    print("\n请输入你认为解题最关键的知识点（可多条，空行结束）：")
    kps: List[str] = []
    while True:
        line = input("> ").strip()
        if not line:
            break
        kps.append(line)
    return kps
