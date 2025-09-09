from __future__ import annotations

import os
from pathlib import Path
from typing import List
import argparse
from src.memory.memory_service import MemoryService
from src.agents import Coordinator


def run_and_capture(coord: Coordinator, question: str) -> str:
    res = coord.handle(question)
    return res.content


def main():
    parser = argparse.ArgumentParser(description="Run multi-agent scenarios or interact with agents")
    parser.add_argument("--out-dir", default="outputs", help="Directory to write scenario outputs (batch mode)")
    parser.add_argument("--prompt", default=None, help="Run a single user->agents turn with the given prompt")
    parser.add_argument("--interactive", action="store_true", help="Start a simple REPL to chat with the agents (enter blank line or 'exit' to quit)")
    parser.add_argument("--out-file", default=None, help="Optional file to save the single-turn result when using --prompt")
    args = parser.parse_args()

    mem = MemoryService()
    coord = Coordinator(mem)

    # Interactive REPL mode
    if args.interactive:
        print("Interactive mode. Type your question and press Enter. Blank line or 'exit' to quit.\n")
        while True:
            try:
                user_in = input("You: ").strip()
            except EOFError:
                break
            if not user_in or user_in.lower() in {"exit", "quit"}:
                break
            res = coord.handle(user_in)
            print("Agent:\n" + res.content + "\n")
        return

    # Single-turn mode
    if args.prompt:
        res = coord.handle(args.prompt)
        print(res.content)
        if args.out_file:
            Path(args.out_file).write_text(res.content, encoding="utf-8")
        return

    # Default: batch scenarios
    outputs_dir = Path(args.out_dir)
    outputs_dir.mkdir(exist_ok=True)

    scenarios = [
        ("simple_query.txt", "What are the main types of neural networks?"),
        ("complex_query.txt", "Research transformer architectures, analyze their computational efficiency, and summarize key trade-offs."),
        ("memory_test.txt", "What did we discuss about neural networks earlier?"),
        ("multi_step.txt", "Find recent papers on reinforcement learning, analyze their methodologies, and identify common challenges."),
        ("collaborative.txt", "Compare two machine-learning approaches and recommend which is better for our use case."),
    ]

    for fname, q in scenarios:
        print("==== Question ====")
        print(q)
        print("==================")
        ans = run_and_capture(coord, q)
        (outputs_dir / fname).write_text(ans, encoding="utf-8")
        print(ans)
        print()

    print(f"All scenarios executed. See {outputs_dir}/ folder.")


if __name__ == "__main__":
    main()
