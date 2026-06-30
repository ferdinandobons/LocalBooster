#!/usr/bin/env python3
"""Convert the local MATH500 snapshot into LocalBooster JSONL benchmark records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPT_TEMPLATE = """Solve the following MATH benchmark problem.

Show the reasoning briefly. End your response with exactly one line in this format:
Final answer: VALUE

Problem:
{problem}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    selected = records[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(selected, start=args.start):
            output = {
                "id": record.get("id", f"math500-{index}"),
                "benchmark": "MATH500",
                "task": "math",
                "grader": "math",
                "prompt": PROMPT_TEMPLATE.format(problem=record["prompt"]),
                "answer": record["answer"],
                "source": record.get("source", "math"),
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
