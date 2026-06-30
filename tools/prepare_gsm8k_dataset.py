#!/usr/bin/env python3
"""Convert the official GSM8K JSONL format into LocalBooster benchmark records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPT_TEMPLATE = """Solve the following grade-school math problem.

Show the reasoning briefly. At the end, write one separate line with the words
Final answer, then a colon, then only the numeric answer.

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

    records = list(read_jsonl(args.input))
    selected = records[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(selected, start=args.start):
            answer = extract_gsm8k_answer(record["answer"])
            output = {
                "id": f"gsm8k-test-{index}",
                "benchmark": "GSM8K",
                "task": "grade_school_math",
                "grader": "math",
                "prompt": PROMPT_TEMPLATE.format(problem=record["question"]),
                "answer": answer,
                "source": "openai/grade-school-math",
                "source_license": "MIT",
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
    print(f"wrote {args.output}")
    return 0


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_gsm8k_answer(answer: str) -> str:
    marker = "####"
    if marker not in answer:
        raise ValueError("expected GSM8K answer to contain `####`")
    return answer.rsplit(marker, 1)[1].strip().replace(",", "")


if __name__ == "__main__":
    raise SystemExit(main())
