#!/usr/bin/env python3
"""Recompute score fields for an existing LocalBooster result JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localbooster.evaluation import score_response


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as source, args.output.open(
        "w",
        encoding="utf-8",
    ) as target:
        for line in source:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            expected = record.get("expected")
            completion = record.get("completion")
            if expected is not None and completion is not None:
                prior_score = record.get("score") or {}
                record["score"] = score_response(
                    completion,
                    str(expected),
                    prior_score.get("grader", "auto"),
                ).to_dict()
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
