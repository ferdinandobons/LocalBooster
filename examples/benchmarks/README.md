# Benchmark Examples

This folder contains small JSONL benchmark slices for local LocalBooster runs.

- `smoke_reasoning.jsonl` and `smoke_coding.jsonl` are local smoke datasets.
- `math500_mini.jsonl` is converted from the local MATH500 snapshot included in the
  official `reasoning-with-sampling` reference folder.
- `gsm8k_mini.jsonl` is converted from OpenAI's GSM8K / grade-school-math test split,
  which is distributed under the MIT license.

These files are intentionally small. They are useful for repeatable local comparisons, not
for claiming leaderboard-level benchmark results.
