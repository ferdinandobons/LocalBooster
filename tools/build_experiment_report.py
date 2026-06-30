#!/usr/bin/env python3
"""Build a tracked experiment folder from LocalBooster JSONL results.

The script is dependency-free by design. It writes CSV/JSON summaries and simple SVG plots that
can be reviewed in GitHub without installing plotting libraries.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class SummaryRow:
    backend: str
    model: str
    sampler: str
    runs: int
    avg_latency_seconds: float
    avg_cost_multiplier: float | None
    avg_acceptance_ratio: float | None
    avg_accuracy: float | None
    avg_generated_tokens: float
    avg_sampled_tokens: float

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "model": self.model,
            "sampler": self.sampler,
            "runs": self.runs,
            "avg_latency_seconds": self.avg_latency_seconds,
            "avg_cost_multiplier": self.avg_cost_multiplier,
            "avg_acceptance_ratio": self.avg_acceptance_ratio,
            "avg_accuracy": self.avg_accuracy,
            "avg_generated_tokens": self.avg_generated_tokens,
            "avg_sampled_tokens": self.avg_sampled_tokens,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    records = list(read_jsonl(args.input))
    summaries = summarize(records)
    write_experiment(args.input, args.output_dir, args.title, records, summaries)
    print(f"wrote {args.output_dir}")
    return 0


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarize(records: list[dict]) -> list[SummaryRow]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    order: list[tuple[str, str, str]] = []
    for record in records:
        metrics = record["result"]["metrics"]
        sampler = record.get("sampler", metrics["sampler"])
        key = (metrics["backend"], metrics["model"], sampler)
        if key not in groups:
            order.append(key)
        groups[key].append(record)

    rows = []
    for backend, model, sampler in order:
        group = groups[(backend, model, sampler)]
        metrics = [record["result"]["metrics"] for record in group]
        rows.append(
            SummaryRow(
                backend=backend,
                model=model,
                sampler=sampler,
                runs=len(group),
                avg_latency_seconds=mean(metric["latency_seconds"] for metric in metrics),
                avg_cost_multiplier=mean_optional(
                    metric.get("cost_multiplier") for metric in metrics
                ),
                avg_acceptance_ratio=mean_optional(
                    metric.get("acceptance_ratio") for metric in metrics
                ),
                avg_accuracy=mean_optional(
                    1.0 if record["score"]["correct"] else 0.0
                    for record in group
                    if record.get("score") is not None
                ),
                avg_generated_tokens=mean(metric["generated_tokens"] for metric in metrics),
                avg_sampled_tokens=mean(metric["sampled_tokens"] for metric in metrics),
            )
        )
    return rows


def mean_optional(values: Iterable[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return mean(filtered)


def write_experiment(
    input_path: Path,
    output_dir: Path,
    title: str,
    records: list[dict],
    summaries: list[SummaryRow],
) -> None:
    data_dir = output_dir / "data"
    plots_dir = output_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(input_path, data_dir / "results.jsonl")
    write_summary_csv(data_dir / "summary.csv", summaries)
    write_summary_json(data_dir / "summary.json", title, records, summaries)

    write_bar_chart(
        plots_dir / "latency_seconds.svg",
        "Average Latency",
        summaries,
        "avg_latency_seconds",
        "seconds",
    )
    write_bar_chart(
        plots_dir / "cost_multiplier.svg",
        "Average Cost Multiplier",
        summaries,
        "avg_cost_multiplier",
        "x standard generation",
    )
    write_bar_chart(
        plots_dir / "acceptance_ratio.svg",
        "MCMC Acceptance Ratio",
        summaries,
        "avg_acceptance_ratio",
        "ratio",
    )
    if any(row.avg_accuracy is not None for row in summaries):
        write_bar_chart(
            plots_dir / "accuracy.svg",
            "Benchmark Accuracy",
            summaries,
            "avg_accuracy",
            "fraction correct",
        )
    write_grouped_token_chart(plots_dir / "tokens.svg", summaries)
    write_readme(output_dir / "README.md", title, records, summaries)


def write_summary_csv(path: Path, summaries: list[SummaryRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summaries[0].as_dict()),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in summaries:
            writer.writerow(row.as_dict())


def write_summary_json(
    path: Path,
    title: str,
    records: list[dict],
    summaries: list[SummaryRow],
) -> None:
    payload = {
        "title": title,
        "record_count": len(records),
        "summaries": [row.as_dict() for row in summaries],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_readme(
    path: Path,
    title: str,
    records: list[dict],
    summaries: list[SummaryRow],
) -> None:
    model = summaries[0].model if summaries else "unknown"
    backend = summaries[0].backend if summaries else "unknown"
    generated_tokens = summaries[0].avg_generated_tokens if summaries else 0
    has_scores = any(row.avg_accuracy is not None for row in summaries)
    purpose = (
        "quality, runtime, and cost benchmark"
        if has_scores
        else "runtime and cost smoke test, not an accuracy benchmark"
    )
    lines = [
        f"# {title}",
        "",
        "This folder contains a tracked LocalBooster experiment.",
        "",
        "## Scope",
        "",
        f"- backend: `{backend}`",
        f"- model: `{model}`",
        f"- records: `{len(records)}` JSONL rows",
        f"- purpose: {purpose}",
        (
            f"- note: runs used short {generated_tokens:.0f}-token completions, "
            "so answers may be incomplete"
        ),
        "",
        "## Summary",
        "",
    ]
    if has_scores:
        lines.extend(
            [
                (
                    "| Sampler | Runs | Accuracy | Avg Latency | Avg Cost x | "
                    "Avg Acceptance | Avg Generated | Avg Sampled |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    else:
        lines.extend(
            [
                (
                    "| Sampler | Runs | Avg Latency | Avg Cost x | Avg Acceptance | "
                    "Avg Generated | Avg Sampled |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    for row in summaries:
        if has_scores:
            lines.append(
                f"| `{row.sampler}` | {row.runs} | {fmt_optional(row.avg_accuracy)} | "
                f"{row.avg_latency_seconds:.2f}s | {fmt_optional(row.avg_cost_multiplier)} | "
                f"{fmt_optional(row.avg_acceptance_ratio)} | "
                f"{row.avg_generated_tokens:.1f} | {row.avg_sampled_tokens:.1f} |"
            )
        else:
            lines.append(
                f"| `{row.sampler}` | {row.runs} | {row.avg_latency_seconds:.2f}s | "
                f"{fmt_optional(row.avg_cost_multiplier)} | "
                f"{fmt_optional(row.avg_acceptance_ratio)} | "
                f"{row.avg_generated_tokens:.1f} | {row.avg_sampled_tokens:.1f} |"
            )
    chart_lines = [
        "![Average latency](plots/latency_seconds.svg)",
        "",
        "![Cost multiplier](plots/cost_multiplier.svg)",
        "",
        "![Acceptance ratio](plots/acceptance_ratio.svg)",
        "",
    ]
    if has_scores:
        chart_lines.extend(
            [
                "![Benchmark accuracy](plots/accuracy.svg)",
                "",
            ]
        )
    chart_lines.extend(
        [
            "![Token counts](plots/tokens.svg)",
            "",
        ]
    )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `data/results.jsonl`: raw LocalBooster output rows",
            "- `data/summary.csv`: tabular aggregate metrics",
            "- `data/summary.json`: aggregate metrics as JSON",
            "- `plots/latency_seconds.svg`: average latency by sampler",
            "- `plots/cost_multiplier.svg`: average sampled-token cost multiplier",
            "- `plots/acceptance_ratio.svg`: MCMC acceptance ratio where applicable",
            "- `plots/accuracy.svg`: benchmark accuracy, when scored answers are present",
            "- `plots/tokens.svg`: generated vs sampled token counts",
            "",
            "## Charts",
            "",
        ]
        + chart_lines
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_bar_chart(
    path: Path,
    title: str,
    summaries: list[SummaryRow],
    attr: str,
    unit: str,
) -> None:
    values = [(row.sampler, getattr(row, attr)) for row in summaries]
    values = [(label, value) for label, value in values if value is not None]
    if not values:
        values = [("n/a", 0.0)]
    max_value = max(value for _, value in values) or 1.0
    width = 720
    height = 320
    margin_left = 150
    margin_top = 52
    bar_height = 42
    gap = 24
    chart_width = 480
    svg = svg_header(width, height, title)
    for index, (label, value) in enumerate(values):
        y = margin_top + index * (bar_height + gap)
        bar_width = chart_width * value / max_value
        svg.append(svg_text(24, y + 27, label, size=14, anchor="start"))
        svg.append(
            f'<rect x="{margin_left}" y="{y}" width="{bar_width:.2f}" '
            f'height="{bar_height}" rx="4" fill="#3366cc" />'
        )
        svg.append(
            svg_text(
                margin_left + bar_width + 10,
                y + 27,
                f"{value:.2f} {unit}",
                size=13,
                anchor="start",
            )
        )
    svg.append("</svg>\n")
    path.write_text("\n".join(svg), encoding="utf-8")


def write_grouped_token_chart(path: Path, summaries: list[SummaryRow]) -> None:
    width = 760
    height = 360
    margin_left = 150
    margin_top = 58
    group_gap = 34
    bar_height = 18
    chart_width = 500
    max_value = max(row.avg_sampled_tokens for row in summaries) or 1.0
    svg = svg_header(width, height, "Generated vs Sampled Tokens")
    for index, row in enumerate(summaries):
        y = margin_top + index * (bar_height * 2 + group_gap)
        generated_width = chart_width * row.avg_generated_tokens / max_value
        sampled_width = chart_width * row.avg_sampled_tokens / max_value
        svg.append(svg_text(24, y + 26, row.sampler, size=14, anchor="start"))
        svg.append(
            f'<rect x="{margin_left}" y="{y}" width="{generated_width:.2f}" '
            f'height="{bar_height}" rx="3" fill="#109d58" />'
        )
        svg.append(
            f'<rect x="{margin_left}" y="{y + bar_height + 6}" '
            f'width="{sampled_width:.2f}" height="{bar_height}" rx="3" fill="#db4437" />'
        )
        svg.append(svg_text(margin_left + generated_width + 10, y + 14, "generated", size=12))
        svg.append(
            svg_text(
                margin_left + sampled_width + 10,
                y + bar_height + 20,
                "sampled",
                size=12,
            )
        )
    svg.append("</svg>\n")
    path.write_text("\n".join(svg), encoding="utf-8")


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        svg_text(24, 32, title, size=20, weight="700", anchor="start"),
    ]


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 13,
    weight: str = "400",
    anchor: str = "start",
) -> str:
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="#222">{escaped}</text>'
    )


def fmt_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
