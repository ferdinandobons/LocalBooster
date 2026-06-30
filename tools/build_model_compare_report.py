#!/usr/bin/env python3
"""Build a model/sampler comparison report from experiment summary JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "experiments",
        nargs="+",
        help="Entries formatted as LABEL:PATH_TO_SUMMARY_JSON",
    )
    args = parser.parse_args()

    rows = []
    for spec in args.experiments:
        label, path_str = spec.split(":", 1)
        payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
        for summary in payload["summaries"]:
            row = dict(summary)
            row["label"] = label
            row["experiment_title"] = payload["title"]
            rows.append(row)
    if not rows:
        parser.error("all experiment summaries were empty")

    data_dir = args.output_dir / "data"
    plots_dir = args.output_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    write_csv(data_dir / "model_compare.csv", rows)
    (data_dir / "model_compare.json").write_text(
        json.dumps({"rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_bar_chart(plots_dir / "accuracy.svg", "Accuracy", rows, "avg_accuracy", "fraction")
    write_bar_chart(
        plots_dir / "latency_seconds.svg",
        "Average Latency",
        rows,
        "avg_latency_seconds",
        "seconds",
    )
    write_bar_chart(
        plots_dir / "cost_multiplier.svg",
        "Average Cost Multiplier",
        rows,
        "avg_cost_multiplier",
        "x",
    )
    write_readme(args.output_dir / "README.md", rows)
    print(f"wrote {args.output_dir}")
    return 0


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["label"] + [key for key in rows[0] if key != "label"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, rows: list[dict]) -> None:
    lines = [
        "# GSM8K Mini Clean3 Model Compare",
        "",
        "This report compares model size and sampler choice on the same three GSM8K examples.",
        "",
        "| Label | Sampler | Runs | Accuracy | Avg Latency | Cost x | Acceptance |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['label']}` | `{row['sampler']}` | {row['runs']} | "
            f"{fmt(row.get('avg_accuracy'))} | {row['avg_latency_seconds']:.2f}s | "
            f"{fmt(row.get('avg_cost_multiplier'))} | "
            f"{fmt(row.get('avg_acceptance_ratio'))} |"
        )
    lines.extend(
        [
            "",
            "## Charts",
            "",
            "![Accuracy](plots/accuracy.svg)",
            "",
            "![Average latency](plots/latency_seconds.svg)",
            "",
            "![Cost multiplier](plots/cost_multiplier.svg)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bar_chart(path: Path, title: str, rows: list[dict], metric: str, unit: str) -> None:
    values = [
        (f"{row['label']} {row['sampler']}", row.get(metric))
        for row in rows
        if row.get(metric) is not None
    ]
    if not values:
        values = [("n/a", 0.0)]
    max_value = max(value for _, value in values) or 1.0
    width = 860
    row_height = 34
    margin_left = 260
    margin_top = 56
    chart_width = 500
    height = margin_top + len(values) * row_height + 52
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        text(24, 30, f"{title} ({unit})", 20, "700"),
    ]
    for index, (label, value) in enumerate(values):
        y = margin_top + index * row_height
        bar_width = chart_width * value / max_value
        color = color_for(label)
        svg.append(text(24, y + 18, label, 12))
        svg.append(
            f'<rect x="{margin_left}" y="{y}" width="{bar_width:.2f}" '
            f'height="20" rx="3" fill="{color}" />'
        )
        svg.append(text(margin_left + bar_width + 8, y + 16, f"{value:.2f}", 12))
    svg.append("</svg>\n")
    path.write_text("\n".join(svg), encoding="utf-8")


def color_for(label: str) -> str:
    if "power-fast" in label:
        return "#db4437"
    if "temperature" in label:
        return "#f4b400"
    return "#109d58"


def text(x: float, y: float, value: str, size: int, weight: str = "400") -> str:
    escaped = (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="#222">{escaped}</text>'
    )


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
