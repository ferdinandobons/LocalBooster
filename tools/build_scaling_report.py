#!/usr/bin/env python3
"""Build a scaling report across multiple LocalBooster experiment summaries."""

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
        path = Path(path_str)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["summaries"]:
            row = dict(row)
            row["token_budget"] = int(label)
            rows.append(row)
    if not rows:
        parser.error("all experiment summaries were empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    data_dir = args.output_dir / "data"
    plots_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    write_csv(data_dir / "scaling_summary.csv", rows)
    (data_dir / "scaling_summary.json").write_text(
        json.dumps({"rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_line_chart(
        plots_dir / "latency_scaling.svg",
        "Latency Scaling",
        rows,
        "avg_latency_seconds",
        "seconds",
    )
    write_line_chart(
        plots_dir / "cost_scaling.svg",
        "Cost Multiplier Scaling",
        rows,
        "avg_cost_multiplier",
        "x",
    )
    write_line_chart(
        plots_dir / "acceptance_scaling.svg",
        "Acceptance Ratio Scaling",
        rows,
        "avg_acceptance_ratio",
        "ratio",
    )
    write_readme(args.output_dir / "README.md", rows)
    print(f"wrote {args.output_dir}")
    return 0


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["token_budget"] + [key for key in rows[0] if key != "token_budget"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, rows: list[dict]) -> None:
    sampler_rank = {
        sampler: index for index, sampler in enumerate(ordered_samplers(rows))
    }
    lines = [
        "# Qwen3 0.6B MLX Smoke Scaling",
        "",
        "This folder compares the tracked smoke runs at 12, 32, and 64 generated tokens.",
        "",
        "These are runtime/cost smoke tests, not accuracy benchmarks.",
        "",
        "## Summary",
        "",
        "| Tokens | Sampler | Runs | Avg Latency | Avg Cost x | Avg Acceptance |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        rows,
        key=lambda item: (item["token_budget"], sampler_rank.get(item["sampler"], 999)),
    ):
        lines.append(
            f"| {row['token_budget']} | `{row['sampler']}` | {row['runs']} | "
            f"{row['avg_latency_seconds']:.2f}s | {fmt(row['avg_cost_multiplier'])} | "
            f"{fmt(row['avg_acceptance_ratio'])} |"
        )
    lines.extend(
        [
            "",
            "## Charts",
            "",
            "![Latency scaling](plots/latency_scaling.svg)",
            "",
            "![Cost scaling](plots/cost_scaling.svg)",
            "",
            "![Acceptance scaling](plots/acceptance_scaling.svg)",
            "",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_line_chart(
    path: Path, title: str, rows: list[dict], metric: str, unit: str
) -> None:
    width = 760
    height = 360
    left = 72
    top = 48
    plot_width = 560
    plot_height = 230
    token_values = sorted({row["token_budget"] for row in rows})
    metric_values = [row[metric] for row in rows if row[metric] is not None]
    max_value = max(metric_values) if metric_values else 1.0
    max_value = max(max_value, 1e-9)
    samplers = ordered_samplers(rows)
    colors = {
        "standard": "#109d58",
        "temperature": "#f4b400",
        "power-fast": "#db4437",
    }

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff" />',
        text(24, 30, title, 20, "700"),
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#777" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top + plot_height}" stroke="#777" />',
    ]
    for token in token_values:
        x = left + scale(token, min(token_values), max(token_values), plot_width)
        svg.append(
            f'<line x1="{x}" y1="{top + plot_height}" x2="{x}" '
            f'y2="{top + plot_height + 5}" stroke="#777" />'
        )
        svg.append(text(x, top + plot_height + 24, str(token), 12, anchor="middle"))
    svg.append(
        text(
            left + plot_width / 2,
            height - 16,
            "generated token budget",
            12,
            anchor="middle",
        )
    )
    svg.append(text(16, top + 10, unit, 12))

    by_sampler = {
        sampler: sorted(
            [row for row in rows if row["sampler"] == sampler and row[metric] is not None],
            key=lambda row: row["token_budget"],
        )
        for sampler in samplers
    }
    for legend_index, sampler in enumerate(samplers):
        color = colors.get(sampler, "#3366cc")
        points = []
        for row in by_sampler[sampler]:
            x = left + scale(
                row["token_budget"], min(token_values), max(token_values), plot_width
            )
            y = top + plot_height - (row[metric] / max_value * plot_height)
            points.append((x, y, row[metric]))
        if len(points) > 1:
            path_data = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in points)
            svg.append(
                f'<polyline points="{path_data}" fill="none" '
                f'stroke="{color}" stroke-width="3" />'
            )
        for x, y, value in points:
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" />')
            svg.append(text(x + 8, y - 8, f"{value:.2f}", 11))
        legend_y = 58 + legend_index * 22
        svg.append(f'<rect x="650" y="{legend_y - 11}" width="12" height="12" fill="{color}" />')
        svg.append(text(668, legend_y, sampler, 12))
    svg.append("</svg>\n")
    path.write_text("\n".join(svg), encoding="utf-8")


def ordered_samplers(rows: list[dict]) -> list[str]:
    preferred = ["standard", "temperature", "power-fast"]
    available = {row["sampler"] for row in rows}
    return [sampler for sampler in preferred if sampler in available] + sorted(
        available - set(preferred)
    )


def scale(value: float, min_value: float, max_value: float, width: float) -> float:
    if max_value == min_value:
        return width / 2
    return (value - min_value) / (max_value - min_value) * width


def text(
    x: float,
    y: float,
    value: str,
    size: int,
    weight: str = "400",
    anchor: str = "start",
) -> str:
    escaped = (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="#222">{escaped}</text>'
    )


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
