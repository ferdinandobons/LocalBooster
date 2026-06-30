"""Command line interface for LocalBooster."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from localbooster.backends.errors import OptionalDependencyError
from localbooster.backends.registry import load_backend
from localbooster.config import GenerationConfig, get_preset
from localbooster.generation import build_sampler
from localbooster.metrics import GenerationResult


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except OptionalDependencyError as exc:
        print(f"localbooster: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"localbooster: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localbooster")
    parser.add_argument("--version", action="version", version="localbooster 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate a single response.")
    _add_generation_args(generate)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    generate.set_defaults(func=cmd_generate)

    compare = subparsers.add_parser("compare", help="Run multiple samplers for one prompt.")
    _add_generation_args(compare)
    compare.add_argument("--prompt", required=True)
    compare.add_argument(
        "--samplers",
        default="standard,temperature,power",
        help="Comma-separated sampler list.",
    )
    compare.set_defaults(func=cmd_compare)

    bench = subparsers.add_parser("bench", help="Run samplers over a JSONL prompt dataset.")
    _add_generation_args(bench)
    bench.add_argument("--dataset", required=True, type=Path)
    bench.add_argument("--out", required=True, type=Path)
    bench.add_argument("--limit", type=int, default=None)
    bench.add_argument(
        "--samplers",
        default="standard,temperature,power-fast",
        help="Comma-separated sampler list.",
    )
    bench.set_defaults(func=cmd_bench)

    report = subparsers.add_parser("report", help="Summarize LocalBooster JSONL results.")
    report.add_argument("paths", nargs="+", type=Path)
    report.set_defaults(func=cmd_report)

    return parser


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=["transformers", "mlx"], default="transformers")
    parser.add_argument("--sampler", choices=["standard", "temperature", "power"], default="standard")
    parser.add_argument("--preset", choices=["fast", "balanced", "deep"], default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--mcmc-steps", type=int, default=2)
    parser.add_argument("--block-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="Transformers device override, e.g. mps/cpu.")


def cmd_generate(args: argparse.Namespace) -> int:
    result = _run_one(args, args.prompt, args.sampler)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False))
    else:
        print(result.text)
        _print_metrics(result)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    backend = _load_backend_from_args(args)
    for sampler_name in _parse_csv(args.samplers):
        result = _run_with_backend(args, backend, args.prompt, sampler_name)
        print(f"\n## {sampler_name}")
        print(result.text)
        _print_metrics(result)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = list(_read_jsonl(args.dataset))
    if args.limit is not None:
        records = records[: args.limit]
    samplers = _parse_csv(args.samplers)
    backend = _load_backend_from_args(args)

    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            prompt = record["prompt"]
            for sampler_name in samplers:
                result = _run_with_backend(args, backend, prompt, sampler_name)
                output = {
                    "id": record.get("id"),
                    "prompt": prompt,
                    "expected": record.get("answer"),
                    "sampler": sampler_name,
                    "result": result.to_dict(),
                }
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                handle.flush()
    print(f"wrote {args.out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    rows = []
    for path in args.paths:
        for record in _read_jsonl(path):
            if "result" not in record or "metrics" not in record["result"]:
                raise ValueError(
                    f"{path} is not a LocalBooster result file; expected records with "
                    "`result.metrics`"
                )
            metrics = record["result"]["metrics"]
            rows.append(metrics)
    if not rows:
        print("No rows.")
        return 0
    print("| Backend | Model | Sampler | Runs | Avg latency | Avg cost x | Avg accept |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for key, group in _group_rows(rows).items():
        backend, model, sampler = key
        latency = _avg(row["latency_seconds"] for row in group)
        cost = _avg(row["cost_multiplier"] for row in group if row["cost_multiplier"] is not None)
        accept = _avg(
            row["acceptance_ratio"] for row in group if row["acceptance_ratio"] is not None
        )
        print(
            f"| {backend} | {model} | {sampler} | {len(group)} | "
            f"{latency:.2f}s | {_fmt(cost)} | {_fmt(accept)} |"
        )
    return 0


def _run_one(args: argparse.Namespace, prompt: str, sampler_name: str) -> GenerationResult:
    backend = _load_backend_from_args(args)
    return _run_with_backend(args, backend, prompt, sampler_name)


def _run_with_backend(
    args: argparse.Namespace,
    backend,
    prompt: str,
    sampler_name: str,
) -> GenerationResult:
    config = _config_from_args(args, sampler_name)
    sampler = build_sampler(backend, config)
    return sampler.generate(prompt)


def _load_backend_from_args(args: argparse.Namespace):
    backend_kwargs = {}
    if args.backend == "transformers" and args.device is not None:
        backend_kwargs["device"] = args.device
    return load_backend(args.backend, args.model, **backend_kwargs)


def _config_from_args(args: argparse.Namespace, sampler_name: str) -> GenerationConfig:
    if sampler_name.startswith("power-"):
        preset_name = sampler_name.split("-", 1)[1]
        config = get_preset(
            preset_name,
            backend=args.backend,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
    elif args.preset is not None:
        config = get_preset(
            args.preset,
            backend=args.backend,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
    else:
        sampler = "power" if sampler_name == "power" else sampler_name
        config = GenerationConfig(
            sampler=sampler,
            backend=args.backend,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            alpha=args.alpha,
            mcmc_steps=args.mcmc_steps,
            block_count=args.block_count,
            seed=args.seed,
        )
    if sampler_name == "standard":
        return replace(config, sampler="standard", temperature=args.temperature)
    if sampler_name == "temperature":
        return replace(config, sampler="temperature", temperature=1.0 / config.alpha)
    if sampler_name == "power" or sampler_name.startswith("power-"):
        return replace(config, sampler="power")
    raise ValueError(f"unknown sampler {sampler_name!r}")


def _print_metrics(result: GenerationResult) -> None:
    metrics = result.metrics
    print("\n--- metrics ---", file=sys.stderr)
    print(f"backend: {metrics.backend}", file=sys.stderr)
    print(f"model: {metrics.model}", file=sys.stderr)
    print(f"sampler: {metrics.sampler}", file=sys.stderr)
    print(f"latency: {metrics.latency_seconds:.2f}s", file=sys.stderr)
    print(f"generated_tokens: {metrics.generated_tokens}", file=sys.stderr)
    print(f"sampled_tokens: {metrics.sampled_tokens}", file=sys.stderr)
    if metrics.cost_multiplier is not None:
        print(f"cost_multiplier: {metrics.cost_multiplier:.2f}x", file=sys.stderr)
    if metrics.acceptance_ratio is not None:
        print(f"acceptance_ratio: {metrics.acceptance_ratio:.2%}", file=sys.stderr)


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "prompt" not in record and "result" not in record:
                raise ValueError(f"{path}:{line_no}: expected a `prompt` field")
            yield record


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _group_rows(rows: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["backend"], row["model"], row["sampler"])
        groups.setdefault(key, []).append(row)
    return groups


def _avg(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _fmt(value: float) -> str:
    if value != value:
        return "n/a"
    return f"{value:.2f}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
