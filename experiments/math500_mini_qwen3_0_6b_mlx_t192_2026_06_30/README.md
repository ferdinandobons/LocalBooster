# MATH500 Mini Qwen3 0.6B MLX T192

This folder contains a tracked LocalBooster experiment.

## Scope

- backend: `mlx`
- model: `mlx-community/Qwen3-0.6B-4bit`
- records: `18` JSONL rows
- purpose: quality, runtime, and cost benchmark
- note: runs used short 192-token completions, so answers may be incomplete

## Summary

| Sampler | Runs | Accuracy | Avg Latency | Avg Cost x | Avg Acceptance | Avg Generated | Avg Sampled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `standard` | 6 | 0.00 | 4.99s | 1.00 | n/a | 192.0 | 192.0 |
| `temperature` | 6 | 0.00 | 4.50s | 1.00 | n/a | 192.0 | 192.0 |
| `power-fast` | 6 | 0.00 | 33.04s | 5.55 | 0.58 | 192.0 | 1065.8 |

## Files

- `data/results.jsonl`: raw LocalBooster output rows
- `data/summary.csv`: tabular aggregate metrics
- `data/summary.json`: aggregate metrics as JSON
- `plots/latency_seconds.svg`: average latency by sampler
- `plots/cost_multiplier.svg`: average sampled-token cost multiplier
- `plots/acceptance_ratio.svg`: MCMC acceptance ratio where applicable
- `plots/accuracy.svg`: benchmark accuracy, when scored answers are present
- `plots/tokens.svg`: generated vs sampled token counts

## Charts

![Average latency](plots/latency_seconds.svg)

![Cost multiplier](plots/cost_multiplier.svg)

![Acceptance ratio](plots/acceptance_ratio.svg)

![Benchmark accuracy](plots/accuracy.svg)

![Token counts](plots/tokens.svg)
