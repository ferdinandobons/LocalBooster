# MLX Backend Notes

Status date: 2026-06-30

## Scope

The first real LocalBooster model tested on the target Mac path is:

- `mlx-community/Qwen3-0.6B-4bit`

This model is intentionally small. It is the smoke-test target for a MacBook M2 with 8 GB unified memory before moving to `Qwen3-1.7B` or any 3B-4B stress target.

## Backend Implementation

The initial MLX backend used direct model calls for every generated token. That worked, but it recomputed the whole prefix repeatedly and was too slow for useful benchmarking.

The backend now uses MLX-LM's `generate_step` API:

- MLX-LM pre-fills the prompt once per sampled continuation.
- Subsequent tokens in that continuation use MLX-LM's KV cache.
- LocalBooster still reads the returned base log-probability vector for each generated token.
- LocalBooster computes:
  - proposal log-probability from `softmax(base_logprobs / temperature)`
  - target log-probability as `alpha * base_logprob[token]`

This keeps the Metropolis-Hastings ratio usable while avoiding full-prefix recomputation for every token in a continuation.

## Remaining Limitation

The current backend caches within each continuation, not across all MCMC proposals.

Power sampling still asks the backend to resample many suffixes from different prefixes. Each suffix proposal gets a fresh MLX-LM cache for that prefix. A deeper optimization would reuse or copy prompt caches across proposals and trim them when resampling suffixes. That is possible in principle because MLX-LM exposes prompt-cache utilities, but it is a second optimization step and should be benchmark-driven.

## First Measurements

Hardware path:

- backend: `mlx`
- model: `mlx-community/Qwen3-0.6B-4bit`
- environment: local `.venv`
- prompt: `Solve step by step: 18 * 7 - 9`

Direct MLX-LM smoke:

```text
Prompt: 23 tokens, 22.041 tokens-per-sec
Generation: 64 tokens, 138.170 tokens-per-sec
Peak memory: 0.456 GB
```

LocalBooster before cache optimization:

```text
standard, 64 tokens: 19.37s
standard, 12 tokens: 1.13s
temperature, 12 tokens: 1.11s
power-fast, 12 tokens: 6.45s, cost 5.42x, acceptance 75%
```

LocalBooster after MLX-LM `generate_step` cache integration:

```text
standard, 64 tokens: 1.35s
standard, 12 tokens: 0.35s
temperature, 12 tokens: 0.94s
power-fast, 12 tokens: 1.72s, cost 4.00x, acceptance 91.67%
```

JSONL smoke benchmark, one reasoning prompt, after cache integration:

```text
standard:     0.22s, cost 1.00x
temperature:  0.17s, cost 1.00x
power-fast:   1.47s, cost 3.25x, acceptance 100%
```

JSONL smoke benchmark, all 3 reasoning prompts, after cache integration:

```text
standard:     0.23s avg, cost 1.00x
temperature:  0.21s avg, cost 1.00x
power-fast:   1.71s avg, cost 5.03x, acceptance 72%
```

These are runtime smoke tests, not accuracy benchmarks. The runs used only 12 generated tokens,
so many answers are incomplete or off-task. The next quality-oriented run should increase
`max-new-tokens` to at least 64 and evaluate final-answer extraction separately.

## Commands Used

Direct MLX-LM smoke:

```bash
.venv/bin/mlx_lm.generate \
  --model mlx-community/Qwen3-0.6B-4bit \
  --prompt "Solve step by step: 18 * 7 - 9" \
  --max-tokens 64
```

LocalBooster single generation:

```bash
.venv/bin/localbooster generate \
  --backend mlx \
  --model mlx-community/Qwen3-0.6B-4bit \
  --prompt "Solve step by step: 18 * 7 - 9" \
  --max-new-tokens 64 \
  --sampler standard \
  --seed 1
```

LocalBooster comparison:

```bash
.venv/bin/localbooster compare \
  --backend mlx \
  --model mlx-community/Qwen3-0.6B-4bit \
  --prompt "Solve step by step: 18 * 7 - 9" \
  --samplers standard,temperature,power-fast \
  --max-new-tokens 12 \
  --seed 2
```

LocalBooster benchmark:

```bash
.venv/bin/localbooster bench \
  --backend mlx \
  --model mlx-community/Qwen3-0.6B-4bit \
  --dataset examples/benchmarks/smoke_reasoning.jsonl \
  --samplers standard,temperature,power-fast \
  --max-new-tokens 12 \
  --limit 1 \
  --seed 3 \
  --out results/smoke-qwen3-0.6b-mlx-cached-limit1.jsonl
```

Report:

```bash
.venv/bin/localbooster report results/smoke-qwen3-0.6b-mlx-cached-limit1.jsonl
```

Full 3-prompt smoke:

```bash
.venv/bin/localbooster bench \
  --backend mlx \
  --model mlx-community/Qwen3-0.6B-4bit \
  --dataset examples/benchmarks/smoke_reasoning.jsonl \
  --samplers standard,temperature,power-fast \
  --max-new-tokens 12 \
  --seed 4 \
  --out results/smoke-qwen3-0.6b-mlx-cached-reasoning3.jsonl
```

## Next Step

Before running larger benchmark slices, use only `mlx-community/Qwen3-0.6B-4bit` and increase one variable at a time:

1. `max-new-tokens=32`
2. full `examples/benchmarks/smoke_reasoning.jsonl`
3. `examples/benchmarks/smoke_coding.jsonl`
4. then `mlx-community/Qwen3-1.7B-4bit`
