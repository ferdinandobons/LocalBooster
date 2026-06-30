# LocalBooster Model And Benchmark Plan

Status date: 2026-06-30

## Core Evaluation Idea

LocalBooster is not a new model. It is a decoding strategy applied to an existing model.

Every serious benchmark should therefore compare:

1. `model + standard sampling`
2. `model + low-temperature sampling`
3. `model + LocalBooster power sampling`
4. larger or newer reference models without LocalBooster

The key product question is:

> Can LocalBooster make a local model perform closer to a larger or newer model on reasoning-heavy tasks, while reporting the extra latency honestly?

## Model Requirements

A model is a good first target if it is:

- decoder-only / causal language model
- supported by Hugging Face Transformers
- exposes logits / log-probabilities
- can regenerate from arbitrary prefixes
- runnable locally at 4-bit, 8-bit, BF16, or FP16
- licensed clearly enough for open source users

Power sampling is easiest to implement faithfully with Transformers first. Runtimes such as Ollama, llama.cpp, MLX, vLLM, or SGLang can be added later only if they expose the token likelihoods needed by the sampler.

## Recommended First Models

### Tier 1: MVP Models

Use these first because they are practical and technically aligned with the paper.

| Role | Model | Why it matters |
| --- | --- | --- |
| Paper-faithful baseline | `Qwen/Qwen2.5-7B` | Base model, Apache 2.0, close to the paper setup, good for measuring the pure effect of sampling. |
| Modern local target | `Qwen/Qwen3-8B` | Newer Qwen model, Apache 2.0, supports thinking/non-thinking mode, strong local reasoning/coding baseline. |
| Small-machine target | `microsoft/Phi-4-mini-instruct` | 3.8B MIT-licensed model, useful for laptops and constrained machines. |
| Reasoning-model comparison | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Strong reasoning-tuned reference at similar size; helps answer whether LocalBooster can narrow the gap with distilled reasoning models. |

### Tier 2: Larger References

Use these when you want headline comparisons.

| Role | Model | Why it matters |
| --- | --- | --- |
| Better same-family target | `Qwen/Qwen3-14B` | Larger modern Qwen checkpoint; useful to ask whether 8B + LocalBooster approaches 14B. |
| Strong reasoning reference | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | Larger distilled reasoning model, still feasible on stronger local hardware. |
| Upper local reference | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | Strong dense reasoning model; expensive but useful as an aspirational comparison. |
| General strong open-weight reference | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Apache 2.0, strong reasoning/coding, local when quantized on higher-end hardware. |

### Tier 3: Popular But Not Default

These are useful for ecosystem comparison but should not be the MVP default.

| Model | Caveat |
| --- | --- |
| `meta-llama/Llama-3.1-8B-Instruct` | Very popular, but uses Meta's custom Llama license and gated Hugging Face access. |
| `meta-llama/Llama-3.1-70B-Instruct` | Strong reference, but too heavy for most local users and custom-licensed. |
| `google/gemma-3-12b-it` / `google/gemma-3-27b-it` | Useful comparison, but Gemma uses Google's Gemma license and requires accepting access terms. |

## Base Vs Instruct Models

The paper is strongest conceptually on base models because it asks whether the base distribution already contains stronger reasoning paths.

For the product, support both:

- base models: cleaner scientific comparison, less useful for normal chat
- instruct models: more useful to users, but sampling improvements may interact with instruction tuning
- reasoning models: useful as competitors, but LocalBooster may have less room to improve them

The first benchmark matrix should include both a base and an instruct model from the same family when possible.

Example:

- `Qwen/Qwen2.5-7B`
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen3-8B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`

## Benchmark Suite

### MVP Benchmark Set

Use a small, reproducible suite before trying to chase every leaderboard.

| Capability | Benchmark | Why |
| --- | --- | --- |
| Math reasoning | MATH-500 | Directly aligned with the paper and widely used by reasoning models. |
| Grade-school math | GSM8K | Cheaper and easier sanity check. |
| Scientific reasoning | GPQA Diamond | Harder reasoning test; useful but slower and noisier. |
| Coding | HumanEval or MBPP | Classic coding signal; not enough alone, but useful first pass. |
| Modern coding | LiveCodeBench | Better contamination-resistant coding benchmark for serious comparisons. |
| General knowledge/reasoning | MMLU-Pro | Better than old MMLU for stronger models, but more expensive. |

### Reporting Metrics

Every benchmark result should report:

- accuracy / pass@1 / exact match, depending on benchmark
- wall-clock latency
- generated token count
- total sampled token count, including rejected proposals
- MCMC acceptance ratio
- cost multiplier vs standard sampling
- model size and quantization
- hardware used
- seed count

Without latency and cost multiplier, the benchmark will be misleading.

## Suggested First Benchmark Matrix

Start small:

| Model | Standard | Low-temp | LocalBooster fast | LocalBooster balanced |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-7B` | yes | yes | yes | yes |
| `Qwen/Qwen3-8B` | yes | yes | yes | yes |
| `microsoft/Phi-4-mini-instruct` | yes | yes | yes | yes |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | yes | optional | optional | optional |

Reference-only:

| Model | Standard only |
| --- | --- |
| `Qwen/Qwen3-14B` | yes |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | yes |
| `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | yes |

This keeps the first research question clean:

> Does LocalBooster on 7B/8B models close part of the gap against 14B/24B reasoning-capable models?

## Recommended Open Source MVP

Build `localbooster bench` around this workflow:

```bash
localbooster bench \
  --model Qwen/Qwen3-8B \
  --tasks math500,gsm8k,humaneval \
  --samplers standard,temperature,power-fast,power-balanced \
  --out results/qwen3-8b.jsonl
```

Then build:

```bash
localbooster report results/*.jsonl --format markdown
```

The generated report should include both quality and cost:

| Model | Sampler | MATH-500 | HumanEval | Latency | Cost x |
| --- | --- | --- | --- | --- | --- |

This can become the strongest public artifact of the project: not just a library, but a transparent local benchmark runner for inference-time boosting.

## Positioning

Good claim:

> LocalBooster tests whether extra local inference compute can make smaller open models behave closer to larger reasoning models.

Avoid:

> LocalBooster makes every model better.

Avoid:

> LocalBooster beats larger models.

That may happen in some slices, but it should be earned by benchmark data.

