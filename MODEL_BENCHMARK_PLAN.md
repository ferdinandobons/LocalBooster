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

## Target Hardware Constraint

The first real target machine is an Apple Silicon MacBook M2 with 8 GB of unified memory.

This changes the model plan:

- 0.5B-2B models are the default target.
- 3B-4B quantized models are stress targets after the pipeline works.
- 7B models are not MVP defaults on this hardware.
- 14B+ and 24B+ models are remote or external-reference models only.

Power sampling increases generation work. Even if model weights fit in memory, repeated proposal generation and KV cache pressure can make a model impractical. The benchmark suite must therefore optimize for small, repeatable comparisons before chasing leaderboard-scale models.

## Model Requirements

A model is a good first target if it is:

- decoder-only / causal language model
- supported by Hugging Face Transformers
- exposes logits / log-probabilities
- can regenerate from arbitrary prefixes
- runnable locally at 4-bit, 8-bit, BF16, or FP16
- licensed clearly enough for open source users

Power sampling is easiest to implement faithfully with Transformers first. Runtimes such as Ollama, llama.cpp, MLX, vLLM, or SGLang can be added later only if they expose the token likelihoods needed by the sampler.

MLX-LM deserves special treatment for the target Mac. It is not just another runtime: it is the Mac-native path that can make the project usable on Apple Silicon. The benchmark plan should therefore include a Transformers lane for correctness and an MLX lane for practical local performance.

## Recommended First Models For M2 8 GB

### Tier 1: Local-Safe MVP Models

Use these first because they are realistic on an 8 GB Mac and still useful for testing whether inference-time boosting helps.

| Role | Model | Why it matters |
| --- | --- | --- |
| Small baseline | `Qwen/Qwen3-0.6B` | Very small, modern, and useful for smoke tests, latency tests, and CI-like local checks. |
| Main local target | `Qwen/Qwen3-1.7B` | Best first balance of feasibility and capability on 8 GB hardware. |
| Reasoning-tuned comparison | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Similar size to Qwen3-1.7B, but tuned for reasoning; useful to compare base/instruct boosting against a reasoning model. |
| Coding target | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Small coding-specialized model for HumanEval/MBPP/LiveCodeBench slices. |

### Tier 2: Local Stress Models

Use these only after the MVP path works on 0.5B-2B models. Prefer 4-bit quantized variants where available.

| Role | Model | Why it matters |
| --- | --- | --- |
| Larger Qwen local stress | `Qwen/Qwen3-4B` | May run quantized on an 8 GB Mac, but power sampling may be slow or memory-constrained. |
| Small general instruct | `microsoft/Phi-4-mini-instruct` | 3.8B model; useful but should be tested after smaller models. |
| Coding stress | `Qwen/Qwen2.5-Coder-3B-Instruct` | Better coding target than 1.5B, but heavier on the target machine. |
| Popular small model | `meta-llama/Llama-3.2-3B-Instruct` | Useful ecosystem comparison, but custom-licensed and not the first default. |

### Tier 3: External References Only

These are useful for comparison reports but should not be expected to run locally on the target 8 GB Mac.

| Model | Caveat |
| --- | --- |
| `Qwen/Qwen3-8B` | Too heavy for default LocalBooster testing on 8 GB, but useful on a 16 GB+ machine. |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Strong reasoning reference, but not a safe default for 8 GB with MCMC. |
| `Qwen/Qwen3-14B` | External reference only. |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | External reference only. |
| `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | External reference only. |

## Base Vs Instruct Models

The paper is strongest conceptually on base models because it asks whether the base distribution already contains stronger reasoning paths.

For the product, support both:

- base models: cleaner scientific comparison, less useful for normal chat
- instruct models: more useful to users, but sampling improvements may interact with instruction tuning
- reasoning models: useful as competitors, but LocalBooster may have less room to improve them

The first benchmark matrix should include both a base and an instruct model from the same family when possible.

Example:

- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3-1.7B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- `Qwen/Qwen2.5-Coder-1.5B-Instruct`

## Benchmark Suite

### MVP Benchmark Set

Use a small, reproducible suite before trying to chase every leaderboard.

| Capability | Benchmark | Why |
| --- | --- | --- |
| Grade-school math | GSM8K small split | Cheapest useful sanity check. |
| Math reasoning | MATH-500 small split, then full MATH-500 | Directly aligned with the paper, but full evaluation can be slow. |
| Coding | MBPP small split, then HumanEval | Better first fit for small coding models and local repeated runs. |
| Modern coding | LiveCodeBench small slice | Useful later, but expensive as an MVP default. |
| Scientific reasoning | GPQA Diamond small split | Keep as later validation; small models may be weak and runs are slower. |
| General knowledge/reasoning | MMLU-Pro small subset | Later comparison only, not initial smoke testing. |

### First Local Benchmark Result

The first scored benchmark run used the local MLX model
`mlx-community/Qwen3-0.6B-4bit` on a 6-example MATH500 mini slice with 192 generated tokens.
The three-way comparison matched the official research framing: standard sampling,
low-temperature sampling, and MCMC/power sampling.

| Model | Sampler | Accuracy | Avg Latency | Cost x | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: |
| `mlx-community/Qwen3-0.6B-4bit` | `standard` | 0.00 | 4.99s | 1.00 | n/a |
| `mlx-community/Qwen3-0.6B-4bit` | `temperature` | 0.00 | 4.50s | 1.00 | n/a |
| `mlx-community/Qwen3-0.6B-4bit` | `power-fast` | 0.00 | 33.04s | 5.55 | 0.58 |

Interpretation:

- MATH500 is too hard for this 0.6B 4-bit model at this token budget.
- `power-fast` did not improve accuracy on this slice, while adding meaningful latency.
- This is still a useful negative result: the benchmark harness now reports quality and cost
  together, and it shows why the next serious run should move to `Qwen3-1.7B-4bit` and/or a
  graduated suite such as GSM8K before relying on MATH500.

### First GSM8K Model Compare

The first cleaner model-vs-model run used the same three GSM8K test examples across
`mlx-community/Qwen3-0.6B-4bit` and `mlx-community/Qwen3-1.7B-4bit`, with 128 generated
tokens and the same sampler set.

| Model | Sampler | Accuracy | Avg Latency | Cost x | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: |
| `Qwen3-0.6B-4bit` | `standard` | 0.00 | 3.35s | 1.00 | n/a |
| `Qwen3-0.6B-4bit` | `temperature` | 0.00 | 3.65s | 1.00 | n/a |
| `Qwen3-0.6B-4bit` | `power-fast` | 0.00 | 24.26s | 5.32 | 0.75 |
| `Qwen3-1.7B-4bit` | `standard` | 0.33 | 3.41s | 1.00 | n/a |
| `Qwen3-1.7B-4bit` | `temperature` | 0.33 | 3.69s | 1.00 | n/a |
| `Qwen3-1.7B-4bit` | `power-fast` | 0.00 | 42.26s | 5.19 | 0.58 |

Interpretation:

- The larger 1.7B model is already better than 0.6B under normal decoding on this tiny slice.
- `power-fast` is not yet beneficial on this GSM8K subset; it adds 5x+ sampled-token cost and
  lowers observed accuracy for the 1.7B run.
- The likely next algorithmic work is sampler quality, prompt discipline, and candidate
  selection, not merely trying larger models.

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

## Local Run Order

The local test sequence should be conservative:

1. Smoke test `Qwen/Qwen3-0.6B` with standard generation.
2. Smoke test `Qwen/Qwen3-0.6B` with `power-fast`.
3. Repeat with `Qwen/Qwen3-1.7B`.
4. Run a 10-example GSM8K slice across standard, low-temp, and `power-fast`.
5. Run a 10-example MBPP slice for the coding path.
6. Only then try `power-balanced`.
7. Only after stable 1.7B runs, try 3B-4B models.

Abort or downgrade a model if:

- the OS starts swapping heavily
- time per answer is unusable for interactive work
- the sampler acceptance ratio is near zero for most prompts
- benchmark runs cannot complete repeatably

For this hardware, a useful result is not "largest model runs once". A useful result is "small model runs repeatably with honest quality and latency data".

## Runtime Implications

The cleanest first implementation is Hugging Face Transformers because it exposes logits and token-level likelihoods in a familiar cross-platform way. On Apple Silicon with 8 GB, this may be slower or heavier than native MLX runtimes.

MLX-LM should be tested early for two reasons:

- it is optimized for Apple Silicon and should be more realistic on an M2 Mac
- its generation utilities expose token-level log-probability data, which is the key requirement for faithful power sampling
- LocalBooster now uses MLX-LM's `generate_step` API so each sampled continuation benefits from KV cache after the prompt prefill

The practical path is:

- implement the algorithm first with Transformers for correctness
- keep prompts, benchmark format, and reports backend-independent
- add an MLX backend as the first Mac-native backend
- allow standard and low-temperature baselines to run on simpler local runtimes sooner
- treat cross-proposal prompt-cache reuse as a later optimization after the `Qwen3-0.6B` and `Qwen3-1.7B` smoke paths are stable

## MLX Benchmark Lane

The MLX lane should answer a different question than the Transformers lane:

> On the actual 8 GB Mac, which small quantized model gives the best quality-latency tradeoff with and without LocalBooster?

Suggested MLX test matrix:

| Model family | MLX target | Standard | Low-temp | Power-fast | Power-balanced |
| --- | --- | --- | --- | --- | --- |
| Qwen small | `mlx-community` Qwen3 0.6B 4-bit variant | yes | yes | yes | optional |
| Qwen main | `mlx-community` Qwen3 1.7B 4-bit variant | yes | yes | yes | yes |
| DeepSeek reasoning | `mlx-community` DeepSeek-R1-Distill-Qwen-1.5B 4-bit variant | yes | yes | optional | optional |
| Qwen Coder | `mlx-community` Qwen2.5-Coder 1.5B Instruct 4-bit variant | yes | yes | yes | optional |

The first implementation can use model IDs as user-provided strings instead of hard-coding exact `mlx-community` repository names, because quantized model names change and multiple quantization variants may exist.

Report backend separately:

- `backend=transformers`
- `backend=mlx`

Do not compare a full-precision Transformers run against a 4-bit MLX run without labeling precision and quantization clearly.

## Suggested First Benchmark Matrix

Start with the models that should actually fit the target machine:

| Model | Standard | Low-temp | LocalBooster fast | LocalBooster balanced |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3-0.6B` | yes | yes | yes | yes |
| `Qwen/Qwen3-1.7B` | yes | yes | yes | yes |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | yes | yes | optional | optional |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | yes | yes | yes | yes |

Stress-only:

| Model | Standard | Low-temp | LocalBooster fast | LocalBooster balanced |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3-4B` | yes | yes | optional | optional |
| `microsoft/Phi-4-mini-instruct` | yes | yes | optional | optional |
| `Qwen/Qwen2.5-Coder-3B-Instruct` | yes | yes | optional | optional |

Reference-only:

| Model | Standard only |
| --- | --- |
| `Qwen/Qwen3-8B` | yes |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | yes |
| `Qwen/Qwen3-14B` | yes |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | yes |
| `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | yes |

This keeps the first research question clean:

> Does LocalBooster make 0.6B-1.7B local models meaningfully better, and can a boosted 1.7B model approach a 3B-4B standard model on selected tasks?

## Recommended Open Source MVP

Build `localbooster bench` around this workflow:

```bash
localbooster bench \
  --model Qwen/Qwen3-1.7B \
  --tasks math500,gsm8k,humaneval \
  --samplers standard,temperature,power-fast,power-balanced \
  --out results/qwen3-1.7b.jsonl
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
