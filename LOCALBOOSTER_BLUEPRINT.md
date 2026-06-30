# LocalBooster Blueprint

## Source Audit

Local input:

- Paper: `2510.14901v1.pdf`
- Official code snapshot: `reasoning-with-sampling-main/`
- Paper title: "Reasoning with Sampling: Your Base Model is Smarter Than You Think"
- Official repository: `https://github.com/aakaran/reasoning-with-sampling`

The official repository is research-oriented. It contains benchmark scripts for MATH500, HumanEval, GPQA, and AlpacaEval, plus one core utility module for power sampling. It is not structured as a reusable Python package, CLI, server, or local-model product.

No `LICENSE` file is present in the local snapshot. Treat the official code as reference material only unless an explicit upstream license is added later. LocalBooster should reimplement the algorithm from the paper, cite the paper and repository, and avoid copying upstream source.

## Product Thesis

LocalBooster is an inference-time reasoning booster for local open models.

It does not fine-tune models and does not create new model weights. Instead, it spends extra local compute during generation to sample better reasoning traces from the base model. The core user-facing promise is:

> Make a local model think harder when the task is worth the latency.

This is stronger and cleaner than a generic "make models better" claim. The project should be explicit that it improves a class of reasoning-heavy tasks by changing the decoding process, not the underlying model.

## Differentiation From The Official Repo

The official repository answers:

- Can the paper's method reproduce benchmark improvements?
- How does power sampling compare with standard sampling, low-temperature sampling, and GRPO?

LocalBooster should answer:

- Can a normal user apply this to a local model from a terminal or API?
- What latency-quality tradeoff should they pick?
- Which prompts benefit enough to justify extra compute?
- How can developers plug this into local coding, planning, and reasoning workflows?

## Initial Scope

The first open source version should be narrow:

- Python package: `localbooster`
- CLI: `localbooster generate`
- Hugging Face Transformers backend first
- One sampler: power sampling via autoregressive MCMC
- One baseline: normal sampled generation
- One comparison mode: low-temperature generation
- Local benchmark runner for a small JSONL prompt set
- Metrics: latency, generated token count, MCMC acceptance ratio, approximate compute multiplier

Do not start with a GUI, agent framework, model manager, or fine-tuning pipeline.

## MVP User Experience

Example target commands:

```bash
localbooster generate \
  --model Qwen/Qwen2.5-7B \
  --prompt "Solve this step by step: ..." \
  --mode balanced
```

```bash
localbooster compare \
  --model Qwen/Qwen2.5-7B \
  --prompt-file examples/prompts/math.txt
```

```bash
localbooster bench \
  --model Qwen/Qwen2.5-7B \
  --dataset examples/benchmarks/reasoning_small.jsonl \
  --mode fast
```

Preset modes:

- `fast`: small MCMC budget, useful for interactive use
- `balanced`: default quality-latency tradeoff
- `deep`: high-compute mode for difficult reasoning or coding

## Architecture

Suggested package layout:

```text
localbooster/
  __init__.py
  cli.py
  config.py
  generation.py
  metrics.py
  sampling/
    __init__.py
    base.py
    power.py
    temperature.py
  backends/
    __init__.py
    transformers_backend.py
  benchmarks/
    __init__.py
    runner.py
tests/
docs/
examples/
pyproject.toml
README.md
```

Core interfaces:

- `LanguageModelBackend`: tokenization, detokenization, next-token logits, sampled continuation
- `Sampler`: generates tokens and returns both text and metrics
- `PowerSampler`: implements autoregressive MCMC power sampling
- `GenerationMetrics`: latency, tokens, acceptance ratio, mode, seed, parameters

This separation keeps the paper algorithm independent from a specific runtime. Transformers can be implemented first; later backends could include llama.cpp, MLX, vLLM, or Ollama if the APIs expose enough log-probability information.

## Algorithm Requirements

Power sampling needs more than ordinary text generation. A backend must expose:

- token IDs for a prompt
- generated token IDs
- per-token log-probabilities under the proposal distribution
- per-token log-probabilities under the base model target distribution
- the ability to resample from an arbitrary token prefix

This requirement is why Transformers is the best first backend. Some local runtimes may not expose enough likelihood information for faithful power sampling.

## Default Parameters

Paper-grounded starting point:

- `alpha = 4.0`
- proposal temperature `temperature = 1 / alpha = 0.25`
- `block_num = 16`
- `mcmc_steps = 10`

Product defaults should be gentler for local use:

- `fast`: `alpha=2.0`, `mcmc_steps=2`, `block_num=8`
- `balanced`: `alpha=4.0`, `mcmc_steps=5`, `block_num=12`
- `deep`: `alpha=4.0`, `mcmc_steps=10`, `block_num=16`

These should be treated as initial presets, not proven final defaults.

## Practical Use Cases

Best first use cases:

- math reasoning
- code generation and debugging
- technical planning
- multi-step local analysis
- privacy-sensitive local assistance where cloud model calls are not acceptable

Weak first use cases:

- casual chat
- summarization where low latency matters
- creative writing where diversity may be more important than sharpened likelihood
- very long generation on small machines

## Roadmap

Phase 1: Reimplementation

- Create package skeleton.
- Implement Transformers backend.
- Implement normal sampling and low-temperature sampling.
- Implement power sampling from the paper.
- Add deterministic seeds and metrics.
- Add unit tests for acceptance-ratio math using fake backends.

Phase 2: Usability

- Add CLI commands.
- Add presets.
- Add benchmark JSONL format.
- Add README examples and caveats.
- Add small smoke tests that run with a tiny model.

Phase 3: Proof

- Reproduce a small benchmark slice locally.
- Compare standard sampling, low-temperature sampling, and power sampling.
- Publish results with latency and cost multipliers, not only accuracy.

Phase 4: Integrations

- OpenAI-compatible local API wrapper.
- Optional llama.cpp or MLX backend if log-probability access is adequate.
- Editor/agent integrations.

## Main Risks

- Compute cost: the method can be several times slower than standard generation.
- Backend compatibility: faithful implementation requires log-probability access.
- Overclaiming: improvements are task-dependent and should be framed around reasoning-heavy prompts.
- Licensing: upstream snapshot has no visible license, so copying code would be unsafe.
- Evaluation: a polished project needs honest local benchmarks, including cases where power sampling does not help.

## Recommended Positioning

Use this framing:

> LocalBooster is a training-free inference-time booster for local language models, based on power sampling. It lets users trade latency for stronger reasoning without fine-tuning or sending prompts to a cloud API.

Avoid this framing:

> LocalBooster makes any model better.

The second claim is too broad and will be easy to disprove.

