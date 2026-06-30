# LocalBooster

LocalBooster is an open source project for testing whether extra local inference compute can improve reasoning from open-weight language models.

The project is inspired by the paper ["Reasoning with Sampling: Your Base Model is Smarter Than You Think"](https://arxiv.org/abs/2510.14901) and its official research repository, [`aakaran/reasoning-with-sampling`](https://github.com/aakaran/reasoning-with-sampling).

## Thesis

LocalBooster is not a fine-tuning project and does not create new model weights.

The goal is to build a practical inference-time booster for local LLMs: a CLI and Python library that can compare normal sampling, low-temperature sampling, and power-sampling-style generation on reasoning-heavy tasks.

The initial hardware target is a small local machine: an Apple Silicon MacBook with 8 GB of unified memory. That means the MVP should prioritize small quantized models first and treat 7B+ models as optional stress tests, not defaults.

For Apple Silicon, LocalBooster should evaluate both Hugging Face Transformers and MLX-LM. Transformers is the most portable first backend for algorithm correctness; MLX-LM is the practical Mac-native backend to test real local usability.

## MVP

The first useful version should provide:

- a Python package named `localbooster`
- a CLI command for generation
- a Hugging Face Transformers backend
- standard, low-temperature, and power sampling modes
- benchmark runners for small reasoning and coding suites
- reports that include both quality and cost metrics

## Install

Core package only:

```bash
python3 -m pip install -e .
```

Transformers backend:

```bash
python3 -m pip install -e ".[transformers]"
```

MLX backend for Apple Silicon:

```bash
python3 -m pip install -e ".[mlx]"
```

Example target workflow:

```bash
localbooster generate \
  --model Qwen/Qwen3-1.7B \
  --prompt "Solve this step by step..." \
  --mode balanced
```

```bash
localbooster bench \
  --model Qwen/Qwen3-1.7B \
  --tasks math500,gsm8k,humaneval \
  --samplers standard,temperature,power-fast,power-balanced \
  --out results/qwen3-1.7b.jsonl
```

Small local smoke dataset:

```bash
localbooster bench \
  --backend mlx \
  --model mlx-community/Qwen3-1.7B-4bit \
  --dataset examples/benchmarks/smoke_reasoning.jsonl \
  --samplers standard,temperature,power-fast \
  --max-new-tokens 128 \
  --out results/smoke-qwen3-1.7b-mlx.jsonl
```

Summarize result costs:

```bash
localbooster report results/*.jsonl
```

## Initial Local Model Targets

For an 8 GB Mac, start with:

- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3-1.7B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- `Qwen/Qwen2.5-Coder-1.5B-Instruct`

For MLX-specific runs, prefer matching `mlx-community` quantized variants of the same models where available, especially 4-bit variants for 8 GB machines.

Use 3B-4B models only after the 1B-2B path is stable. Use 7B+ models only on stronger hardware or as slow stress tests.

## Benchmark Principle

LocalBooster should be evaluated as a decoding strategy, not as a new model.

The core comparison is:

- `model + standard sampling`
- `model + low-temperature sampling`
- `model + LocalBooster power sampling`
- larger or newer reference models without LocalBooster

Every report should include latency, token cost, acceptance ratio, hardware, quantization, and benchmark score. A quality-only benchmark would be misleading.

## Documentation

- [Project blueprint](LOCALBOOSTER_BLUEPRINT.md)
- [Model and benchmark plan](MODEL_BENCHMARK_PLAN.md)
- [MLX backend notes](docs/MLX_BACKEND_NOTES.md)
- [Experiments](experiments/README.md)

## License

LocalBooster is released under the MIT License.

The official research repository snapshot in this workspace is treated as reference material only and is not vendored into this project.
