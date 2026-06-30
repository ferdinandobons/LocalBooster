# Experiments

This folder stores tracked LocalBooster experiment artifacts.

Each experiment should include:

- raw LocalBooster JSONL output
- aggregate CSV/JSON summaries
- SVG plots that render on GitHub
- a short README describing scope, hardware path, model, and whether the run is a smoke test or an accuracy benchmark

Current experiments:

- [Qwen3 0.6B MLX smoke experiment, 12 tokens](qwen3_0_6b_mlx_smoke_2026_06_30/README.md)
- [Qwen3 0.6B MLX smoke experiment, 32 tokens](qwen3_0_6b_mlx_smoke_t32_2026_06_30/README.md)
- [Qwen3 0.6B MLX smoke experiment, 64 tokens](qwen3_0_6b_mlx_smoke_t64_2026_06_30/README.md)
- [Qwen3 0.6B MLX smoke scaling, 12 vs 32 vs 64 tokens](qwen3_0_6b_mlx_smoke_scaling_2026_06_30/README.md)
- [MATH500 mini Qwen3 0.6B MLX benchmark, 192 tokens](math500_mini_qwen3_0_6b_mlx_t192_2026_06_30/README.md)
- [GSM8K mini clean3 Qwen3 0.6B MLX benchmark, 128 tokens](gsm8k_mini_clean3_qwen3_0_6b_mlx_t128_2026_06_30/README.md)
- [GSM8K mini clean3 Qwen3 1.7B MLX benchmark, 128 tokens](gsm8k_mini_clean3_qwen3_1_7b_mlx_t128_2026_06_30/README.md)
- [GSM8K mini clean3 model comparison](gsm8k_mini_clean3_model_compare_2026_06_30/README.md)
