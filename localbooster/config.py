"""Configuration models and presets for LocalBooster generation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


SamplerName = Literal["standard", "temperature", "power"]
BackendName = Literal["transformers", "mlx"]
Preset = Literal["fast", "balanced", "deep"]


@dataclass(frozen=True)
class GenerationConfig:
    """Runtime parameters shared across samplers.

    `max_new_tokens` is intentionally conservative for the 8 GB Mac target. Users can raise it,
    but the default should make smoke tests complete before memory pressure hides the signal.
    """

    sampler: SamplerName = "standard"
    backend: BackendName = "transformers"
    max_new_tokens: int = 256
    temperature: float = 0.7
    seed: int | None = None
    alpha: float = 4.0
    mcmc_steps: int = 2
    block_count: int = 8
    stop_on_eos: bool = True

    @property
    def proposal_temperature(self) -> float:
        """Temperature used by the power-sampling proposal distribution."""

        if self.alpha <= 0:
            raise ValueError("alpha must be greater than 0")
        return 1.0 / self.alpha


PRESETS: dict[Preset, GenerationConfig] = {
    "fast": GenerationConfig(sampler="power", alpha=2.0, mcmc_steps=2, block_count=8),
    "balanced": GenerationConfig(sampler="power", alpha=4.0, mcmc_steps=5, block_count=12),
    "deep": GenerationConfig(sampler="power", alpha=4.0, mcmc_steps=10, block_count=16),
}


def get_preset(name: Preset, **overrides: object) -> GenerationConfig:
    """Return a generation config preset with optional dataclass overrides."""

    if name not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"unknown preset {name!r}; expected one of: {valid}")
    return replace(PRESETS[name], **overrides)

