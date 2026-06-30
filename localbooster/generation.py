"""High-level generation entrypoints."""

from __future__ import annotations

from dataclasses import replace

from localbooster.backends.base import LanguageModelBackend
from localbooster.config import GenerationConfig
from localbooster.sampling import PowerSampler, StandardSampler, TemperatureSampler


def build_sampler(backend: LanguageModelBackend, config: GenerationConfig):
    if config.sampler == "standard":
        return StandardSampler(backend=backend, config=config)
    if config.sampler == "temperature":
        return TemperatureSampler(backend=backend, config=config)
    if config.sampler == "power":
        return PowerSampler(backend=backend, config=config)
    raise ValueError(f"unknown sampler {config.sampler!r}")


def with_sampler(config: GenerationConfig, sampler: str) -> GenerationConfig:
    if sampler == "standard":
        return replace(config, sampler="standard", temperature=0.7)
    if sampler == "temperature":
        return replace(config, sampler="temperature", temperature=config.proposal_temperature)
    if sampler in {"power", "power-fast", "power-balanced", "power-deep"}:
        return replace(config, sampler="power")
    raise ValueError(f"unknown sampler {sampler!r}")

