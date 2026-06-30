"""Shared sampler helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass

from localbooster.backends.base import LanguageModelBackend
from localbooster.config import GenerationConfig
from localbooster.metrics import GenerationMetrics, GenerationResult


@dataclass
class Sampler:
    backend: LanguageModelBackend
    config: GenerationConfig

    def generate(self, prompt: str) -> GenerationResult:
        raise NotImplementedError

    def _result(
        self,
        *,
        text: str,
        token_ids: list[int],
        started_at: float,
        generated_tokens: int,
        sampled_tokens: int,
        accepted_proposals: int = 0,
        attempted_proposals: int = 0,
        extra: dict[str, object] | None = None,
    ) -> GenerationResult:
        metrics = GenerationMetrics(
            backend=self.backend.backend_name,
            sampler=self.config.sampler,
            model=self.backend.model_id,
            latency_seconds=time.perf_counter() - started_at,
            generated_tokens=generated_tokens,
            sampled_tokens=sampled_tokens,
            accepted_proposals=accepted_proposals,
            attempted_proposals=attempted_proposals,
            seed=self.config.seed,
            extra=extra or {},
        )
        return GenerationResult(text=text, token_ids=token_ids, metrics=metrics)

