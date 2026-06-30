"""Standard and low-temperature generation samplers."""

from __future__ import annotations

import time

from localbooster.sampling.base import Sampler


class StandardSampler(Sampler):
    """Single-pass sampled generation."""

    def generate(self, prompt: str):
        started_at = time.perf_counter()
        prefix = self.backend.encode(prompt)
        continuation = self.backend.sample_continuation(
            prefix,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            alpha=1.0,
            seed=self.config.seed,
        )
        token_ids = prefix + continuation.token_ids
        text = self.backend.decode(token_ids)
        return self._result(
            text=text,
            token_ids=token_ids,
            started_at=started_at,
            generated_tokens=len(continuation.token_ids),
            sampled_tokens=continuation.sampled_tokens,
        )


class TemperatureSampler(StandardSampler):
    """Low-temperature baseline sampler."""

