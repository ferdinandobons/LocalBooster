from __future__ import annotations

import unittest

from localbooster.backends.base import SampledContinuation
from localbooster.config import GenerationConfig
from localbooster.generation import build_sampler


class FakeBackend:
    backend_name = "fake"
    model_id = "fake-model"
    eos_token_id = None

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)

    def sample_continuation(
        self,
        prefix: list[int],
        *,
        max_new_tokens: int,
        temperature: float,
        alpha: float,
        seed: int | None = None,
    ) -> SampledContinuation:
        self.calls += 1
        token = ord("a") + (self.calls % 3)
        token_ids = [token] * max_new_tokens
        return SampledContinuation(
            token_ids=token_ids,
            proposal_logprobs=[-0.2] * max_new_tokens,
            target_logprobs=[-0.1 * alpha] * max_new_tokens,
            sampled_tokens=max_new_tokens,
        )


class SamplingTest(unittest.TestCase):
    def test_standard_sampler_generates_text_and_metrics(self):
        backend = FakeBackend()
        config = GenerationConfig(sampler="standard", max_new_tokens=3)
        result = build_sampler(backend, config).generate("x")

        self.assertEqual(result.text, "xbbb")
        self.assertEqual(result.metrics.generated_tokens, 3)
        self.assertEqual(result.metrics.sampled_tokens, 3)
        self.assertIsNone(result.metrics.acceptance_ratio)

    def test_power_sampler_tracks_attempts_and_cost(self):
        backend = FakeBackend()
        config = GenerationConfig(
            sampler="power",
            max_new_tokens=4,
            alpha=2.0,
            mcmc_steps=2,
            block_count=2,
            seed=7,
        )
        result = build_sampler(backend, config).generate("x")

        self.assertTrue(result.text.startswith("x"))
        self.assertEqual(result.metrics.generated_tokens, 4)
        self.assertEqual(result.metrics.attempted_proposals, 4)
        self.assertGreater(result.metrics.sampled_tokens, result.metrics.generated_tokens)
        self.assertIsNotNone(result.metrics.cost_multiplier)


if __name__ == "__main__":
    unittest.main()
