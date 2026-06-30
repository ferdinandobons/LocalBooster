"""MLX backend for Apple Silicon."""

from __future__ import annotations

import math
from typing import Any

from localbooster.backends.base import SampledContinuation
from localbooster.backends.errors import OptionalDependencyError


class MLXBackend:
    """MLX-LM backend with cached continuation sampling.

    MLX-LM handles prompt prefill and per-token generation with a KV cache. LocalBooster keeps
    the returned base log-probability vectors so power sampling can compute both proposal and
    target weights for Metropolis-Hastings.
    """

    backend_name = "mlx"

    def __init__(self, model_id: str, **load_kwargs: Any) -> None:
        try:
            import mlx.core as mx
            import numpy as np
            from mlx_lm.generate import generate_step
            from mlx_lm import load
            from mlx_lm.sample_utils import make_sampler
        except Exception as exc:  # pragma: no cover - exercised only without optional deps
            raise OptionalDependencyError("MLX backend requires `localbooster[mlx]`.") from exc

        self.mx = mx
        self.np = np
        self.generate_step = generate_step
        self.make_sampler = make_sampler
        self.model_id = model_id
        self.model, self.tokenizer = load(model_id, **load_kwargs)
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        self.context_size = int(getattr(self.model, "max_seq_length", 4096))

    def encode(self, text: str) -> list[int]:
        encoded = self.tokenizer.encode(text)
        if hasattr(encoded, "ids"):
            encoded = encoded.ids
        return [int(token_id) for token_id in encoded]

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def sample_continuation(
        self,
        prefix: list[int],
        *,
        max_new_tokens: int,
        temperature: float,
        alpha: float,
        seed: int | None = None,
    ) -> SampledContinuation:
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")
        if seed is not None:
            self.mx.random.seed(seed)
        generated: list[int] = []
        proposal_logprobs: list[float] = []
        target_logprobs: list[float] = []
        hit_eos = False
        prompt = self.mx.array(prefix[-self.context_size :], dtype=self.mx.uint32)
        sampler = self.make_sampler(temp=temperature)

        for token, logprobs in self.generate_step(
            prompt,
            self.model,
            max_tokens=max_new_tokens,
            sampler=sampler,
        ):
            next_token = int(token)
            base_logprobs = self._to_numpy(logprobs)
            proposal_logprobs_arr = _log_softmax(base_logprobs / temperature, self.np)

            generated.append(next_token)
            proposal_logprobs.append(float(proposal_logprobs_arr[next_token]))
            target_logprobs.append(float(alpha * base_logprobs[next_token]))

            if self.eos_token_id is not None and next_token == self.eos_token_id:
                hit_eos = True
                break

        return SampledContinuation(
            token_ids=generated,
            proposal_logprobs=proposal_logprobs,
            target_logprobs=target_logprobs,
            sampled_tokens=len(generated),
            hit_eos=hit_eos,
        )

    def _to_numpy(self, array):
        array = array.astype(self.mx.float32)
        self.mx.eval(array)
        return self.np.array(array.tolist(), dtype=float)


def _log_softmax(logits, np):
    max_logit = logits.max()
    shifted = logits - max_logit
    return shifted - math.log(float(np.exp(shifted).sum()))
