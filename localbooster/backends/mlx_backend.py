"""MLX backend for Apple Silicon."""

from __future__ import annotations

import math
import random
from typing import Any

from localbooster.backends.base import SampledContinuation
from localbooster.backends.errors import OptionalDependencyError


class MLXBackend:
    """Minimal MLX-LM backend.

    The implementation intentionally samples token-by-token from model logits. It is slower than
    an optimized cached path, but it gives LocalBooster the log-probability traces needed by power
    sampling and keeps the first implementation easy to verify.
    """

    backend_name = "mlx"

    def __init__(self, model_id: str, **load_kwargs: Any) -> None:
        try:
            import mlx.core as mx
            import numpy as np
            from mlx_lm import load
        except Exception as exc:  # pragma: no cover - exercised only without optional deps
            raise OptionalDependencyError("MLX backend requires `localbooster[mlx]`.") from exc

        self.mx = mx
        self.np = np
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
        rng = random.Random(seed)
        generated: list[int] = []
        proposal_logprobs: list[float] = []
        target_logprobs: list[float] = []
        tokens = list(prefix)
        hit_eos = False

        for _ in range(max_new_tokens):
            input_tokens = tokens[-self.context_size :]
            logits = self._next_logits(input_tokens)
            base_logprobs = _log_softmax(logits, self.np)
            proposal_logprobs_arr = _log_softmax(logits / temperature, self.np)
            probs = self.np.exp(proposal_logprobs_arr)
            probs = probs / probs.sum()
            next_token = int(rng.choices(range(len(probs)), weights=probs, k=1)[0])

            generated.append(next_token)
            tokens.append(next_token)
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

    def _next_logits(self, token_ids: list[int]):
        mx = self.mx
        np = self.np
        inputs = mx.array([token_ids])
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        mx.eval(outputs)
        return np.array(outputs[0, -1], dtype=float)


def _log_softmax(logits, np):
    max_logit = logits.max()
    shifted = logits - max_logit
    return shifted - math.log(float(np.exp(shifted).sum()))
