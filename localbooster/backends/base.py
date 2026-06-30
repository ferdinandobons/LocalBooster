"""Backend protocol used by the samplers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SampledContinuation:
    """A token continuation sampled by a backend.

    `proposal_logprobs` are normalized log-probabilities under the proposal distribution. `target`
    log-probabilities are under the base model distribution scaled by alpha, matching the
    Metropolis-Hastings ratio used by power sampling.
    """

    token_ids: list[int]
    proposal_logprobs: list[float]
    target_logprobs: list[float]
    sampled_tokens: int
    hit_eos: bool = False


class LanguageModelBackend(Protocol):
    model_id: str
    backend_name: str
    eos_token_id: int | None

    def encode(self, text: str) -> list[int]:
        """Encode text into model token IDs."""

    def decode(self, token_ids: list[int]) -> str:
        """Decode model token IDs into text."""

    def sample_continuation(
        self,
        prefix: list[int],
        *,
        max_new_tokens: int,
        temperature: float,
        alpha: float,
        seed: int | None = None,
    ) -> SampledContinuation:
        """Sample a continuation and return proposal/target log-probability traces."""

