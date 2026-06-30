"""Result and metric types produced by LocalBooster."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationMetrics:
    """Metrics needed to judge quality together with local compute cost."""

    backend: str
    sampler: str
    model: str
    latency_seconds: float
    generated_tokens: int
    sampled_tokens: int
    accepted_proposals: int = 0
    attempted_proposals: int = 0
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def acceptance_ratio(self) -> float | None:
        if self.attempted_proposals == 0:
            return None
        return self.accepted_proposals / self.attempted_proposals

    @property
    def cost_multiplier(self) -> float | None:
        if self.generated_tokens == 0:
            return None
        return self.sampled_tokens / self.generated_tokens

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["acceptance_ratio"] = self.acceptance_ratio
        data["cost_multiplier"] = self.cost_multiplier
        return data


@dataclass(frozen=True)
class GenerationResult:
    """Text plus machine-readable generation metrics."""

    text: str
    token_ids: list[int]
    metrics: GenerationMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "token_ids": self.token_ids,
            "metrics": self.metrics.to_dict(),
        }

