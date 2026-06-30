"""Hugging Face Transformers backend."""

from __future__ import annotations

from typing import Any

from localbooster.backends.base import SampledContinuation
from localbooster.backends.errors import OptionalDependencyError


class TransformersBackend:
    """Token-by-token Transformers backend with proposal and target log-prob traces."""

    backend_name = "transformers"

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        dtype: str | None = "auto",
        trust_remote_code: bool = True,
        **from_pretrained_kwargs: Any,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - exercised only without optional deps
            raise OptionalDependencyError(
                "Transformers backend requires `localbooster[transformers]`."
            ) from exc

        self.torch = torch
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        kwargs.update(from_pretrained_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.device = device or _default_torch_device(torch)
        self.model.to(self.device)
        self.model.eval()
        self.eos_token_id = self.tokenizer.eos_token_id
        self.context_size = int(getattr(self.model.config, "max_position_embeddings", 4096))

    def encode(self, text: str) -> list[int]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        return [int(token_id) for token_id in encoded]

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

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
        torch = self.torch
        if seed is not None:
            torch.manual_seed(seed)
        generated: list[int] = []
        proposal_logprobs: list[float] = []
        target_logprobs: list[float] = []
        hit_eos = False

        with torch.no_grad():
            tokens = list(prefix)
            for _ in range(max_new_tokens):
                input_tokens = tokens[-self.context_size :]
                input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
                logits = self.model(input_ids).logits[0, -1, :]
                base_logprobs = torch.log_softmax(logits, dim=-1)
                proposal_logits = logits / temperature
                proposal_probs = torch.softmax(proposal_logits, dim=-1)
                next_token = int(torch.multinomial(proposal_probs, num_samples=1).item())

                generated.append(next_token)
                tokens.append(next_token)
                proposal_logprobs.append(float(torch.log(proposal_probs[next_token]).item()))
                target_logprobs.append(float((alpha * base_logprobs[next_token]).item()))

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


def _default_torch_device(torch: Any) -> str:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
