"""Power sampling with autoregressive Metropolis-Hastings updates."""

from __future__ import annotations

import math
import random
import time

from localbooster.sampling.base import Sampler


class PowerSampler(Sampler):
    """Approximate power-distribution sampling with local MCMC resampling.

    This implements the product-facing version of the paper algorithm: grow the sequence in
    blocks, then repeatedly resample suffixes and accept/reject with a Metropolis-Hastings ratio
    computed from target and proposal log-probability traces.
    """

    def generate(self, prompt: str):
        started_at = time.perf_counter()
        rng = random.Random(self.config.seed)
        prefix = self.backend.encode(prompt)
        current = list(prefix)
        generated_target = self.config.max_new_tokens
        block_count = max(1, self.config.block_count)
        block_size = max(1, math.ceil(generated_target / block_count))

        proposal_logprobs: list[float] = []
        target_logprobs: list[float] = []
        sampled_tokens = 0
        attempts = 0
        acceptances = 0

        while len(current) - len(prefix) < generated_target:
            remaining = generated_target - (len(current) - len(prefix))
            extension_len = min(block_size, remaining)
            extension = self.backend.sample_continuation(
                current,
                max_new_tokens=extension_len,
                temperature=self.config.proposal_temperature,
                alpha=self.config.alpha,
                seed=rng.randrange(2**32),
            )
            current.extend(extension.token_ids)
            proposal_logprobs.extend(extension.proposal_logprobs)
            target_logprobs.extend(extension.target_logprobs)
            sampled_tokens += extension.sampled_tokens

            if extension.hit_eos and self.config.stop_on_eos:
                break

            for _ in range(max(0, self.config.mcmc_steps)):
                if len(current) <= len(prefix):
                    break
                attempts += 1
                idx = rng.randrange(len(prefix), len(current))
                suffix_len = len(current) - idx
                proposal = self.backend.sample_continuation(
                    current[:idx],
                    max_new_tokens=suffix_len,
                    temperature=self.config.proposal_temperature,
                    alpha=self.config.alpha,
                    seed=rng.randrange(2**32),
                )
                sampled_tokens += proposal.sampled_tokens

                old_start = idx - len(prefix)
                old_end = old_start + len(proposal.token_ids)
                old_prop = proposal_logprobs[old_start:old_end]
                old_target = target_logprobs[old_start:old_end]

                log_ratio = (
                    sum(proposal.target_logprobs)
                    + sum(old_prop)
                    - sum(old_target)
                    - sum(proposal.proposal_logprobs)
                )
                if _accept(log_ratio, rng):
                    acceptances += 1
                    current = current[:idx] + proposal.token_ids
                    proposal_logprobs = (
                        proposal_logprobs[:old_start] + proposal.proposal_logprobs
                    )
                    target_logprobs = target_logprobs[:old_start] + proposal.target_logprobs

            if (
                self.config.stop_on_eos
                and self.backend.eos_token_id is not None
                and self.backend.eos_token_id in current[len(prefix) :]
            ):
                eos_idx = current.index(self.backend.eos_token_id, len(prefix))
                keep = eos_idx + 1 - len(prefix)
                current = current[: eos_idx + 1]
                proposal_logprobs = proposal_logprobs[:keep]
                target_logprobs = target_logprobs[:keep]
                break

        return self._result(
            text=self.backend.decode(current),
            token_ids=current,
            started_at=started_at,
            generated_tokens=len(current) - len(prefix),
            sampled_tokens=sampled_tokens,
            accepted_proposals=acceptances,
            attempted_proposals=attempts,
            extra={
                "alpha": self.config.alpha,
                "mcmc_steps": self.config.mcmc_steps,
                "block_count": self.config.block_count,
                "block_size": block_size,
            },
        )


def _accept(log_ratio: float, rng: random.Random) -> bool:
    if log_ratio >= 0:
        return True
    return math.log(max(rng.random(), 1e-300)) < log_ratio
