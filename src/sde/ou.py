"""Ornstein-Uhlenbeck process."""

import math

import torch

from src.sde.base import SDE


class OU(SDE):
    def __init__(self, mean: float = 0, sigma: float = math.sqrt(2)):
        self.mean = mean
        self.sigma = sigma

    def semigroup(
        self,
        samples: torch.Tensor,
        t: float,
        k: int,
    ) -> torch.Tensor:
        """Draw k samples from the pushforward of each sample under the OU semigroup.

        For each sample :math:`X_0` from the initial distribution at time :math:`0`,
        draws k independent samples from the law of the OU process at time :math:`t`.

        Args:
            samples (torch.Tensor): ``(m, d)`` batch of samples.
            t (float): Time to pushforward to.
            k (int): Number of samples to draw from each input sample.
            mean (float): Mean of OU process.
            sigma (float): Diffusion coefficient of OU process.

        Returns:
            torch.Tensor: ``(m*k, d)`` batch of expanded samples.
        """
        # Repeat each sample k times
        expanded = samples.repeat_interleave(k, dim=0)  # (m*k, d)

        # Apply OU transition to all samples
        t_tensor = torch.as_tensor(t, device=expanded.device, dtype=expanded.dtype)
        decay_factor = torch.exp(-t_tensor)
        noise_variance = (self.sigma**2) * (1 - torch.exp(-2 * t_tensor)) / 2

        deterministic_part = self.mean + (expanded - self.mean) * decay_factor
        noise = torch.randn_like(expanded) * torch.sqrt(noise_variance)

        return deterministic_part + noise
