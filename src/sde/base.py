"""Stochastic differential equation base class."""

import math
from abc import ABC, abstractmethod

import torch


class SDE(ABC):

    @abstractmethod
    def semigroup(
        self,
        samples: torch.Tensor,
        t: float,
        k: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def simulate(
        self,
        prior_samples: torch.Tensor,
        dt: float,
        final_t: float,
        n_samples: int,
    ) -> torch.Tensor:
        """Simulate forward process with fixed sample count.

        Given prior distribution μ represented by m samples, simulates the
        forward process (P_t)_#μ for t = dt, 2*dt, ..., final_t with n samples
        at each time.

        Expands from m to n samples at the first timestep, then maintains n samples.

        Args:
            prior_samples (torch.Tensor): ``(m, d)`` batch of samples from prior.
            dt (float): Time step.
            final_t (float): Final time.
            n_samples (int): Number of samples at each positive time.

        Returns:
            torch.Tensor: ``(num_steps, n_samples, d)`` samples at each time.
        """
        num_steps = int(final_t / dt)
        m = prior_samples.shape[0]

        # Calculate expansion factor for first step
        k = math.ceil(n_samples / m)

        # Expand m -> k*m at first timestep
        current_samples = self.semigroup(prior_samples, dt, k)

        # Trim to exactly n_samples if we overshot
        if current_samples.shape[0] > n_samples:
            # Shuffle to remove particles from all modes, not just the last mode
            indices = torch.randperm(current_samples.shape[0], device=current_samples.device)
            current_samples = current_samples[indices[:n_samples]]

        # Store results
        all_samples = torch.empty(
            (num_steps, n_samples, *prior_samples.shape[1:]),
            device=prior_samples.device,
            dtype=prior_samples.dtype,
        )
        all_samples[0] = current_samples

        # Evolve n -> n for remaining timesteps
        for step in range(1, num_steps):
            current_samples = self.semigroup(current_samples, dt, k=1)
            all_samples[step] = current_samples

        return all_samples

    def expanding_forward_process(
        self,
        prior_samples: torch.Tensor,
        dt: float,
        final_t: float,
        k: int,
    ) -> list[torch.Tensor]:
        """Simulate forward process with exponential sample expansion.

        Given prior distribution μ represented by m samples, simulates the
        forward process (P_t)_#μ for t = dt, 2*dt, ..., final_t, expanding
        by factor k at each timestep for best statistical representation.

        WARNING: Memory grows as m * k^num_steps. Use only for small k and num_steps.

        Args:
            prior_samples (torch.Tensor): ``(m, d)`` batch of samples from prior.
            dt (float): Time step.
            final_t (float): Final time.
            k (int): Expansion factor at each timestep.

        Returns:
            list[torch.Tensor]: ``(num_steps,)`` list where element ``i`` has
                shape ``(m * k^(i+1), d)``.
        """
        num_steps = int(final_t / dt)
        all_samples = []

        current_samples = prior_samples.clone()

        for _ in range(num_steps):
            # Expand by factor k at each step
            current_samples = self.semigroup(current_samples, dt, k)
            all_samples.append(current_samples.clone())

        return all_samples
