import torch

from src.sde.base import SDE


def alpha_bar_from_times(times: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    r"""VP :math:`\bar\alpha` schedule evaluated at a tensor of times.

    Args:
        times: batch of times at which to compute :math:`\bar\alpha`
        beta_min: minimum :math:`\beta` of noise schedule
        beta_max: maximum :math:`\beta` of noise schedule

    Returns:
        batch of :math:`\bar\alpha` values
    """
    return torch.exp(-beta_min * times - 0.5 * (beta_max - beta_min) * times * times)


class VPSDE(SDE):
    def __init__(self, beta_min=1e-2, beta_max=20):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.alpha_bar_fn = self._construct_alpha_bar(beta_min, beta_max)

    @staticmethod
    def _construct_alpha_bar(beta_min=1e-2, beta_max=20):
        def alpha_bar(t):
            t_tensor = torch.as_tensor(t)
            alpha = torch.exp(-beta_min * t_tensor - 0.5 * (beta_max - beta_min) * t_tensor * t_tensor)
            if alpha.ndim == 0:
                return alpha.item()
            return alpha
        return alpha_bar

    def semigroup(
        self,
        samples: torch.Tensor,
        t: float,
        k: int,
    ) -> torch.Tensor:
        """Draw k samples from the pushforward of each sample under the VP semigroup.

        For each sample X_0 from the initial distribution at time 0,
        draws k independent samples from the VP process at time t.

        Args:
            samples (torch.Tensor): ``(m, d)`` batch of samples.
            t (float): Time to pushforward to.
            k (int): Number of samples to draw from each input sample.
            alpha_bar_fn (Callable[[float], float]): Function that computes ᾱ(t).

        Returns:
            torch.Tensor: ``(m*k, d)`` batch of expanded samples.
        """
        # Repeat each sample k times
        expanded = samples.repeat_interleave(k, dim=0)  # (m*k, d)

        # Apply VP transition: X_t ~ N(√ᾱ(t) * X_0, (1 - ᾱ(t)) * I)
        t_tensor = torch.as_tensor(t, device=expanded.device, dtype=expanded.dtype)
        alpha_bar_t = torch.exp(
            -self.beta_min * t_tensor
            - 0.5 * (self.beta_max - self.beta_min) * t_tensor * t_tensor
        )
        decay_factor = torch.sqrt(alpha_bar_t)
        noise_variance = 1 - alpha_bar_t

        deterministic_part = expanded * decay_factor
        noise = torch.randn_like(expanded) * torch.sqrt(noise_variance)

        return deterministic_part + noise
