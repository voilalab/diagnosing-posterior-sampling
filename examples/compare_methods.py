r"""Posterior-density overlay: FSR vs SigmaDPS vs ZetaDPS vs PiGDM vs TMPD.

All five likelihood approximations run on the same three-atom discrete
testbed, each driven by the *analytic* prior marginal score from
:meth:`Discrete.marginal_score` so the only difference between methods is
their treatment of :math:`p(y \mid x_t)`.  Empirical posterior histograms
are overlaid against the closed-form
:meth:`Discrete.posterior_density` at a short positive time.

Run from the repository root::

    uv run python -m examples.compare_methods
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from examples._common.plotting import empirical_pdf, setup_style
from examples._common.sampling import reverse_em
from src.distributions.discrete import Discrete
from src.forward_model import AffineForwardModel
from src.scores import make_score_fn
from src.sde import VPSDE

setup_style()

NUM_PARTICLES = 4096
NUM_STEPS = 500
T_MIN = 1e-2
T_MAX = 1.0
NUM_ATOMS = 1024


def main() -> None:
    """Run all five methods on the tri_equal discrete testbed and overlay results."""
    sde = VPSDE()
    dtype = torch.float64

    atoms_support = torch.tensor([-1.8, 0.2, 2.2], dtype=dtype)
    prior_weights = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)
    forward = AffineForwardModel(
        matrix=torch.tensor(1.0, dtype=dtype),
        bias=torch.tensor(0.0, dtype=dtype),
        name="identity",
    )
    noise_scale = 0.3
    prior = Discrete(
        sde, atoms_support, prior_weights,
        forward_model=forward, noise_scale=noise_scale,
    )
    y_value = -1.8

    # Analytic prior score driving every method.
    def prior_score_fn(x_t: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        t = float(times[0].item())
        return prior.marginal_score(x_t, t)

    # FSR uses an empirical prior; the other methods use the analytic prior
    # score above.  All five share the same atoms for the FSR-style methods.
    torch.manual_seed(0)
    atoms = prior.prior_sampler(NUM_ATOMS).unsqueeze(-1)                    # (N, 1)

    methods: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {}
    for kind in ("fsr", "sigma_dps", "pigdm", "tmpd"):
        methods[kind] = make_score_fn(
            kind,
            prior_score_fn=prior_score_fn,
            beta_min=sde.beta_min,
            beta_max=sde.beta_max,
            measurement=y_value,
            noise_variance=noise_scale * noise_scale,
            forward_model=forward,
            atoms=atoms,
        )

    # ZetaDPS isn't exposed via make_score_fn (the factory only wires its
    # SigmaDPS sibling); instantiate it directly.
    from src.scores import ZetaDPS

    zeta_obj = ZetaDPS(
        prior_score_fn=prior_score_fn,
        beta_min=sde.beta_min,
        beta_max=sde.beta_max,
        noise_variance=noise_scale * noise_scale,
        forward_model=forward,
        zeta=0.3,
    )
    methods["zeta_dps"] = lambda x_t, times: zeta_obj.posterior_score(
        y_value, x_t, times, prior_score_fn,
    )

    samples_per_method: dict[str, torch.Tensor] = {}
    for kind, score_fn in methods.items():
        samples_per_method[kind] = reverse_em(
            score_fn, sde=sde,
            num_particles=NUM_PARTICLES, dim=1, num_steps=NUM_STEPS,
            t_min=T_MIN, t_max=T_MAX, dtype=dtype, seed=42,
        )

    x_grid = torch.linspace(-4.0, 4.0, 400, dtype=dtype)
    y_b = torch.full_like(x_grid, y_value)
    analytic = prior.posterior_density(x_grid, y_b, T_MIN)

    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.plot(x_grid.numpy(), analytic.numpy(), color="black", label=f"analytic at $t={T_MIN}$")
    label_map = {
        "fsr": "FSR", "sigma_dps": r"$\sigma$-DPS", "zeta_dps": r"$\zeta$-DPS",
        "pigdm": "PiGDM", "tmpd": "TMPD",
    }
    for kind, samples in samples_per_method.items():
        ax.plot(
            x_grid.numpy(), empirical_pdf(samples, x_grid).numpy(),
            label=label_map[kind], alpha=0.85,
        )
    ax.axvline(y_value, linestyle="--", color="gray", linewidth=0.6, label="$y$")
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=7)

    out = Path(__file__).parent / "compare_methods.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
