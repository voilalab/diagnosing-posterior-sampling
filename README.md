# When, why, and how do diffusion posterior samplers fail? A finite-sample lens

Official code for 

- Title: *"When, why, and how do diffusion posterior samplers fail? A finite-sample lens"*
- Authors: Benjamin A. Burns and Sara Fridovich-Keil
- Affil: Georgia Institute of Technology
- Link: [arXiv:2605.30330](https://arxiv.org/abs/2605.30330).

Diffusion-based posterior samplers must replace the intractable likelihood
$p(\mathbf y \mid \mathbf x_t)$ at intermediate diffusion times with an inexact approximation, and
the downstream effect of that approximation on the sampled posterior is poorly
understood. This library provides a **finite-sample reference (FSR)**: by treating
the prior as the empirical measure of $N$ samples, the posterior becomes
analytically computable at every diffusion time and approaches the true posterior
as $N \to \infty$, for any forward model and prior. Measuring the popular approximations
(DPS, ΠiGDM, TMPD) against the FSR reveals *when, why, and how* they fail — they
tend to under- or over-estimate the posterior spread at intermediate times,
causing sensitivity to early-stopping time, inaccurate relative weighting of
posterior modes, and hallucination of prior modes absent from the posterior (or
likelihood modes unsupported by the prior). The FSR is agnostic to the
approximation and to the (linear or nonlinear) forward model, so it serves as a
drop-in diagnostic for existing and future posterior samplers.

The library is designed to be used off the shelf: bring your own prior samples and
a forward model, and recover posterior samples in a few lines.

## Install

```bash
# Install uv (if you don't have it already)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Install dependencies
uv sync                    # runtime + test dependencies
uv sync --group docs       # docs toolchain (optional)
```

## Quickstart

```python
import torch
from src.fsr import run_fsr
from src.forward_model import AffineForwardModel

atoms = torch.randn(512, 1)           # (N, d) empirical prior samples
y = torch.tensor([0.7])               # (m,) observation
forward = AffineForwardModel(
    matrix=torch.tensor(1.0), bias=torch.tensor(0.0), name="identity",
)
samples = run_fsr(atoms, y, forward, noise_scale=0.3)
```

`samples` has shape `(num_particles, d)` and is drawn from the FSR approximation to
`p(x_0 | y)`. If your data is a `numpy` array, a CSV, a `.pt` file, or a list of
individual sample tensors, `src.data_loaders` (`from_numpy`, `from_csv`,
`from_torch_file`, `from_tensors`) lands it into the `(N, d)` tensor `run_fsr`
expects. See `examples/quickstart.py` for the full version with a plotted
histogram; run it with:

```bash
uv run python -m examples.quickstart
```

## Repo layout

- `src/fsr.py` (`run_fsr`) / `src/scores/fsr.py` (`FSR`) — the **finite-sample
  reference**: the analytic finite-sample posterior that every approximation is
  measured against (paper §4).
- `src/scores/` — the likelihood-score approximations the paper compares:
  - `dps.py` — **$\sigma$-DPS / $\zeta$-DPS** (Chung et al. 2023): one-moment Dirac denoiser
    approximation; works for any forward model. ζ-DPS adds a data-dependent
    likelihood weight.
  - `pigdm.py` — **$\Pi$GDM** (Song et al. 2023): two-moment Gaussian approximation
    with an isotropic denoiser covariance; **linear forward models only**.
  - `tmpd.py` — **TMPD** (Boys et al. 2024): two-moment Gaussian approximation
    using the true denoiser covariance; **linear forward models only**.
- `src/distributions/` — analytic priors (discrete, Gaussian, GMM) with closed-form
  marginals, denoisers, likelihoods, posteriors, and scores — the ground truth the
  examples overlay against.
- `src/samplers/` — Euler–Maruyama reverse-time VP-SDE integration.
- `src/sde/` — VP / OU noise schedules.
- `src/tweedie.py`, `src/weights.py` — Tweedie moment helpers and finite-sample
  mixture weights / prior-score machinery shared across methods.
- `examples/` — runnable scripts:
  - `quickstart.py` — the five-liner above with a plot.
  - `discrete_example.py`, `gaussian_example.py`, `gmm_example.py` — FSR posterior
    samples overlaid on the closed-form posterior, per prior family.
  - `compare_methods.py` — posterior-density overlay of FSR vs σ-DPS vs ζ-DPS vs
    ΠiGDM vs TMPD on a discrete testbed (the per-method comparison from the paper).
  - `heatmap_example.py` — a paper-style `(t, x)` posterior-density heatmap.
  - `_common/` — shared helpers: `plotting.py` (style + histograms),
    `sampling.py` (generic reverse-EM loop), `heatmap.py` (the `(t, x)` density
    heatmaps that form the paper's headline figures).
- `tests/` — pytest suite.
- `docsrc/` — Sphinx documentation source (build from `docsrc/source/`).

## Producing figures

The shipped examples plus `examples/_common/heatmap.py` reproduce the *kinds* of
figures in the paper on the small testbeds: `heatmap_example.py` builds a
`(t, x)` density map like the "True posterior" / FSR column of the linear
multimodal grid, and `compare_methods.py` builds the per-method posterior overlay.

## Citation

```bibtex
@misc{burns2026whenwhydiffusionposterior,
      title={When, why, and how do diffusion posterior samplers fail? A finite-sample lens},
      author={Benjamin A. Burns and Sara Fridovich-Keil},
      year={2026},
      eprint={2605.30330},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.30330},
}
```

## License

See `LICENSE`.
