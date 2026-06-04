# Diagnosing posterior sampling: a finite-sample lens

Official code for *"When, why, and how do diffusion posterior samplers fail?
A finite-sample lens"* (Benjamin A. Burns and Sara Fridovich-Keil, 2026),
[arXiv:2605.30330](https://arxiv.org/abs/2605.30330).

Diffusion-based posterior samplers must replace the intractable likelihood
$p(\mathbf y \mid \mathbf x_t)$ at intermediate diffusion times with an inexact
approximation. This library provides a **finite-sample reference (FSR)**: by
treating the prior as the empirical measure of $N$ samples, the posterior is
computable analytically at every diffusion time and approaches the true
posterior as $N \to \infty$, for any forward model and prior. Measured against
the FSR, the popular approximations (DPS, ΠiGDM, TMPD) reveal *when, why, and
how* they fail — under- or over-estimated posterior spread at intermediate
times, sensitivity to early-stopping time, inaccurate relative weighting of
posterior modes, and hallucination of prior modes absent from the posterior.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                       # runtime + test dependencies
uv sync --group docs          # add the docs toolchain
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

`samples` has shape `(num_particles, d)` and is drawn from the FSR approximation
to $p(\mathbf x_0 \mid \mathbf y)$. Worked examples — analytic-prior overlays,
the five-method comparison, and a paper-style $(t, x)$ heatmap — live under
`examples/`; see {doc}`discrete_testbed` for the walkthrough.

## Theory in one paragraph

A diffusion model whose prior is $N$ samples is, at every positive time $t$, a
Gaussian mixture over those samples. From this empirical-measure view the
backwards posterior $p(\mathbf x_0 \mid \mathbf x_t)$ is a *Dirac mixture*, so
likelihood and posterior integrals against it are exact — this is the FSR. The
moment-matching approximations instead collapse the denoiser to a single moment
(a Dirac at its mean, σ-DPS / ζ-DPS) or two moments (a Gaussian, ΠiGDM / TMPD);
the FSR is the controlled reference against which to measure the gap each
approximation introduces. See {doc}`theory/index`.

## The methods

All are dispatched through {py:func}`src.scores.make_score_fn` (kinds `"fsr"`,
`"sigma_dps"`, `"tmpd"`, `"pigdm"`), or constructed directly.

::::{grid} 2
:gutter: 3

:::{grid-item-card} `fsr` — finite-sample reference
**Exact** posterior of the empirical-measure prior. The reference against which
every approximation is measured ({doc}`theory/finite_sample`).
+++
{py:class}`src.scores.FSR` · {py:func}`src.fsr.run_fsr`
:::

:::{grid-item-card} σ-DPS / ζ-DPS
**Chung et al. (2023).** One-moment Dirac denoiser approximation; works for any
forward model. ζ-DPS adds a data-dependent likelihood weight
({doc}`theory/one_moment_chung`).
+++
{py:class}`src.scores.SigmaDPS` · {py:class}`src.scores.ZetaDPS`
:::

:::{grid-item-card} ΠiGDM
**Song et al. (2023).** Two-moment Gaussian approximation with an isotropic
denoiser covariance. Linear forward models only.
+++
{py:class}`src.scores.PiGDM`
:::

:::{grid-item-card} TMPD
**Boys et al. (2024).** Two-moment Gaussian approximation using the true
denoiser covariance. Linear forward models only.
+++
{py:class}`src.scores.TMPD`
:::
::::

## Indices

* {ref}`genindex`
* {ref}`modindex`

```{toctree}
:maxdepth: 0
:titlesonly:
:hidden:

theory/index
discrete_testbed
api/src
```
