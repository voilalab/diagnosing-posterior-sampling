# One-moment (Chung) approximation

The Chung et al. one-moment approximation collapses the backwards process
$p(\mathbf x_0 \mid \mathbf x_t)$ to a Dirac at its mean. Two equivalent
derivations:

## Derivation 1 — "Jensen's inequality"

Move the expectation into the conditioning:

$$
p(\mathbf y \mid \mathbf x_t) \approx p\!\left(\mathbf y \,\Big|\, \int_{\mathcal X}\! \mathbf x_0 \, p(\mathbf x_0 \mid \mathbf x_t)\, \mathrm d\mathbf x_0\right) = p(\mathbf y \mid \mathbb{E}[\mathbf x_0 \mid \mathbf x_t]).
$$

## Derivation 2 — Dirac at the mean

Equivalently, replace the backwards-process density with a Dirac at its mean:

$$
p(\mathbf y \mid \mathbf x_t) \approx \int_{\mathcal X}\! p(\mathbf y \mid \mathbf x_0)\, \delta\!\bigl(\mathbf x_0 - \mathbb{E}[\mathbf x_0 \mid \mathbf x_t]\bigr) \,\mathrm d\mathbf x_0.
$$

Either way, the resulting likelihood is

$$
p(\mathbf y \mid \mathbb{E}[\mathbf x_0 \mid \mathbf x_t]) = \mathcal{N}\!\bigl(\mathbf y;\, \mathcal{A}(\mathbb{E}[\mathbf x_0 \mid \mathbf x_t]),\, \bm\Sigma_{\mathrm n}\bigr).
$$

This requires only access to $\mathbb{E}[\mathbf x_0 \mid \mathbf x_t]$ —
delivered for free by Tweedie's formula — and differentiability of
$\mathcal{A}$ (almost everywhere) for the score.

## Two variants in this codebase

| Class | Strategy |
| --- | --- |
| {py:class}`~src.scores.SigmaDPS` | Plugs Tweedie's formula directly with the constant measurement-noise prefactor $\sigma_y^{-2}$ — the textbook DPS baseline. |
| {py:class}`~src.scores.ZetaDPS` | Same Dirac approximation, but replaces the prefactor with a data-dependent step size $\zeta / \lVert \mathbf y - \mathcal A(\mathbb{E}[\mathbf x_0 \mid \mathbf x_t]) \rVert$ (equivalent to a Laplace measurement likelihood). |

Both collapse the denoiser to a Dirac at its mean and work for any forward
model; they differ only in how the likelihood is weighted.

## What goes wrong

The mean of the backwards process need not lie on the support of the prior.
For nonlinear $\mathcal{A}$ this means
$\mathcal{A}(\mathbb{E}[\mathbf x_0 \mid \mathbf x_t])$ is a measurement of an
*off-manifold* point — geometrically, this is "a measurement of the average
of the data" rather than "the average of measurements". See
{doc}`failure_modes` for the picture.
