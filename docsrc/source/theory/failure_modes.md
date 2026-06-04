# Failure modes

The one-moment approximation evaluates the likelihood at
$\mathcal{A}(\mathbb{E}[\mathbf x_0 \mid \mathbf x_t])$ — a measurement of
the *expectation*. In the finite-sample picture this is

$$
\mathcal{A}\!\left( \sum_{i=1}^N w_i(\mathbf x_t)\, \mathbf x^{(i)} \right)
$$

— a measurement of a **convex combination of training points**. The exact
likelihood instead computes

$$
\sum_{i=1}^N w_i(\mathbf x_t)\, p\!\bigl(\mathbf y \mid \mathbf x^{(i)}\bigr)
\propto
\sum_{i=1}^N w_i(\mathbf x_t)\, \exp\!\Bigl(-\tfrac{1}{2}\| \mathbf y - \mathcal{A}(\mathbf x^{(i)})\|_{\bm\Sigma_{\mathrm n}^{-1}}^2\Bigr),
$$

— an average of measurements. These two objects coincide when $\mathcal{A}$
is linear, but diverge sharply when:

## 1. $\mathcal{A}$ is nonlinear

The convex hull of the training cloud need not lie on the prior's support.
For data on a manifold (Swiss roll, torus, sphere, …), the mean of two
samples is *off-manifold*, and applying a manifold-aware $\mathcal{A}$ —
radial distance, polar angle, an MLP-wrapped operator — to that off-manifold
point gives a measurement that has nothing to do with what any single
training point would produce. The one-moment approximation then steers samples
toward this fictional point.

## 2. The prior has disconnected support

With two well-separated mixture modes, the denoiser mean sits in the *gap*
between them. The one-moment approximation evaluates the likelihood there,
while the finite-sample reference averages over both modes properly. The paper
shows that this multimodal-prior effect alone — with a linear operator,
Gaussian noise, and a unimodal posterior — is enough to produce erroneous
posterior spread and mode mis-weighting.

## 3. Low measurement noise

As $\bm\Sigma_{\mathrm n} \to 0$, the likelihood becomes peaked. Any
disagreement between
$\mathcal{A}(\mathbb{E}[\mathbf x_0 \mid \mathbf x_t])$ and $\mathbf y$ is
amplified, so the geometric error from §1 / §2 turns into a posterior with
vanishing mass in the right place. This regime is where the one-moment
approximation visibly diverges from the truth on otherwise benign priors; the
`noise_scale` argument of {py:func}`src.fsr.run_fsr` and the testbeds controls
how peaked the likelihood is.

## What the two-moment approximation fixes — and doesn't

For **linear** $\mathcal{A}(\mathbf x) = \mathbf A \mathbf x$, Boys'
covariance correction restores the true posterior covariance and the error
from §3 collapses. For **nonlinear** $\mathcal{A}$ the approximation fails
to close (no Gaussian-Gaussian conjugacy) and the geometric problem of §1
returns.

The exact finite-sample sampler ({doc}`finite_sample`) is unaffected by any
of §1–§3 by construction, since it never collapses the mixture in the first
place. That's what makes it a useful reference.
