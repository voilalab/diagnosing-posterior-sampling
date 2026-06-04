# Framework

We assume a prior $p_0(\mathbf x)$, a measurement model

$$
\mathbf y = \mathcal{A}(\mathbf x) + \bm\eta, \qquad \bm\eta \sim \mathcal{N}(\mathbf 0, \bm\Sigma_{\mathrm n}),
$$

and a diffusion model whose forward semigroup acts as

$$
p(\mathbf x_t \mid \mathbf x_0) = \mathcal{N}\!\bigl(\mathbf x_t;\sqrt{\bar\alpha(t)}\,\mathbf x_0, (1-\bar\alpha(t))\mathbf I_d\bigr).
$$

The measurement operator $\mathcal{A}\colon \mathcal X \to \mathcal Y$ may be
nonlinear; in the linear special case we write $\mathcal{A}(\mathbf x) = \mathbf A \mathbf x$.

The goal is to sample from the posterior $p(\mathbf x_0 \mid \mathbf y)$. By
Bayes' rule, this can be done with a diffusion model whose score is the sum of
the unconditional prior score and the time-$t$ likelihood score:

$$
\nabla_{\mathbf x_t} \log p(\mathbf x_t \mid \mathbf y) = \nabla_{\mathbf x_t} \log p(\mathbf x_t) + \nabla_{\mathbf x_t} \log p(\mathbf y \mid \mathbf x_t).
$$

The prior score is accessible (e.g. by training an unconditional diffusion
model). The **likelihood score** is the obstacle:

$$
p(\mathbf y \mid \mathbf x_t) = \int_{\mathcal X} p(\mathbf y \mid \mathbf x_0) \, p(\mathbf x_0 \mid \mathbf x_t) \, \mathrm d\mathbf x_0.
$$

The integrand $p(\mathbf y \mid \mathbf x_0) = \mathcal{N}(\mathbf y; \mathcal{A}(\mathbf x_0), \bm\Sigma_{\mathrm n})$
is easy at fixed $\mathbf x_0$, but a Monte-Carlo estimate of the integral
requires denoising many $\mathbf x_0$ samples for *every single* evaluation
of $\mathbf x_t$ — prohibitively expensive in practice.

Existing tractable approximations replace the integral with a closed-form
surrogate: the one-moment Dirac approximation (σ-DPS / ζ-DPS,
{doc}`one_moment_chung`) and the two-moment Gaussian approximations (ΠiGDM and
TMPD), the latter restricted to affine $\mathcal A$ with Gaussian noise. This
project's contribution is the {doc}`finite-sample <finite_sample>` reference
that lets us measure the gap each approximation introduces.

In code, the SDE machinery lives in {py:mod}`src.sde` (the abstract
{py:class}`~src.sde.SDE` plus {py:class}`~src.sde.OU` and
{py:class}`~src.sde.VPSDE`), the measurement operator is wrapped by
{py:class}`~src.forward_model.ForwardModel`, and the prior / posterior
score functions live in {py:mod}`src.scores`.
