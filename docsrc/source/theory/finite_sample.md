# Finite-sample perspective

In practice we never see the prior $\pi$ — only $N$ samples from it, the
empirical measure $\pi^N = \tfrac{1}{N} \sum_{i=1}^N \delta_{\mathbf x^{(i)}}$.
Pushing $\pi^N$ forward through the diffusion semigroup gives a
**Gaussian mixture**:

$$
p(\mathbf x_t) = \tfrac{1}{N} \sum_{i=1}^N \mathcal{N}\!\bigl(\mathbf x_t;\, \sqrt{\bar\alpha(t)}\,\mathbf x^{(i)},\, (1-\bar\alpha(t))\mathbf I_d\bigr).
$$

By Bayes' rule on this mixture, the backwards posterior is a **Dirac mixture**:

$$
p(\mathbf x_0 \mid \mathbf x_t) = \sum_{i=1}^N w_i(\mathbf x_t)\, \delta\!\bigl(\mathbf x_0 - \mathbf x^{(i)}\bigr),
$$

with weights $w_i(\mathbf x_t)$ proportional to each component's density at
$\mathbf x_t$. Integrals against this measure are not just tractable — they
are *exact*: we are simply summing a function over the training cloud.

## Consequences

Combining with the likelihood gives an *exact* finite-sample posterior with
re-weighted atoms:

$$
p(\mathbf x_0 \mid \mathbf x_t, \mathbf y) = \sum_{i=1}^N \tilde w_i(\mathbf x_t, \mathbf y)\, \delta\!\bigl(\mathbf x_0 - \mathbf x^{(i)}\bigr),
\qquad
\tilde w_i \propto w_i(\mathbf x_t)\, p(\mathbf y \mid \mathbf x^{(i)}).
$$

The resulting **likelihood score** is

$$
\nabla_{\mathbf x_t} \log p(\mathbf y \mid \mathbf x_t)
= \frac{\sqrt{\bar\alpha(t)}}{1-\bar\alpha(t)} \left( \sum_{i=1}^N \tilde w_i \, \mathbf x^{(i)} - \sum_{i=1}^N w_i \, \mathbf x^{(i)} \right).
$$

That is, the prefactor times the difference between the *posterior-weighted
mean* and the *prior-weighted mean* — both computable directly from the
training cloud. This is implemented by {py:class}`src.scores.FSR`
(`likelihood_score` for the term above, `posterior_score` for the sum with the
prior score); {py:func}`src.fsr.run_fsr` wraps it with reverse-time sampling.

## Why this is the right reference

Convergence guarantees:

- $\pi^N \to \pi$ almost surely as $N \to \infty$.
- At positive time $t > 0$, the diffusion semigroup is non-degenerate Gaussian
  smoothing, so we get convergence in **total variation** (a Feller argument).
- Hence the finite-sample posterior converges in TV to the true posterior.

These guarantees are not *quantitative*. For high-dimensional, spiky priors
(natural images at every realistic scale resemble a Gaussian mixture) we will
never see "enough" of the support, so $N$ is genuinely finite — the rate at
which finite-$N$ artefacts decay matters, and the paper measures this Monte
Carlo convergence of the FSR posterior in total variation.

## In code

| Symbol | Object |
| --- | --- |
| $w_i(\mathbf x_t)$ | {py:func}`src.weights.compute_weights` |
| Finite-sample likelihood / posterior score | {py:class}`src.scores.FSR` (`likelihood_score` / `posterior_score`) |
| End-to-end posterior sampling | {py:func}`src.fsr.run_fsr` |
| Mixture-weight log-prior + score | {py:func}`src.weights.prior_terms` |
| $\sqrt{\bar\alpha(t)} / (1-\bar\alpha(t))$ prefactor | {py:func}`src.weights.likelihood_prefactor` |
