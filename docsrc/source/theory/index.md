# Theory

The mathematical core of the project. Read in order:

1. {doc}`framework` — the measurement model, diffusion semigroup, and how
   posterior sampling decomposes via Bayes' rule.
2. {doc}`one_moment_chung` — the one-moment (σ-DPS / ζ-DPS) approximation, in
   two equivalent presentations.
3. {doc}`finite_sample` — the empirical-measure perspective that lets us
   compute posterior integrals exactly given a finite training set (the FSR).
4. {doc}`failure_modes` — the geometric argument for why the one-moment
   approximation collapses under nonlinear measurement operators.

The two-moment Gaussian approximations — ΠiGDM (Song et al. 2023) and TMPD
(Boys et al. 2024) — are implemented in {py:class}`src.scores.PiGDM` and
{py:class}`src.scores.TMPD`; both are restricted to linear forward models.

```{toctree}
:hidden:
:maxdepth: 1

framework
one_moment_chung
finite_sample
failure_modes
```
