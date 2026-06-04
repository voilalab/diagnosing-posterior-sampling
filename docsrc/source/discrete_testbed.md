# Examples and the discrete testbed

The shipped examples validate the finite-sample reference and the
approximations against priors whose posteriors are available in closed form.
The discrete prior is the natural validation surface, so we start there.

## Why discrete

Take the prior to be a discrete measure
$\pi = \sum_i w_i \delta_{x_i}$ on a small atom set
$\{x_i\}_{i=1}^{K}$. The conditional marginal at diffusion time $t$ is then a
Gaussian mixture, and the posterior weights
$\psi_i \propto w_i\, \mathcal N(y;\, \mathcal A(x_i), \Sigma_n)$ depend on the
observation but factor cleanly. Posterior densities, marginal densities, and
their scores are therefore available in **closed form for both linear and
nonlinear** forward operators $\mathcal A$ — the discrete prior is the one
setting where the oracle can be written down without integrating against a
continuous distribution. {py:class}`src.distributions.discrete.Discrete`
implements these closed forms; {py:class}`src.distributions.gaussian.Gaussian`
and {py:class}`src.distributions.gmm.GMM` add the unimodal- and
mixture-Gaussian priors (closed-form under affine $\mathcal A$).

## Shipped scripts

Run each from the repository root with `uv run python -m examples.<name>`:

| Script | What it shows |
| --- | --- |
| `quickstart.py` | The five-line {py:func}`src.fsr.run_fsr` call with a posterior histogram. |
| `discrete_example.py` | FSR posterior samples overlaid on the closed-form `Discrete.posterior_density`. |
| `gaussian_example.py` | The same overlay for a 1-D Gaussian prior. |
| `gmm_example.py` | The same overlay for a two-component Gaussian mixture. |
| `compare_methods.py` | Posterior-density overlay of FSR vs σ-DPS vs ζ-DPS vs ΠiGDM vs TMPD on a discrete testbed — the per-method comparison from the paper. |
| `heatmap_example.py` | A paper-style $(t, x)$ posterior-density heatmap built from `Discrete.posterior_density`. |

Each script writes a single PDF next to itself (gitignored).

## What the examples import

The examples use only the public surface:

- {py:func}`src.fsr.run_fsr` and {py:func}`src.scores.make_score_fn`
- {py:class}`src.forward_model.AffineForwardModel`
- {py:class}`src.distributions.discrete.Discrete` (and `Gaussian` / `GMM`)
- {py:class}`src.sde.VPSDE`
- the shared helpers in `examples/_common/`
  (`plotting`, `sampling`, `heatmap`).

## Validation methodology

The examples are visual overlays of each score variant against the closed-form
oracle; the same closed forms back the automated checks in `tests/`. In
particular {py:func}`src.fsr.run_fsr` is regression-tested against the analytic
Gaussian–linear posterior in `tests/test_fsr_api.py`, and the score
approximations are dispatch- and shape-checked in `tests/test_scores_factory.py`
and `tests/test_scores.py`.
