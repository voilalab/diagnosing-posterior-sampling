"""Finite-sample diagnostics for diffusion posterior samplers.

Official code for "When, why, and how do diffusion posterior samplers fail?
A finite-sample lens" (Burns & Fridovich-Keil, 2026, arXiv:2605.30330).

Public sub-packages:

- :mod:`src.forward_model` — measurement-operator abstraction.
- :mod:`src.sde` — stochastic differential equations (OU, VPSDE).
- :mod:`src.weights` — finite-sample mixture weights and prior-score helpers.
- :mod:`src.scores` — likelihood / posterior score variants and the dispatcher.
- :mod:`src.samplers` — discrete-time SDE samplers (Euler-Maruyama).
- :mod:`src.tweedie` — Tweedie moment helpers shared across posterior-score methods.
"""

__all__: list[str] = []
