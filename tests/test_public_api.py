"""Smoke test that every name in every ``__all__`` resolves to an attribute.

Catches typos in ``__all__`` lists and broken re-exports across the public
surface of :mod:`src`. Sphinx autosummary depends on this surface being
internally consistent.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

PACKAGES = ("src",)


def _walk(pkg_name: str) -> list[str]:
    pkg = importlib.import_module(pkg_name)
    found = [pkg_name]
    if hasattr(pkg, "__path__"):
        for info in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg_name}."):
            found.append(info.name)
    return found


@pytest.mark.parametrize("module_name", [m for p in PACKAGES for m in _walk(p)])
def test_all_public_names_resolve(module_name: str) -> None:
    """Every name listed in ``module.__all__`` must be a real attribute."""
    module = importlib.import_module(module_name)
    declared = getattr(module, "__all__", None)
    if declared is None:
        return
    for name in declared:
        assert hasattr(module, name), f"{module_name}.__all__ lists {name!r} which is missing"
