"""Flatten the right-hand in-page TOC for auto-generated API pages.

Adapted from scikit-learn's ``doc/sphinxext/override_pst_pagetoc.py`` (BSD-3).

``sphinx`` is only available when running inside a docs build, so its imports
live inside :func:`setup` rather than at module top level — that keeps
``ty check`` happy on environments without the ``[docs]`` dependency group
installed.
"""

from __future__ import annotations

from functools import cache
from typing import Any


def _build_callback() -> Any:
    """Construct the ``html-page-context`` callback with sphinx imported lazily."""
    # sphinx is in the [docs] dependency group only, so import it
    # dynamically — ty does not see it in the default environment.
    import importlib

    logging_mod = importlib.import_module("sphinx.util.logging")
    logger = logging_mod.getLogger(__name__)

    def override_pst_pagetoc(app, pagename, templatename, context, doctree):
        del app, templatename, doctree

        @cache
        def generate_api_toc_html(kind: str = "html") -> str:
            soup = context["pst_generate_toc_html"](kind="soup")
            try:
                soup.ul.unwrap()
                soup.li.unwrap()
                soup.a.decompose()

                lis = soup.ul.select("li.toc-h2")
                main_li = lis[0]
                meth_list = main_li.ul

                if meth_list is not None:
                    meth_list["class"].append("visible")
                    for meth in meth_list.find_all("li", {"class": "toc-h3"}):
                        target = meth.a.code.span
                        target.string = target.string.split(".", 1)[1]

                return str(soup) if kind == "html" else soup
            except (AttributeError, IndexError, KeyError, TypeError) as exc:
                logger.warning("Failed to generate API pagetoc for %s: %s", pagename, exc)
                return context["pst_generate_toc_html"](kind=kind)

        if pagename.startswith("api/"):
            context["pst_generate_toc_html"] = context["generate_toc_html"]
            context["generate_toc_html"] = generate_api_toc_html

    return override_pst_pagetoc


def setup(app: Any) -> dict[str, Any]:
    """Register the page-context hook after the pydata theme installs its own."""
    app.connect("html-page-context", _build_callback(), priority=900)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
