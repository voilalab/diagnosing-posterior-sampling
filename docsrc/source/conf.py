"""Sphinx configuration for the diagnosing-posterior-sampling documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the package importable when Sphinx runs.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "sphinxext"))

# ── Project metadata ──────────────────────────────────────────────────────────
project = "Diagnosing Posterior Sampling"
author = "Benjamin A. Burns and Sara Fridovich-Keil"
copyright = "2026-%Y, Benjamin A. Burns and Sara Fridovich-Keil"  # noqa: A001

# ── Extensions ────────────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "myst_nb",
    "sphinx_design",
    "sphinxcontrib.bibtex",
    "override_pst_pagetoc",
]

# Mock heavy runtime deps so docs build on machines without GPU torch.
autodoc_mock_imports = ["torch", "wandb"]

# ── Intersphinx ───────────────────────────────────────────────────────────────
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# ── BibTeX ────────────────────────────────────────────────────────────────────
bibtex_bibfiles = ["references.bib"]

# ── MyST Markdown extensions ──────────────────────────────────────────────────
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ["http", "https", "mailto"]

# ── Napoleon (Google-style docstrings) ────────────────────────────────────────
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

# ── autodoc / autosummary ─────────────────────────────────────────────────────
autosummary_ignore_module_all = False
autosummary_imported_members = False
autoclass_content = "both"
autosummary_generate = True
autodoc_member_order = "bysource"

coverage_show_missing_items = True

# ── MathJax ───────────────────────────────────────────────────────────────────
# Alias \bm{...} to \boldsymbol{...} so the docs can use the same bold-italic
# notation as overleaf/paper (which loads the LaTeX `bm` package). MathJax 3
# does not provide \bm by default.
mathjax3_config = {
    "tex": {
        "macros": {
            "bm": [r"\boldsymbol{#1}", 1],
        },
    },
}

# ── Notebooks ─────────────────────────────────────────────────────────────────
nb_execution_mode = "off"
html_sourcelink_suffix = ""

# ── Templates & static files ──────────────────────────────────────────────────
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]

exclude_patterns: list[str] = []

# Hide autosummary detail pages from the main TOC.
remove_from_toctrees = ["_autosummary/*"]

# Suppress label collisions that myst-nb creates from notebook headings.
_examples_dir = Path(__file__).resolve().parents[2] / "examples"
suppress_warnings = (
    [
        f"autosectionlabel._examples/{nb.stem}"
        for nb in _examples_dir.glob("*.ipynb")
    ]
    if _examples_dir.exists()
    else []
)

# ── HTML theme ────────────────────────────────────────────────────────────────
html_theme = "pydata_sphinx_theme"
html_title = "Diagnosing posterior sampling: a finite-sample lens"
html_show_sourcelink = False

# Logo and favicon are placeholders until branding ships; conf.py only sets
# them when the asset is actually present so a fresh clone builds clean.
_static_dir = Path(__file__).resolve().parent / "_static"
if (_static_dir / "logo_light.png").exists():
    html_logo = "_static/logo_light.png"
if (_static_dir / "favicon.ico").exists():
    html_favicon = "_static/favicon.ico"

html_theme_options = {
    "use_edit_page_button": True,
    "icon_links_label": "Icon Links",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/voilalab/diagnosing-posterior-sampling",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
    "navbar_align": "left",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
}

html_context = {
    "github_url": "https://github.com",
    "github_user": "voilalab",
    "github_repo": "diagnosing-posterior-sampling",
    "github_version": "main",
    "doc_path": "docsrc/source",
}

todo_include_todos = True

# Wire the dual light/dark logo only when both assets are present.
if (_static_dir / "logo_light.png").exists() and (_static_dir / "logo_dark.png").exists():
    html_theme_options["logo"] = {
        "alt-text": "Diagnosing Posterior Sampling",
        "image_light": "_static/logo_light.png",
        "image_dark": "_static/logo_dark.png",
    }

del os  # avoid leaking imports into autodoc namespace
