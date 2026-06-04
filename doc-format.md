# Converting a Sphinx Codebase to the BayesFlow Docs Format

This document describes the full documentation setup used by the BayesFlow project
and serves as a step-by-step conversion guide for adapting any Sphinx-based (or
docstring-annotated) Python project to match this style.

The result is a professional, modern documentation site with:
- Light/dark mode toggle
- Multi-version switcher in the navbar
- Clean API reference auto-generated from Google-style docstrings
- Integrated Jupyter notebooks (rendered, not executed)
- Responsive Bootstrap layout with a left-sidebar + right in-page TOC
- Cross-referencing to NumPy, SciPy, Matplotlib, etc.

---

## 1. Visual Theme: pydata-sphinx-theme

The entire visual identity comes from **[pydata-sphinx-theme](https://pydata-sphinx-theme.readthedocs.io/)**.
This is the same theme used by NumPy, SciPy, Pandas, and scikit-learn.

Install it alongside the rest of the doc dependencies:

```toml
# pyproject.toml  [project.optional-dependencies]
docs = [
    "sphinx>=8.1.3",
    "pydata-sphinx-theme>=0.16.1",
    # sphinx.ext.napoleon is bundled with Sphinx — no extra package needed
    "myst-nb>=1.3.0",
    "sphinx-design>=0.6.1",
    "sphinxcontrib-bibtex>=2.6.5",
    "snowballstemmer>=3.0.1",
]
```

Install with:

```bash
pip install -e ".[docs]"
```

---

## 2. Repository / Directory Layout

```
<project-root>/
├── pyproject.toml
├── <package>/          # your Python package
└── docsrc/
    ├── Makefile
    ├── source/
    │   ├── conf.py
    │   ├── index.md         # landing page (MyST Markdown)
    │   ├── examples.rst     # toctree pointing at example notebooks
    │   ├── references.bib   # BibTeX file (can be empty)
    │   ├── api/
    │   │   └── <package>.rst
    │   ├── _templates/
    │   │   ├── base.rst
    │   │   └── custom-module-template.rst
    │   ├── _static/
    │   │   ├── custom.css
    │   │   ├── logo_light.png
    │   │   ├── logo_dark.png
    │   │   └── favicon.ico
    │   └── sphinxext/
    │       └── override_pst_pagetoc.py
```

---

## 3. `conf.py` — Annotated Full Configuration

Create `docsrc/source/conf.py`:

```python
import os
import sys

# Make the package importable when Sphinx runs
sys.path.insert(0, os.path.abspath("../.."))
# Make local extensions importable
sys.path.insert(0, os.path.abspath("sphinxext"))

# ── Project metadata ──────────────────────────────────────────────────────────
project   = "YourProject"
author    = "The YourProject authors"
copyright = "2024-%Y, YourProject authors"

# ── Extensions ────────────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",       # pull docstrings into RST
    "sphinx.ext.napoleon",      # parse Google-style docstrings (bundled with Sphinx)
    "sphinx.ext.autosummary",   # generate per-object summary pages
    "sphinx.ext.todo",
    "sphinx.ext.coverage",      # warn on undocumented public API
    "sphinx.ext.viewcode",      # "source" links on every API page
    "myst_nb",                  # MyST Markdown + Jupyter notebooks
    "sphinx.ext.extlinks",      # shorthand external link roles
    "sphinx.ext.intersphinx",   # cross-ref NumPy / SciPy / etc.
    "sphinx_design",            # Bootstrap grid cards, tabs, badges
    "sphinxcontrib.bibtex",     # BibTeX citations
    "override_pst_pagetoc",     # local — see §8 below
]

# ── Intersphinx: cross-reference external docs ────────────────────────────────
intersphinx_mapping = {
    "python":     ("https://docs.python.org/3", None),
    "numpy":      ("https://numpy.org/doc/stable", None),
    "scipy":      ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas":     ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

# ── BibTeX ────────────────────────────────────────────────────────────────────
bibtex_bibfiles = ["references.bib"]

# ── MyST Markdown extensions ──────────────────────────────────────────────────
myst_enable_extensions = [
    "amsmath",       # \begin{equation} blocks
    "colon_fence",   # :::{directive} syntax
    "deflist",       # definition lists
    "dollarmath",    # $...$ and $$...$$ math
    "html_image",    # <img> tags in Markdown
]
myst_url_schemes = ["http", "https", "mailto"]

# ── napoleon (Google-style docstrings) ───────────────────────────────────────
napoleon_google_docstring = True
napoleon_numpy_docstring  = False  # disable NumPy parsing to avoid ambiguity
napoleon_use_param        = True   # render Args as a definition list
napoleon_use_rtype        = True   # render return type on its own line

# ── autodoc / autosummary ─────────────────────────────────────────────────────
autosummary_ignore_module_all = False  # honour __all__ in each module
autosummary_imported_members  = False  # don't document re-exports
autoclass_content             = "both" # class docstring + __init__ docstring
autosummmary_generate         = True

coverage_show_missing_items = True

# ── Notebooks ─────────────────────────────────────────────────────────────────
nb_execution_mode      = "off"  # do NOT re-run notebooks; use pre-executed
html_sourcelink_suffix = ""     # download link shows .ipynb not .ipynb.txt

# ── Templates & static files ──────────────────────────────────────────────────
templates_path   = ["_templates"]
html_static_path = ["_static"]
html_css_files   = ["custom.css"]

exclude_patterns = []

# ── Hide autosummary pages from main TOC ──────────────────────────────────────
remove_from_toctrees = ["_autosummary/*"]

# ── HTML theme ────────────────────────────────────────────────────────────────
html_theme      = "pydata_sphinx_theme"
html_title      = "YourProject: A One-Line Description"
html_logo       = "_static/logo_light.png"
html_favicon    = "_static/favicon.ico"
html_baseurl    = "https://yourproject.org/"
html_show_sourcelink = False

html_theme_options = {
    # "Edit this page" button linking to GitHub
    "use_edit_page_button": True,

    # Light / dark logo pair
    "logo": {
        "alt-text": "YourProject",
        "image_light": "_static/logo_light.png",
        "image_dark":  "_static/logo_dark.png",
    },

    # Icon links in the navbar (FontAwesome icons)
    "icon_links_label": "Icon Links",
    "icon_links": [
        {
            "name": "GitHub",
            "url":  "https://github.com/yourorg/yourproject",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],

    # Navbar layout
    "navbar_align":      "left",
    "navbar_start":      ["navbar-logo"],
    "navbar_center":     ["navbar-nav"],
    "navbar_end":        ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],   # always visible on mobile
}

# Required for the "Edit this page" button
html_context = {
    "github_url":     "https://github.com",
    "github_user":    "yourorg",
    "github_repo":    "yourproject",
    "github_version": "main",
    "doc_path":       "docsrc/source",
}

todo_include_todos = True
```

> **Version switcher (optional):** If you want a version dropdown in the navbar
> (as BayesFlow has), add `"navbar_end": ["theme-switcher", "navbar-icon-links", "version-switcher"]`
> and provide a `switcher` dict pointing at a `versions.json` file.
> See §9 for the multi-version build system.

---

## 4. Docstring Format: Google Style

All public API uses **Google-style docstrings**, parsed by `sphinx.ext.napoleon`
(bundled with Sphinx — no extra install needed).

### Functions

```python
def my_function(x, y, method="exact"):
    """Short one-line summary (imperative mood, no trailing period).

    Longer description if needed.  Can span multiple paragraphs.
    Supports reStructuredText markup and ``inline code``.

    Args:
        x (array-like of shape (n,)): Description of x.
        y (float): Description of y.
        method ({"exact", "approximate"}, optional): Which algorithm to use.
            Default is ``"exact"``.

    Returns:
        ndarray of shape (n,): Description of the return value.

    Raises:
        ValueError: If *x* is empty.

    Note:
        Use :math:`\\alpha` for inline LaTeX.

    Example:
        >>> import yourproject as yp
        >>> yp.my_function([1, 2, 3], y=0.5)
        array([...])
    """
```

### Classes

Put the docstring on the class, not `__init__`.
`autoclass_content = "both"` in `conf.py` merges both into one rendered page.

```python
class MyModel:
    """Short class description.

    Args:
        units (int): Number of hidden units.
        activation (str, optional): Activation function name.
            Default is ``"relu"``.

    Example:
        >>> model = MyModel(units=64)
        >>> model(inputs)
    """
    def __init__(self, units, activation="relu"):
        ...
```

### All supported section headers

Napoleon recognises these Google-style section names:

| Section | Use for |
|---|---|
| `Args:` | Function / method parameters |
| `Returns:` | Return value(s) |
| `Yields:` | Generator yield values |
| `Raises:` | Exceptions that may be thrown |
| `Attributes:` | Class-level attributes |
| `Note:` / `Notes:` | Implementation notes |
| `Warning:` / `Warnings:` | Caution for callers |
| `Example:` / `Examples:` | Usage examples (doctest-compatible) |
| `See Also:` | Related functions or classes |
| `References:` | Citations |
| `Todo:` | Future work (rendered if `todo_include_todos = True`) |

### `__all__`

Define `__all__` in every public module.
`autosummary_ignore_module_all = False` means only names listed there are documented.

```python
# yourpackage/networks/__init__.py
from .mlp import MLP
from .transformer import Transformer

__all__ = ["MLP", "Transformer"]
```

---

## 5. API Reference Auto-Generation

### 5a. Entry-point RST (`api/<package>.rst`)

```rst
API Reference
=============

This page will be overridden by the autosummary call for <package>.
To modify the actual output, refer to _templates/custom-module-template.rst.

.. autosummary::
    :toctree: .
    :template: custom-module-template.rst
    :recursive:

    yourpackage
```

This single directive recursively walks `yourpackage`, generating one page per
module/class/function, respecting each module's `__all__`.

### 5b. Module template (`_templates/custom-module-template.rst`)

```rst
{% if objname == 'yourpackage' %}
API Reference
=============

This is the reference for the public API.

{% else %}
{{ objname | escape | underline}}
{% endif %}

.. automodule:: {{ fullname }}
  :member-order: alphabetical

  {% block modules %}
  {% if modules %}
  .. rubric:: Modules

  .. autosummary::
    :toctree:
    :template: custom-module-template.rst
    :recursive:
  {% for item in modules %}
    {{ item }}
  {%- endfor %}

  {% endif %}
  {% endblock %}

  {% block attributes %}
  {% if attributes %}
  .. rubric:: Module Attributes

  .. autosummary::
  {% for item in attributes %}
    {{ item }}
  {%- endfor %}
  {% endif %}
  {% endblock %}

  {% block functions %}
  {% if functions %}
  .. rubric:: {{ _('Functions') }}

  .. autosummary::
    :toctree:
    :template: base.rst
  {% for item in functions %}
    {{ item }}
  {%- endfor %}
  {% endif %}
  {% endblock %}

  {% block classes %}
  {% if classes %}
  .. rubric:: {{ _('Classes') }}

  .. autosummary::
    :toctree:
    :template: base.rst
  {% for item in classes %}
    {{ item }}
  {%- endfor %}
  {% endif %}
  {% endblock %}

  {% block exceptions %}
  {% if exceptions %}
  .. rubric:: {{ _('Exceptions') }}

  .. autosummary::
    :toctree:
    :template: base.rst
  {% for item in exceptions %}
    {{ item }}
  {%- endfor %}
  {% endif %}
  {% endblock %}
```

### 5c. Per-object template (`_templates/base.rst`)

```rst
{{ objname | escape | underline(line="=") }}

{% if objtype == "function" -%}

.. currentmodule:: {{ module }}

.. autofunction:: {{ objname }}

{%- elif objtype == "class" -%}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __call__
   :member-order: bysource
   :undoc-members:

{%- else -%}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}

{%- endif -%}
```

Key choices here:
- `:inherited-members:` — shows inherited methods (important for subclassed models).
- `:special-members: __call__` — documents `__call__` as a named method.
- `:member-order: bysource` — preserves the order methods appear in the file.

---

## 6. Landing Page (`index.md`)

Use **MyST Markdown** for the landing page.  The hidden toctree at the bottom
drives the left-sidebar navigation.

```markdown
# YourProject

One-sentence pitch.

## Install

\```bash
pip install yourpackage
\```

## Getting Started

\```python
import yourpackage as yp
# minimal example
\```

## Indices

* {ref}`genindex`
* {ref}`modindex`


\```{toctree}
:maxdepth: 0
:titlesonly:
:hidden:

examples
api/yourpackage
about
Contributing <contributing>
\```
```

The `:hidden:` flag means the toctree is **not** rendered inline on the page but
**is** used to build the sidebar.  `:titlesonly:` shows only the top-level title
of each linked document in the sidebar.

---

## 7. Jupyter Notebook Integration (myst-nb)

`myst-nb` renders notebooks as static HTML without re-executing them.

### Workflow

1. Store raw notebooks in `examples/` (project root) or wherever suits your repo.
2. In a pre-build script (`pre-build.py`), copy them into `docsrc/source/_examples/`:

```python
# docsrc/pre-build.py
import shutil, os

SRC = "../../examples"
DST = "_examples"        # relative to docsrc/source/

os.makedirs(DST, exist_ok=True)
for f in os.listdir(SRC):
    if f.endswith(".ipynb"):
        shutil.copy(os.path.join(SRC, f), os.path.join(DST, f))
```

Run this before Sphinx: `python pre-build.py` (or wire it into the Makefile).

3. Create `docsrc/source/examples.rst` to list them:

```rst
Examples
========

.. toctree::
   :maxdepth: 1
   :glob:

   _examples/*
```

4. Notebooks are then cross-referenced from anywhere with:

```markdown
{doc}`Descriptive title <_examples/My_Notebook>`
```

### conf.py settings already shown above

```python
nb_execution_mode      = "off"   # never re-run; use pre-executed outputs
html_sourcelink_suffix = ""      # download = .ipynb, not .ipynb.txt
```

---

## 8. Custom Sphinx Extension: API Sidebar Cleanup (`override_pst_pagetoc.py`)

The pydata theme generates a right-hand "in-page TOC" for every page.
On API pages the default output nests items under the class name, which is noisy.
This extension (adapted from scikit-learn) flattens it.

Create `docsrc/source/sphinxext/override_pst_pagetoc.py`:

```python
# Adapted from https://github.com/scikit-learn/scikit-learn
# doc/sphinxext/override_pst_pagetoc.py (BSD 3-Clause)

from functools import cache
from sphinx.util.logging import getLogger

logger = getLogger(__name__)


def override_pst_pagetoc(app, pagename, templatename, context, doctree):
    """Flattens the in-page TOC for auto-generated API pages."""

    @cache
    def generate_api_toc_html(kind="html"):
        soup = context["pst_generate_toc_html"](kind="soup")
        try:
            soup.ul.unwrap()
            soup.li.unwrap()
            soup.a.decompose()

            lis = soup.ul.select("li.toc-h2")
            main_li = lis[0]
            meth_list = main_li.ul

            if meth_list is not None:
                # Always show methods; strip the class-name prefix
                meth_list["class"].append("visible")
                for meth in meth_list.find_all("li", {"class": "toc-h3"}):
                    target = meth.a.code.span
                    target.string = target.string.split(".", 1)[1]

            return str(soup) if kind == "html" else soup
        except Exception as e:
            logger.warning(f"Failed to generate API pagetoc for {pagename}: {e}")
            return context["pst_generate_toc_html"](kind=kind)

    if pagename.startswith("api/"):
        context["pst_generate_toc_html"] = context["generate_toc_html"]
        context["generate_toc_html"]      = generate_api_toc_html


def setup(app):
    # Must run after pydata_sphinx_theme (priority 500), so use 900
    app.connect("html-page-context", override_pst_pagetoc, priority=900)
```

This extension requires `beautifulsoup4` (pulled in transitively by
pydata-sphinx-theme).  Add `"override_pst_pagetoc"` to `extensions` in `conf.py`.

---

## 9. Custom CSS (`_static/custom.css`)

Minimal CSS to polish the sidebar version widget and ensure PNG logos have
transparent backgrounds:

```css
/* Version caption in sidebar */
.sidebar-primary-item .rst-versions p.caption {
  margin-bottom: 0;
  padding-left: 0.65rem;
  font-weight: 600;
}
.sidebar-primary-item .rst-versions ul {
  display: block;
  list-style: none;
  margin-bottom: 0.2rem;
}
.sidebar-primary-item .rst-versions li {
  display: list-item;
  text-align: match-parent;
}

/* Highlight the currently displayed version */
.sidebar-primary-item .rst-versions .current {
  color: var(--pst-color-accent);
  font-weight: 600;
}

.bd-sidebar-primary div#rtd-footer-container {
  margin: 0;
}

/* Transparent background for PNG logos in both light and dark modes */
img[src$=".png"] {
  background-color: transparent !important;
}
```

---

## 10. Makefile

```makefile
# docsrc/Makefile

SPHINXOPTS  ?=
SPHINXBUILD ?= sphinx-build
SOURCEDIR    = source
BUILDDIR     = _build

.PHONY: help local-docs clean view-docs

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

local-docs:
	cd $(SOURCEDIR) && python ../../docsrc/pre-build.py
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR)

view-docs:
	python -m http.server 8090 --directory $(BUILDDIR)/html
```

Build with:

```bash
cd docsrc
make local-docs
make view-docs   # then open http://localhost:8090
```

---

## 11. Multi-Version Builds (Optional, Advanced)

BayesFlow uses **[sphinx-polyversion](https://github.com/real-yfprojects/sphinx-polyversion)**
to build and host multiple versions (branches/tags) simultaneously.

Add to dependencies:

```toml
"sphinx-polyversion>=2.0.0",
```

Create `docsrc/poly.py` (the polyversion orchestration script):

```python
from sphinx_polyversion import *
from sphinx_polyversion.git import *
from sphinx_polyversion.pyvenv import Poetry

root  = Git.root(Path("."))
src   = "docsrc/source"
out   = "docs"

# Which branches and tags to build
branches = GitBranchSelector(r"^(main|stable)$")
tags     = GitTagSelector(r"^v\d+\.\d+\.\d+$")

# One virtual-env per version (cached in .docs_venvs/)
env = VenvSuffix(
    name       = ".docs_venvs",
    install    = ["pip install -e '.[docs]'"],
)

app = apply(
    root,
    src,
    out,
    refs        = branches | tags,
    environment = env,
    template    = "polyversion/templates",
    static      = "polyversion/static",
)
```

Run with:

```bash
python poly.py
```

Output lands in `docs/` with one subdirectory per version.  Host with GitHub Pages
by setting Pages source to the `docs/` folder on `main`.

Add a `versions.json` to the repo root (generated by polyversion or manually):

```json
[
  {"name": "main",    "version": "main",    "url": "/main/"},
  {"name": "v1.0.0",  "version": "v1.0.0",  "url": "/v1.0.0/"}
]
```

Then add to `conf.py`:

```python
html_theme_options["navbar_end"] = [
    "theme-switcher", "navbar-icon-links", "version-switcher"
]
html_theme_options["switcher"] = {
    "json_url":      "/versions.json",
    "version_match": current,   # set dynamically by polyversion
}
html_theme_options["check_switcher"] = False
```

---

## 12. Handling Mixed Docstring Formats (e.g., Keras Inheritance)

If your codebase inherits from a library that uses Markdown code fences in
its docstrings (e.g., Keras), those fences will break RST rendering.
Add a lightweight `autodoc-process-docstring` hook to convert them:

`docsrc/source/sphinxext/adapt_autodoc_docstring.py`:

```python
def docstring(app, what, name, obj, options, lines):
    """Convert Markdown fenced code blocks to RST code blocks."""
    updated_lines = []
    prefix = ""
    for line in lines:
        if line.count("```") == 1:
            if prefix == "":
                prefix = "    "
                updated_lines[-1] = updated_lines[-1] + "::"
            else:
                prefix = ""
            updated_lines.append("\n")
        else:
            updated_lines.append(prefix + line)
    if prefix != "":
        raise ValueError(
            f"Unmatched code fence in docstring: what='{what}', name='{name}'"
        )
    lines.clear()
    lines += updated_lines


def setup(app):
    app.connect("autodoc-process-docstring", docstring)
```

Add `"adapt_autodoc_docstring"` to `extensions` in `conf.py`.

---

## 13. Suppressing Duplicate Label Warnings from Notebooks

Jupyter notebooks auto-create section labels that can collide.  Suppress the
warnings by adding to `conf.py`:

```python
import os

suppress_warnings = [
    f"autosectionlabel._examples/{f.split('.')[0]}"
    for f in os.listdir("../../examples")
    if os.path.isfile(os.path.join("../../examples", f))
]
```

Adjust the path to wherever your notebooks live.

---

## 14. Quick-Start Checklist

Use this as a conversion checklist when migrating an existing project:

- [ ] Install `pydata-sphinx-theme>=0.16.1` and update `pyproject.toml`
- [ ] Set `html_theme = "pydata_sphinx_theme"` in `conf.py`
- [ ] Configure `html_theme_options` with logo pair, icon links, navbar layout
- [ ] Add `sphinx.ext.napoleon`, `myst_nb`, `sphinx_design`, `sphinxcontrib.bibtex` to extensions
- [ ] Set `napoleon_google_docstring = True` and `napoleon_numpy_docstring = False`
- [ ] Add `intersphinx_mapping` for NumPy, SciPy, etc.
- [ ] Convert all public docstrings to Google style
- [ ] Add `__all__` to every public module
- [ ] Create `api/<package>.rst` with single recursive `autosummary` directive
- [ ] Add `_templates/custom-module-template.rst` and `_templates/base.rst`
- [ ] Add `sphinxext/override_pst_pagetoc.py` and register it in extensions
- [ ] Add `_static/custom.css` with PNG transparency fix
- [ ] Set `nb_execution_mode = "off"` and add pre-build notebook copy script
- [ ] Create MyST Markdown landing page `index.md` with hidden toctree
- [ ] Add `remove_from_toctrees = ["_autosummary/*"]` to keep sidebar clean
- [ ] Test build: `cd docsrc && make local-docs && make view-docs`

---

## 15. Key Dependencies Reference

| Package | Minimum | Role |
|---|---|---|
| `sphinx` | 8.1.3 | Core doc builder |
| `pydata-sphinx-theme` | 0.16.1 | Visual theme |
| `sphinx.ext.napoleon` | bundled | Google-style docstring parser (no install needed) |
| `myst-nb` | 1.3.0 | MyST Markdown + Jupyter notebooks |
| `sphinx-design` | 0.6.1 | Bootstrap components (cards, tabs, badges) |
| `sphinxcontrib-bibtex` | 2.6.5 | BibTeX citation support |
| `snowballstemmer` | 3.0.1 | Search stemming |
| `sphinx-polyversion` | 2.0.0 | Multi-version builds (optional) |
