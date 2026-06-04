# Sphinx Makefile for dps-collapse docs.

SPHINXOPTS  ?=
SPHINXBUILD ?= uv run --group docs sphinx-build
SOURCEDIR    = source
BUILDDIR     = _build

.PHONY: help local-docs clean view-docs coverage

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

local-docs:
	uv run --group docs python pre-build.py
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR)
	rm -rf $(SOURCEDIR)/_examples
	rm -rf $(SOURCEDIR)/api/_autosummary

view-docs:
	uv run --group docs python -m http.server 8090 --directory $(BUILDDIR)/html

coverage:
	$(SPHINXBUILD) -b coverage "$(SOURCEDIR)" "$(BUILDDIR)/coverage" $(SPHINXOPTS)
	@cat "$(BUILDDIR)/coverage/python.txt"
