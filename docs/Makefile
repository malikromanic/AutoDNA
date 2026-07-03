# Minimal makefile for Sphinx documentation.
#
#   make html      -> HTML site in build/html/index.html
#   make latexpdf  -> single PDF in build/latex/AutoDNA.pdf (needs a LaTeX install)

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all: route any unknown target (html, latexpdf, ...) to sphinx-build.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
