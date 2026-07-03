"""Sphinx configuration for the AutoDNA application (app/) API docs."""

import sys
from pathlib import Path

# conf.py lives at AutoDNA/docs/source/conf.py.
# Modules import as "AutoDNA.app.xxx", so the directory that CONTAINS the
# AutoDNA repo (its parent) needs to be on sys.path — same setup app/main.py
# does at runtime.
_SOURCE_DIR = Path(__file__).resolve().parent
_AUTODNA_DIR = _SOURCE_DIR.parents[1]   # .../AutoDNA
_GITHUB_DIR = _AUTODNA_DIR.parent       # one level above AutoDNA/
sys.path.insert(0, str(_GITHUB_DIR))

project = "AutoDNA"
copyright = "2026, AutoDNA Project"
author = "AutoDNA Project"
release = "1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# The codebase mixes Google-style ("Args:"/"Returns:") and NumPy-style
# ("Returns\n-------") docstrings depending on who wrote them — accept both
# rather than rewriting working docstrings just to normalize style.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

autosummary_generate = False
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"

# Heavy / optional third-party deps that the research + device scripts import
# but that need not be installed just to build the docs. autodoc replaces them
# with stubs so every ai/ and stm32/ module can be imported for its docstrings.
# (numpy is intentionally NOT mocked — it is a hard dependency used everywhere.)
autodoc_mock_imports = [
    "torch",
    "xgboost",
    "sklearn",
    "scipy",
    "serial",
    "matplotlib",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]

# -- Single-PDF (LaTeX) output ----------------------------------------------
# `sphinx-build -b latexpdf docs/source docs/build/latex` produces ONE
# AutoDNA.pdf. The "manual" documentclass turns each top-level page in the
# index toctree (App Modules, AI, STM32, Tests) into its own chapter, so the
# PDF is a single document split by module — mirroring the HTML structure.
latex_engine = "pdflatex"
latex_documents = [
    (
        "index",                            # start (master) doc
        "AutoDNA.tex",                      # target name -> yields AutoDNA.pdf
        "AutoDNA — Project API Reference",  # title
        "AutoDNA Project",                  # author
        "manual",                           # documentclass: chapters per page
    ),
]
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    # Keep long autodoc signatures from overflowing the page margin.
    "preamble": r"\setlength{\emergencystretch}{3em}",
}
