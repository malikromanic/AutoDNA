"""Sphinx configuration for the AutoDNA application (app/) API docs."""

import sys
from pathlib import Path

# conf.py lives at AutoDNA/app/docs_sphinx/source/conf.py.
# Modules import as "AutoDNA.app.xxx", so the directory that CONTAINS the
# AutoDNA repo (its parent) needs to be on sys.path — same setup app/main.py
# does at runtime.
_SOURCE_DIR = Path(__file__).resolve().parent
_AUTODNA_DIR = _SOURCE_DIR.parents[2]   # .../AutoDNA
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

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]
