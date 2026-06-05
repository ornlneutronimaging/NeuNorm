"""Sphinx configuration for the NeuNorm documentation build."""

try:
    from neunorm import __version__ as _version
except ImportError:  # pragma: no cover - docs can still build without an install
    _version = "unknown"

# -- Project information -----------------------------------------------------
project = "NeuNorm"
copyright = "2024, Oak Ridge National Laboratory"  # noqa: A001 - Sphinx requires this name
author = "Jean Bilheux, Chen Zhang"
release = _version
version = _version

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The workflow guides embed mermaid diagrams, for which Pygments has no lexer.
# Rendering them as diagrams is a documentation fast-follow; until then this keeps
# the build clean rather than emitting a highlight warning per diagram.
suppress_warnings = ["misc.highlighting_failure"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Autodoc / Napoleon ------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# -- MyST (Markdown) ---------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist"]

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "scipp": ("https://scipp.github.io", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_title = f"NeuNorm {version} documentation"
