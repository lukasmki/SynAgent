# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import dataclasses

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SynAgent"
copyright = "2026, Lukas Kim"
author = "Lukas Kim"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",  # Google-style "Args:" / "Returns:" docstrings
    "sphinx.ext.viewcode",
]

# Section labels are referenced as :ref:`<document>:<section title>`
autosectionlabel_prefix_document = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Autodoc -----------------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
# Show defaults as written (DEFAULT_SPEC), not as machine-specific absolute paths
autodoc_preserve_defaults = True
autodoc_default_options = {"show-inheritance": True}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "SynAgent"
html_favicon = "../assets/synagent-icon-32.png"
# Contents of assets/ are copied into _static/, where the furo logo options look
html_static_path = ["_static", "../assets"]
html_theme_options = {
    "light_logo": "synagent-lockup.png",
    "dark_logo": "synagent-lockup-inverted.png",
    "sidebar_hide_name": True,  # the lockup already shows the name
}


# -- Hooks -------------------------------------------------------------------


def _drop_generated_docstring(app, what, name, obj, options, lines):
    """Hide the docstring dataclasses generate for classes without one.

    It repeats the signature with evaluated defaults, such as absolute paths.
    """
    doc = getattr(obj, "__doc__", None) or ""
    if (
        what == "class"
        and dataclasses.is_dataclass(obj)
        and doc.startswith(f"{obj.__name__}(")
    ):
        lines.clear()


def setup(app):
    app.connect("autodoc-process-docstring", _drop_generated_docstring)
