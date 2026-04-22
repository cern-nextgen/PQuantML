# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('../'))

project = 'PQuantML'
copyright = '2025, Roope Niemi'
author = 'Roope Niemi, Anastasiia Petrovych'
release = "1.0.0"
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

myst_enable_extensions = [
    "amsmath",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]


autosummary_generate = True

extensions = ['myst_parser', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'sphinx.ext.napoleon', 'sphinx_rtd_theme']

source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = ['_build']

html_logo = "_static/pquant.png"
html_theme_options = {
    'logo_only': True,
    'display_version': True,
}

html_context = {
    'display_github': True,  # Integrate GitHub
    'github_user': 'nroope',  # Username
    'github_repo': "PQuant",  # Repo name
    'github_version': 'master',  # Version
    'conf_py_path': '/docs/',  # Path in the checkout to the docs root
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_favicon = '_static/pquant.png'

html_css_files = [
    'custom.css',
]
