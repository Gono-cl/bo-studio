# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'BO-STUDIO'
copyright = '2025, Araya Vargas, Gonzalo'
author = 'Araya Vargas, Gonzalo'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_parser"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
html_sidebars = {
    '**': [
        'globaltoc.html',   # global menu
        'relations.html',   # previous/next links
        'searchbox.html',   # search box
    ]
}