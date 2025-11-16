# Configuration file for the Sphinx documentation builder.

project = 'py_sphyxer'
author = 'Ptolemy'
release = '1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))
