import os
import sphinx.ext.autodoc
import sys

conf_file_abs_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(conf_file_abs_path, ".."))

project = 'DComEX Framework'
copyright = '2022, NTUA, ETHZ, CSCS'
author = 'NTUA, ETHZ, CSCS'

extensions = [
    'sphinx.ext.mathjax', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary',
    'sphinx_automodapi.automodapi'
]

html_theme_options = {
  "logo": {
      "image_light": "logo.png",
      "image_dark": "logo.png",
  },
  "github_url": "https://github.com/DComEX/dcomex-prototype",
  "collapse_navigation": True,
}


html_title = project
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_favicon = '_static/favicon.ico'
html_context = {"default_mode": "dark"}

latex_documents = [
    ('index', 'dcomex.tex', u'DComEX Documentation',
     author, 'howto'),
]

autoclass_content = 'both'
