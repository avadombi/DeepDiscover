"""Sphinx documentation for DeepDiscover 1.0.1."""
project = 'DeepDiscover'
author = 'Vincent Adombi'
copyright = '2026, Vincent Adombi'
release = version = '1.0.1'
extensions = ['myst_parser', 'sphinx.ext.mathjax', 'sphinx.ext.githubpages']
source_suffix = {'.md': 'markdown'}
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
myst_enable_extensions = ['dollarmath', 'amsmath', 'colon_fence', 'deflist']
myst_heading_anchors = 3
numfig = False
math_number_all = False
math_eqref_format = "({number})"
html_theme = 'sphinx_rtd_theme'
html_title = 'DeepDiscover documentation'
html_logo = '_static/brand/deepdiscover_logo_sidebar.svg'
html_favicon = '_static/brand/deepdiscover_icon.svg'
html_static_path = ['_static']
html_css_files = ['theme-colors.css', 'custom.css']
html_theme_options = {
    'logo_only': True,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 3,
    'titles_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
}
