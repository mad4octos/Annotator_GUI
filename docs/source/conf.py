project = "Annotator GUI"

master_doc = 'index'

extensions = ['myst_parser', 'sphinx_copybutton', 'sphinx_design', 'sphinx.ext.extlinks', 'sphinx.ext.autodoc']
myst_enable_extensions = ["colon_fence", "attrs_inline", "attrs_block", "tasklist", "substitution"]
myst_enable_checkboxes = True

source_suffix = {
    '.txt': 'markdown',
    '.md': 'markdown',
}

html_theme = 'sphinx_book_theme'

myst_heading_anchors=6

html_static_path = ['_static']
html_css_files = ["custom.css"]

html_theme_options = {
   "repository_url": "https://github.com/mad4octos/Annotator_GUI",
   "use_repository_button": True,
}

html_context = {
   "default_mode": "dark"
}
