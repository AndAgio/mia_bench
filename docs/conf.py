"""Sphinx configuration of the mia_bench documentation.

Build it from the repository root with:
    pip install -r docs/requirements.txt
    sphinx-build -W --keep-going -b html docs docs/_build/html
"""
import pathlib
import sys

DOCS = pathlib.Path(__file__).resolve().parent
REPO = DOCS.parent
# The command-line reference and the generated tables import the package, which needs its runtime dependencies.
sys.path.insert(0, str(REPO))

project = 'mia_bench'
author = 'the mia_bench contributors'
copyright = '2026, the mia_bench contributors'

extensions = ['myst_parser', 'autoapi.extension', 'sphinxarg.ext']
myst_heading_anchors = 3
exclude_patterns = ['_build', '_generated']

html_theme = 'furo'
html_title = 'mia_bench'

# The API reference is read from the source without importing it. Its page lists the classes to show
# (reference/api.md), so no page is generated per module, and members without a docstring are left out.
autoapi_dirs = [str(REPO / 'src')]
# Several folders of src have no __init__.py and are imported as namespace packages.
autoapi_python_use_implicit_namespaces = True
autoapi_generate_api_docs = False
autoapi_options = ['members', 'show-inheritance']
# Static analysis cannot follow some re-exports (e.g. torch's optimizers through src.optimizers); they are not errors.
suppress_warnings = ['autoapi.python_import_resolution']

# autoapi gives a class without a docstring the docstring of its closest ancestor, e.g. torch.nn.Module's, which
# describes a different class. Show only the class's own docstring. The check fails the build if autoapi renames the
# function, instead of silently falling back again.
import autoapi._astroid_utils
assert hasattr(autoapi._astroid_utils, 'get_class_docstring'), 'autoapi changed: update the docstring patch in conf.py'
autoapi._astroid_utils.get_class_docstring = lambda node: node.doc_node.value if node.doc_node else ''


def skip_undocumented_members(app, what, name, obj, skip, options):
    # Members without a docstring, such as instance attributes, are left out of the listed classes. The directives of
    # api.md go through autodoc, which would count the docstring of autoapi's own wrapper object as the member's.
    docstring = getattr(obj, 'docstring', None)
    return skip or docstring == ''


def setup(app):
    app.connect('autodoc-skip-member', skip_undocumented_members)


# ---- Tables generated from the code, so that the pages cannot drift from it ----
DATASET_NAMES = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100', 'purchase': 'Purchase', 'texas': 'Texas'}


def write_generated(name: str, text: str):
    # Pages include these files with {include}; they are rebuilt on every build and not committed.
    folder = DOCS / '_generated'
    folder.mkdir(exist_ok=True)
    (folder / name).write_text(text)


def hamp_parameters():
    from src.utils.variables import HAMP_PAPER_PARAMETERS, HAMP_FALLBACK_DATASET
    rows = [f'| {DATASET_NAMES.get(dataset, dataset)} | {gamma} | {alpha} |' for dataset, (gamma, alpha) in HAMP_PAPER_PARAMETERS.items()]
    fallback = DATASET_NAMES.get(HAMP_FALLBACK_DATASET, HAMP_FALLBACK_DATASET)
    return '\n'.join(['| Dataset | gamma | alpha |', '|---|---|---|', *rows, '',
                      f'Other datasets use the {fallback} values.', ''])


write_generated('hamp_parameters.md', hamp_parameters())
