"""Sphinx configuration of the mia_bench documentation.

Build it from the repository root with:
    pip install -r docs/requirements.txt
    sphinx-build -W --keep-going -b html docs docs/_build/html
"""
import ast
import pathlib
import shutil
import sys

DOCS = pathlib.Path(__file__).resolve().parent
REPO = DOCS.parent
# The command-line reference and the generated tables import the package, which needs its runtime dependencies.
sys.path.insert(0, str(REPO))

project = 'mia_bench'
author = 'the mia_bench contributors'
copyright = '2026, the mia_bench contributors'

# napoleon renders the Args:, Returns: and Note: sections that some docstrings use.
extensions = ['myst_parser', 'autoapi.extension', 'sphinxarg.ext', 'sphinx.ext.napoleon']
napoleon_numpy_docstring = False
myst_heading_anchors = 3
exclude_patterns = ['_build', '_generated']

html_theme = 'furo'
html_title = 'mia_bench'

# The API reference is read from the source without importing it. API_SECTIONS below lists the classes to show, so no
# page is generated per module, and members without a docstring are left out.
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

# For the same reason, a method that overrides an ancestor's keeps its own docstring, even if it is empty.
import autoapi._parser
assert hasattr(autoapi._parser.Parser, '_resolve_inheritance'), 'autoapi changed: update the docstring patches'
_resolve_inheritance = autoapi._parser.Parser._resolve_inheritance


def resolve_inheritance_keeping_own_docstrings(self, *mro_data):
    # Called with the class alone first, which lets overloads share a docstring, then with its ancestors.
    if len(mro_data) == 1:
        return _resolve_inheritance(self, *mro_data)
    own_docstrings = [(child, child['doc']) for child in mro_data[0]['children']]
    children = list(_resolve_inheritance(self, *mro_data))
    for child, docstring in own_docstrings:
        child['doc'] = docstring
    return children


autoapi._parser.Parser._resolve_inheritance = resolve_inheritance_keeping_own_docstrings


def skip_undocumented_members(app, what, name, obj, skip, options):
    # Members without a docstring, such as instance attributes, are left out of the listed classes. The directives of
    # the API pages go through autodoc, which would count the docstring of autoapi's wrapper object as the member's.
    docstring = getattr(obj, 'docstring', None)
    return skip or docstring == ''


def note_class_source(app, docname, source):
    # autoapi does not record the module behind a class page, so the page would keep the docstrings of the build that
    # first read it.
    if docname in API_SOURCES:
        app.env.note_dependency(API_SOURCES[docname], docname=docname)


def setup(app):
    app.connect('autodoc-skip-member', skip_undocumented_members)
    app.connect('source-read', note_class_source)


# ---- Tables generated from the code, so that the pages cannot drift from it ----
DATASET_NAMES = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100', 'purchase': 'Purchase', 'texas': 'Texas'}


def write_if_changed(path: pathlib.Path, text: str):
    # An unchanged file keeps its timestamp, so Sphinx does not reread the pages that use it and sphinx-autobuild does
    # not start another build.
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def write_generated(name: str, text: str):
    # Pages include these files with {include}; they are rebuilt on every build and not committed.
    folder = DOCS / '_generated'
    folder.mkdir(exist_ok=True)
    write_if_changed(folder / name, text)


def hamp_parameters():
    from src.utils.variables import HAMP_PAPER_PARAMETERS, HAMP_FALLBACK_DATASET
    rows = [f'| {DATASET_NAMES.get(dataset, dataset)} | {gamma} | {alpha} |' for dataset, (gamma, alpha) in HAMP_PAPER_PARAMETERS.items()]
    fallback = DATASET_NAMES.get(HAMP_FALLBACK_DATASET, HAMP_FALLBACK_DATASET)
    return '\n'.join(['| Dataset | gamma | alpha |', '|---|---|---|', *rows, '',
                      f'Other datasets use the {fallback} values.', ''])


write_generated('hamp_parameters.md', hamp_parameters())


def ldl_parameters():
    from src.utils.variables import LDL_CODE_N_QUERIES, LDL_PAPER_FLIP_PROBS
    rows = [f'| {DATASET_NAMES.get(dataset, dataset)} | bit flips | {LDL_CODE_N_QUERIES["bernoulli"]} | {flip_prob} |'
            for dataset, flip_prob in LDL_PAPER_FLIP_PROBS.items()]
    return '\n'.join(['| Dataset | Noise | Copies per query | Flip probability |', '|---|---|---|---|', *rows,
                      f'| Other datasets | Gaussian | {LDL_CODE_N_QUERIES["normal"]} | |', ''])


write_generated('ldl_parameters.md', ldl_parameters())


def purifier_parameters():
    from src.utils.variables import PURIFIER_PAPER_REFORMER, PURIFIER_FALLBACK_DATASET
    names = ['reformer_hidden_dim', 'reformer_latent_dim', 'reformer_epochs', 'reformer_lr']
    rows = ['| ' + ' | '.join([DATASET_NAMES.get(dataset, dataset), *[str(values[name]) for name in names]]) + ' |'
            for dataset, values in PURIFIER_PAPER_REFORMER.items()]
    fallback = DATASET_NAMES.get(PURIFIER_FALLBACK_DATASET, PURIFIER_FALLBACK_DATASET)
    return '\n'.join(['| Dataset | h | Latent size | Epochs | Learning rate |', '|---|---|---|---|---|', *rows, '',
                      f'Other datasets use the {fallback} values.', ''])


write_generated('purifier_parameters.md', purifier_parameters())


def mist_parameters():
    from src.utils.variables import MIST_PAPER_PARAMETERS, MIST_FALLBACK_DATASET
    rows = [f'| {DATASET_NAMES.get(dataset, dataset)} | {values["num_submodels"]} | {values["lmbd"]} |'
            for dataset, values in MIST_PAPER_PARAMETERS.items()]
    fallback = DATASET_NAMES.get(MIST_FALLBACK_DATASET, MIST_FALLBACK_DATASET)
    return '\n'.join(['| Dataset | Sub-models C | λ |', '|---|---|---|', *rows, '',
                      f'Other datasets use the {fallback} values.', ''])


write_generated('mist_parameters.md', mist_parameters())


def format_default(value) -> str:
    # Defaults as they are typed on the command line.
    if value is None:
        return 'unset'
    if isinstance(value, bool):
        return 'on' if value else 'off'
    if isinstance(value, float):
        # Keep the decimal point of whole numbers, so that 1.0 does not read as an integer option.
        text = f'{value:g}'
        return text if any(char in text for char in '.en') else f'{text}.0'
    if isinstance(value, (list, tuple)):
        return ' '.join(format_default(item) for item in value)
    return str(value)


def memguard_parameters():
    from src.utils.variables import MEMGUARD_PAPER_SETTINGS
    meanings = {'c1': 'weight of the membership loss', 'c2': 'weight of the label loss',
                'c3_init': 'first weight of the distortion loss', 'c3_max': 'largest weight of the distortion loss',
                'max_iter': 'gradient steps for each weight of the distortion loss',
                'step_size': 'step size β of the normalized gradient steps'}
    assert set(meanings) == set(MEMGUARD_PAPER_SETTINGS), 'describe every MemGuard setting in memguard_parameters'
    rows = [f'| `{name}` | {format_default(value)} | {meanings[name]} |'
            for name, value in MEMGUARD_PAPER_SETTINGS.items()]
    return '\n'.join(['| Setting | Value | Meaning |', '|---|---|---|', *rows, ''])


write_generated('memguard_parameters.md', memguard_parameters())


def option_table(prefixes: list) -> str:
    from src.utils.settings import build_parser
    actions = [action for action in build_parser()._actions
               if action.option_strings and action.dest.startswith(tuple(prefixes))]
    for prefix in prefixes:
        assert any(action.dest.startswith(prefix) for action in actions), f'no option name starts with {prefix}'
    rows = []
    for action in actions:
        assert action.help, f'{action.option_strings[0]} has no help string'
        # A "|" would end the table cell.
        assert '|' not in action.help, f'the help string of {action.option_strings[0]} contains "|"'
        description = action.help % vars(action) if '%(' in action.help else action.help
        if action.choices:
            description += ' (one of ' + ', '.join(f'`{choice}`' for choice in action.choices) + ')'
        rows.append(f'| `{action.option_strings[0]}` | `{format_default(action.default)}` | {description} |')
    return '\n'.join(['| Option | Default | Description |', '|---|---|---|', *rows, ''])


# The command-line options listed in the Parameters section of each method's page: the page's name and the prefixes
# of the options' names.
METHOD_OPTIONS = {'hamp': ['defender_hamp_'], 'ldl': ['defender_ldl_'], 'purifier': ['defender_purifier_'],
                  'memguard': ['defender_mem_guard_'], 'mist': ['defender_mist_', 'defender_mixup_alpha'],
                  'selena': ['defender_selena_']}

for name, prefixes in METHOD_OPTIONS.items():
    write_generated(f'options_{name}.md', option_table(prefixes))


# ---- API reference: one page per class, generated into reference/api/<section>/ and not committed ----
# The classes of each section, in the order of its table. Each section has its page, reference/api/<section>.md.
# Helper classes, such as HampLoss, are left out.
API_SECTIONS = {
    'extending': [
        'src.mia.defenses.base.BaseDefender',
        'src.mia.defenses.base.DefendedModel',
        'src.mia.attacks.base_mia.BaseMIA',
        'src.trainer.train_manager.TrainManager',
    ],
    'defenses': [
        'src.mia.defenses.vanilla.VanillaDefender',
        'src.mia.defenses.training.differential_privacy.DifferentialPrivacyDefender',
        'src.mia.defenses.training.mixup.MixupDefender',
        'src.mia.defenses.training.data_augmentation.DataAugmentationDefender',
        'src.mia.defenses.training.adversarial_regularization.AdvRegDefender',
        'src.mia.defenses.training.relax_loss.RelaxLossDefender',
        'src.mia.defenses.training.selena.SelenaDefender',
        'src.mia.defenses.training.mist.MistDefender',
        'src.mia.defenses.training.weigthed_smoothing.WeightedSmoothingDefender',
        'src.mia.defenses.training.mmd.MmdDefender',
        'src.mia.defenses.post_hoc.mem_guard.MemGuardDefender',
        'src.mia.defenses.post_hoc.ldl.LdlDefender',
        'src.mia.defenses.post_hoc.purifier.PurifierDefender',
        'src.mia.defenses.hybrid.hamp.HampDefender',
    ],
    'attacks': [
        'src.mia.attacks.black_box.lira.LiRA',
        'src.mia.attacks.black_box.robust.RobustMIA',
        'src.mia.attacks.black_box.attack_p.AttackPMIA',
        'src.mia.attacks.black_box.attack_r.AttackRMIA',
        'src.mia.attacks.black_box.quantile.QuantileMIA',
        'src.mia.attacks.black_box.neural.NeuralMIA',
        'src.mia.attacks.label_only.noise_robustness.NoiseRobustnessMIA',
        'src.mia.attacks.label_only.supervised_boundary.SupervisedBoundaryMIA',
        'src.mia.attacks.label_only.unsupervised_boundary.UnsupervisedBoundaryMIA',
        'src.mia.attacks.label_only.transfer.TransferMIA',
        'src.mia.attacks.label_only.oslo.OsloMIA',
        'src.mia.attacks.label_only.dh_attack.DHAttack',
        'src.mia.attacks.label_only.yoqo.Yoqo',
    ],
}


def class_source(path: str) -> pathlib.Path:
    # The file of the class's module.
    source = REPO / (path.rsplit('.', 1)[0].replace('.', '/') + '.py')
    assert source.exists(), f'{path}: no module {source}'
    return source


def class_summary(path: str) -> str:
    # The first paragraph of the class's docstring, read from the source like the rest of the API reference.
    name = path.rsplit('.', 1)[1]
    source = class_source(path)
    node = next((node for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.ClassDef) and node.name == name), None)
    assert node, f'{path}: no class {name} in {source}'
    docstring = ast.get_docstring(node)
    assert docstring, f'{path} has no docstring'
    summary = ' '.join(docstring.split('\n\n')[0].split())
    # A "|" would end the table cell.
    assert '|' not in summary, f'the summary of {path} contains "|"'
    return summary


# The module file of each class page, by page name.
API_SOURCES = {}


def api_pages():
    root = DOCS / 'reference' / 'api'
    # The folders are generated, so a removed section or class leaves no page behind outside every toctree.
    for folder in root.iterdir():
        if folder.is_dir() and folder.name not in API_SECTIONS:
            shutil.rmtree(folder)
    for section, paths in API_SECTIONS.items():
        names = [path.rsplit('.', 1)[1] for path in paths]
        assert len(set(names)) == len(names), f'two classes of the {section} section have the same name'
        pages = {f'{name}.md': '\n'.join(['% Generated from API_SECTIONS by docs/conf.py: edits here are lost.', '',
                                         f'# {name}', '', '```{eval-rst}', f'.. autoapiclass:: {path}',
                                         '   :members:', '   :show-inheritance:', '```', ''])
                 for name, path in zip(names, paths)}
        API_SOURCES.update({f'reference/api/{section}/{name}': class_source(path) for name, path in zip(names, paths)})
        folder = root / section
        folder.mkdir(exist_ok=True)
        for page in folder.iterdir():
            if page.name not in pages:
                page.unlink()
        for page, text in pages.items():
            write_if_changed(folder / page, text)
        rows = [f'| {{py:class}}`~{path}` | {class_summary(path)} |' for path in paths]
        toctree = ['```{toctree}', ':hidden:', '', *[f'{section}/{name}' for name in names], '```']
        write_generated(f'api_{section}.md', '\n'.join(['| Class | Summary |', '|---|---|', *rows, '', *toctree, '']))


api_pages()
