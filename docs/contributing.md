# Documenting a method

Keep docstrings short: name the paper and point to the method's page. If a line of code makes a choice the paper
doesn't explain, add a comment there.

Each method gets a page in `docs/defenses/` (or `docs/attacks/`) with:

1. a short description, with links to the paper and the authors' code;
2. Usage: an example command;
3. Parameters: what they do and their defaults;
4. Differences from the paper: one bullet each, with the reason.

Add the differences to the [table](fidelity.md) too. If a value is defined in the code, generate its table in
`docs/conf.py` instead of copying it, as `hamp_parameters` does. The table of the method's command-line options is
generated too: add the prefix of their names to `METHOD_OPTIONS` and include `_generated/options_<method>.md`. Add
the method's class to `API_SECTIONS`, which generates its page in the [API reference](reference/api.md).

A defense that changes how the model answers queries returns a `DefendedModel` from `defend_model` and implements
`defend`. If it leaves training unchanged, it sets `inference_only = True`. An attack that reads the model's internal
features overrides `reads_features`. With both, the run log notes that the result measures the undefended model's
features.

To build the docs:

```bash
pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs docs/_build/html
```
