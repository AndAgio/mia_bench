# HAMP

HAMP ([Chen and Pattabiraman, NDSS 2024](https://arxiv.org/abs/2307.01610)) makes the model less confident on its
training data. It trains on high-entropy soft labels with an entropy regularizer, and at inference it replaces the
scores of each output with those of a random input, keeping the order of the classes. Since the order never changes,
label-only attacks only see the effect of training. The authors' code is
[here](https://github.com/DependableSystemsLab/MIA_defense_HAMP).

Each query draws a new random input, so the same sample gets different scores every time it is queried, and every
query runs the model twice.

## Usage

```bash
python run.py --defender_mode hamp --dataset cifar10 --attacker_mode lira
```

`hamp` (or `hamp_full`) applies both parts. `hamp_train` only changes training, and `hamp_test` trains normally and
only replaces the outputs.

`neural_feat` reads the model's internal features, which the output replacement does not change: against `hamp` it
measures only the training part, and against `hamp_test` the undefended model.

The defense is {py:class}`~src.mia.defenses.hybrid.hamp.HampDefender`.

## Parameters

`--defender_hamp_gamma` (between 0 and 1) sets how soft the labels are: their entropy is gamma times the maximum.
`--defender_hamp_alpha` (0 or more) sets the strength of the entropy regularizer. When left unset, both take the
values of Table III in the paper:

```{include} ../_generated/hamp_parameters.md
```

The log shows a warning if you use other values.

```{include} ../_generated/options_hamp.md
```
