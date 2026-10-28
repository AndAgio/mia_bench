# HAMP

HAMP ([Chen and Pattabiraman, NDSS 2024](https://arxiv.org/abs/2307.01610)) reduces membership leakage by making the
model less confident on its training data. It trains on high-entropy soft labels with an entropy regularizer, and at
inference it replaces the scores of each output with those of a random input, keeping the order of the classes. Since
the order never changes, label-only attacks only see the effect of training. The authors' code is
[here](https://github.com/DependableSystemsLab/MIA_defense_HAMP).

Each query draws a new random input, so the same sample gets different scores every time it is queried, and every query
runs the model twice.

## Usage

```bash
python run.py --defender_mode hamp --dataset cifar10 --attacker_mode lira
```

`hamp` (or `hamp_full`) applies both parts. `hamp_train` only changes training, and `hamp_test` trains normally and
only replaces the outputs.

## Parameters

`--defender_hamp_gamma` (between 0 and 1) sets how soft the labels are: their entropy is gamma times the maximum.
`--defender_hamp_alpha` (0 or more) sets the strength of the entropy regularizer. When left unset, both take the values of Table III
in the paper:

```{include} ../_generated/hamp_parameters.md
```

The log shows a warning if you use other values.

## Differences from the paper

- The KL term is averaged over the classes instead of summed (Eq. 6). This is what the authors' code does, and the
  paper's parameters were tuned with it.
- Random images are drawn in pixel space and then normalized, because our models take normalized inputs. Purchase and
  Texas features are 0 or 1 with equal probability, as in the paper; News features, which the paper does not cover,
  are uniform in [0, 1].
- With gamma = 1, no soft label satisfies the paper's condition. We stop while the true class is still the most likely
  and log a message.
- On image datasets, the paper keeps a new best model only if validation accuracy improves by at least one point. We
  keep any improvement.
- The paper only considers attacks on the outputs. `neural_feat` reads the trained model's features, which the output
  replacement does not change, so it only measures HAMP's training; report these results as an extension. With
  `hamp_test` the trained model is not defended at all, so `neural_feat` measures the undefended model.
