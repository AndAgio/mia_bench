# Quantile regression

Quantile regression ([Bertran et al., NeurIPS 2023](https://arxiv.org/abs/2307.03694)) needs no shadow models of the
target. It trains a single regression model on data the attacker knows to be non-members, to predict for each sample
a high quantile of the target's score on non-members like it. The score is the hinge: the true class's logit minus the
largest other logit. A sample whose score exceeds its predicted quantile at level 1 − α is a member, so that about α
of the non-members are. The authors' code is [here](https://github.com/amazon-science/quantile-mia).

The regression model has one of two heads. The Gaussian head, the default and the paper's choice on CIFAR, predicts a
mean μ and a scale σ for each sample and ranks samples by (s − μ)/σ. The pinball head predicts one threshold for each
level of a grid and ranks samples by how far their score exceeds the threshold at level 1 − α.

## Usage

```bash
python run.py --attacker_mode quantile --dataset cifar10 --defender_mode none
```

Add `--no-attacker_quantile_use_gaussian` for the pinball head. The attack is
{py:class}`~src.mia.attacks.black_box.quantile.QuantileMIA`.

After training, the log reports the share of the held-out non-members whose score exceeds their threshold: the
attacker's own estimate of its false-positive rate at α. If that share is significantly above α, a warning suggests
`--decision_rule calibrated`, which sets the threshold on held-out non-members instead, or the Gaussian head.

## Parameters

`--attacker_quantile_alpha` is α, the false-positive rate the attack's own decisions aim for. The Gaussian head
decides at exactly 1 − α. The pinball head learns a grid: `--attacker_quantile_n` levels from
`--attacker_quantile_low` to `--attacker_quantile_high`, evenly spaced or, with `--attacker_quantile_use_logscale`,
crowded near 1 as in the authors' grid.

```{include} ../_generated/quantile_parameters.md
```

Of the attacker's training options, only `--attacker_model` applies to the regression model. The others train the
shadow models of the other attacks.

```{include} ../_generated/options_quantile.md
```

## Differences from the paper

The main difference is the regression model: the `--attacker_model`, ResNet-18 by default, trained from scratch
rather than the authors' pretrained ConvNeXt-Tiny at 224 pixels, so that the attack needs no download and stays
cheap. The others are small:

- The epoch is chosen on a held-out share of the attacker's data. The authors choose it, with their hyperparameters,
  on the very non-members they evaluate on, which a real attacker could not do.
- Batches of 64, the effective batch the paper's tuning chose, rather than the code's default of 16.
- The Gaussian head ranks samples by (s − μ)/σ, the paper's sweep over α taken at every α; the authors' code samples
  41 levels.
- The pinball head ranks samples by the distance to its 1 − α threshold, since a sweep over its levels gives no
  points at very low false-positive rates.
- The pinball grid always adds levels near 1 and the level 1 − α, so that its thresholds at small α fit better and
  the decisions use the α given.
- The warning on the held-out false-positive rate is ours.
