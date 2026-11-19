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

- The regression model is the `--attacker_model`, ResNet-18 by default, trained from scratch at the dataset's
  resolution, not a pretrained ConvNeXt-Tiny at 224 pixels with an MLP head. Pretrained weights need a download, and
  the larger inputs cost far more.
- The epoch is the one with the lowest loss on a held-out share of the attacker's data. The authors choose the epoch,
  and their hyperparameters over 32 trials, by the loss on the CIFAR test split, where they also measure the attack:
  their choice sees the very non-members it is evaluated on. We search no hyperparameters.
- Batches of 64, the effective batch that the paper's tuning chose (App. D), instead of the code's default of 16.
- The pinball head ranks samples by the distance to its 1 − α threshold, not by sweeping the levels. A sweep has no
  point of the ROC curve below the false-positive rate of its top level: on CIFAR-10, the 0.9999 level let 0.64% of
  the non-members through, so the sweep had no point at 0.1%.
- The pinball grid always adds levels near 1, and the level 1 − α, to the grid of the options. The loss averages over
  the levels, so more levels near 1 fit that tail better: at α = 0.01 on CIFAR-10, the false-positive rate fell from
  4.8% with 100 evenly spaced levels to 1.8%, as low as with the authors' grid. The level 1 − α lets the decisions use
  the α given instead of the nearest level.
- The Gaussian head ranks samples by (s − μ)/σ, the paper's ROC curve "by varying α", taken at every α. The authors'
  code evaluates 41 levels.
- The warning on the held-out false-positive rate is ours.
